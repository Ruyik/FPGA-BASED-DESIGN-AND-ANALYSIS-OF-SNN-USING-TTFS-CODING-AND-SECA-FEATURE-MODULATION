`timescale 1ns / 1ps

module snn_hardware_wrapper (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,       // Physical Push Button
    input  logic [1:0]  view_sel,    // Slide switches to route voltage monitor paths
    output logic [2:0]  led_pred,    // 3 LEDs -> LED[0]=Digit 0, LED[1]=Digit 1, LED[2]=Digit 2
    output logic        busy_led,    // Status LED showing processing active
    output logic        done_led,    // Routes early stop termination flag to a pin
    output logic [15:0] debug_v_out  // 16-pin bus to probe internal voltages
);

    // Internal signals to interconnect wrapper logic to SNN core
    logic [7:0] internal_pixels [196];
    logic [2:0] prediction;
    logic signed [31:0] v_mon_internal [3];
    logic done;
    
    // Synchronizer signals to capture physical push buttons safely
    logic start_sync_0;
    logic start_sync_1;

    assign done_led = done; 

    // =========================================================================
    // 2-STAGE METASTABILITY SYNCHRONIZER FOR PHYSICAL START BUTTON
    // =========================================================================
    // Cleans up the 'checking no_clock' and asynchronous driver check violations
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_sync_0 <= 1'b0;
            start_sync_1 <= 1'b0;
        end begin
            start_sync_0 <= start;
            start_sync_1 <= start_sync_0; // start_sync_1 is now safe to run core logic
        end
    end

    // =========================================================================
    // HARDWIRED PIXEL FRAMEBUFFER MATRIX
    // =========================================================================
    always_comb begin
        for(int i=0; i<196; i++) begin
            internal_pixels[i] = 8'h00;
        end
        
        // Rows 2 to 12 map out a clean structural Digit "1"
        internal_pixels[34] = 8'hD0; internal_pixels[35] = 8'hD0;
        internal_pixels[48] = 8'hE0; internal_pixels[49] = 8'hE0;
        internal_pixels[62] = 8'hF0; internal_pixels[63] = 8'hF0;
        internal_pixels[76] = 8'hF0; internal_pixels[77] = 8'hF0;
        internal_pixels[90] = 8'hF0; internal_pixels[91] = 8'hF0;
        internal_pixels[104] = 8'hF0; internal_pixels[105] = 8'hF0;
        internal_pixels[118] = 8'hE0; internal_pixels[119] = 8'hE0;
        internal_pixels[132] = 8'hD0; internal_pixels[133] = 8'hD0;
        internal_pixels[146] = 8'hC0; internal_pixels[147] = 8'hC0;
    end

    // =========================================================================
    // CORE SNN HARDWARE SYSTEM INSTANTIATION
    // =========================================================================
    snn_top #(
        .H(12),
        .OUT(3),
        .TMAX(31)
    ) core_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_sync_1), // Fed by safe synchronous register pin
        .pixel_values(internal_pixels),
        .prediction(prediction),
        .done(done),
        .busy(busy_led),
        .v_mon(v_mon_internal)
    );

    // =========================================================================
    // OUTPUT MULTIPLEXER ROUTING BLOCK 
    // =========================================================================
    always_comb begin
        case(view_sel)
            2'd0:    debug_v_out = v_mon_internal[0][15:0]; 
            2'd1:    debug_v_out = v_mon_internal[1][15:0]; 
            2'd2:    debug_v_out = v_mon_internal[2][15:0]; 
            default: debug_v_out = 16'hDEAD;                 
        endcase
    end

    // =========================================================================
    // FIXED: EXPLICIT WINNER-TAKE-ALL OUTPUT CLASS MAPPING (Removes Latch)
    // =========================================================================
    always_comb begin
        case(prediction)
            3'd0:    led_pred = 3'b001; // Digit 0 Active
            3'd1:    led_pred = 3'b010; // Digit 1 Active
            3'd2:    led_pred = 3'b100; // Digit 2 Active
            default: led_pred = 3'b000; // Safe flush fallback
        endcase
    end

endmodule