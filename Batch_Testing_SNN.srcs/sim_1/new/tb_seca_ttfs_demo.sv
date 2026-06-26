`timescale 1ns / 1ps

module tb_seca_ttfs_demo();

    // Global Testbench Controls
    logic clk;
    logic rst_n;
    logic start;
    
    // Arrays to hold the 196 downsampled pixels
    logic [7:0] index3_pixels [0:195];
    
    // Ground Truth Label Configuration
    localparam [2:0] CORRECT_LABEL = 3'd0; // Index 3 is a digit 0
    
    // Core 1 Signals: Baseline (WITHOUT SECA)
    logic [2:0]  pred_no_seca;
    logic        done_no_seca;
    logic        busy_no_seca;
    int          cycles_no_seca;

    // Core 2 Signals: Proposed Architecture (WITH SECA)
    logic [2:0]  pred_with_seca;
    logic        done_with_seca;
    logic        busy_with_seca;
    logic        early_term_with_seca;
    int          cycles_with_seca;
    wire [8:0] seca_scales_monitor [0:11];

    // --- AUTOMATIC MEMORY INITIALIZATION ---
    initial begin
        $readmemh("mnist_index3.mem", index3_pixels);
    end

    // Clock Generator (~3.33 MHz matching your timing constraints)
    initial clk = 0;
    always #150.15 clk = ~clk; 

    // --- UUT 1: Baseline SNN Core WITHOUT SECA ---
    snn_top_no_seca uut_baseline (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .pixel_values(index3_pixels),
        .prediction(pred_no_seca),
        .done(done_no_seca),
        .busy(busy_no_seca),
        .v_mon() 
    );

    // --- UUT 2: Proposed SNN Core WITH SECA ---
    snn_top uut_seca_enhanced (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .pixel_values(index3_pixels),
        .prediction(pred_with_seca),
        .done(done_with_seca),
        .busy(busy_with_seca),
        .early_termination(early_term_with_seca), 
        .v_mon()
    );
    
    generate
        for (genvar i = 0; i < 12; i++) begin : assign_seca_wires
            assign seca_scales_monitor[i] = uut_seca_enhanced.output_inst.scale_factors[i];
        end
    endgenerate

    // Synchronous Baseline Cycle Tracking
    always_ff @(posedge clk) begin
        if (!rst_n || start) begin
            cycles_no_seca <= 0;
        end else if (busy_no_seca && !done_no_seca) begin
            cycles_no_seca <= cycles_no_seca + 1;
        end
    end

    // Synchronous Proposed Cycle Tracking
    always_ff @(posedge clk) begin
        if (!rst_n || start) begin
            cycles_with_seca <= 0;
        end else if (busy_with_seca && !done_with_seca) begin
            cycles_with_seca <= cycles_with_seca + 1;
        end
    end

    // --- EXECUTION STIMULUS WITH TABLE-FORMATTED LOGGING ---
    initial begin
        rst_n = 0; start = 0; #1000;
        rst_n = 1; #500;
        
        // Pulse Start
        @(posedge clk); start = 1;
        @(posedge clk); start = 0;
        
        // Wait for hardware execution to begin
        wait(busy_with_seca);
        
        $display("\n==================================================================");
        $display("       REAL-TIME SECA ATTENTION CHANNEL MODULATION LOG            ");
        $display("==================================================================");
        $display(" Cycle |  N3[D0]  N4[D0]  N5[D0]  N6[D0]  |  N7[D2]  N8[D2]  N9[D2]  N10[D2] N11[D2]");
        $display("-------+----------------------------------+-----------------------------------");
        
        // Loop and print the scale factors as an aligned row on every active cycle
        while (busy_with_seca && !done_with_seca) begin
            @(posedge clk);
            // Only log if the module has left its neutral state
            if (uut_seca_enhanced.output_inst.scale_factors[3] != 128 || uut_seca_enhanced.output_inst.scale_factors[7] != 128) begin
                $display("  #%2d  |   %3d     %3d     %3d     %3d   |   %3d     %3d     %3d     %3d     %3d", 
                         cycles_with_seca,
                         uut_seca_enhanced.output_inst.scale_factors[3],
                         uut_seca_enhanced.output_inst.scale_factors[4],
                         uut_seca_enhanced.output_inst.scale_factors[5],
                         uut_seca_enhanced.output_inst.scale_factors[6],
                         uut_seca_enhanced.output_inst.scale_factors[7],
                         uut_seca_enhanced.output_inst.scale_factors[8],
                         uut_seca_enhanced.output_inst.scale_factors[9],
                         uut_seca_enhanced.output_inst.scale_factors[10],
                         uut_seca_enhanced.output_inst.scale_factors[11]);
            end
        end
        
        // Safe execution barrier for both asynchronously running cores
        wait(done_no_seca);
        wait(done_with_seca);
        repeat(5) @(posedge clk); // Settle pipeline registers
        
        // Final Summary Block
        $display("\n=========================================================");
        $display("   MNIST INDEX 3 TARGETED DEMO VALIDATION REPORT         ");
        $display("=========================================================");
        $display("   Correct Target Label (Ground Truth) : Digit %d", CORRECT_LABEL);
        $display("---------------------------------------------------------");
        $display("   Baseline Core (No SECA) Prediction   : Digit %d", pred_no_seca);
        $display("   Baseline Total Latency               : %0d Cycles %s", 
                 cycles_no_seca, (pred_no_seca == CORRECT_LABEL) ? "[MATCH]" : "[MISMATCH!]");
        $display("---------------------------------------------------------");
        $display("   Proposed Core (With SECA) Prediction : Digit %d", pred_with_seca);
        $display("   Proposed Total Latency               : %0d Cycles %s", 
                 cycles_with_seca, (pred_with_seca == CORRECT_LABEL) ? "[MATCH]" : "[MISMATCH!]");
        $display("   TTFS Early Termination Triggered     : %b", early_term_with_seca);
        $display("---------------------------------------------------------");
       
        
        if ((pred_no_seca != CORRECT_LABEL) && (pred_with_seca == CORRECT_LABEL)) begin
            $display("   DEMO STATUS: SUCCESS! SECA corrected the inference failure.");
        end
        $display("=========================================================");
        $finish;
    end

endmodule