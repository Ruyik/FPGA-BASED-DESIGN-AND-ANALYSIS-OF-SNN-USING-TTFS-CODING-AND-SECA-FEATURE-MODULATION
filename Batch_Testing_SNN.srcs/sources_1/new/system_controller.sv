`timescale 1ns / 1ps

module system_controller #(
    parameter TIME_STEP = 32
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        early_stop,    // Early termination flag from output layer
    output logic        step_pulse,    // Sent to Encoders to advance time
    output logic        done,
    output logic        busy,
    output logic [4:0] time_counter
);

    // Streamlined Parallel State Diagram
    typedef enum logic [1:0] {IDLE=2'b00, RUN=2'b01, DONE_STATE=2'b11} state_t;
    state_t state, next_state;

    // Internal tracker flag to manage the 1-cycle latency pipeline pre-charge fill
    logic pipeline_filled;

    // --- State Register & Pipeline Latency Tracker ---
    // FIXED: Upgraded to pure synchronous logic to prevent LUTAR-1 timing anomalies
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state           <= IDLE;
            pipeline_filled <= 1'b0;
        end else begin
            state           <= next_state;
            
            // Manage the pipeline register delay flag when moving to the RUN state
            if (state == IDLE) begin
                pipeline_filled <= 1'b0;
            end else if (state == RUN) begin
                pipeline_filled <= 1'b1; // Latches high after exactly 1 cycle
            end
        end
    end

    // --- Time Counter Handling ---
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            time_counter <= 0;
        end else begin
            if (state == IDLE) begin
                time_counter <= 0;
            end else if (state == RUN && step_pulse) begin
                // FIXED: Wait 1 cycle for the hidden_subsystem spike barrier 
                // registers to fill up before advancing the systemic tracking timeline
                if (!pipeline_filled) begin
                    time_counter <= 0;
                end else begin
                    time_counter <= time_counter + 1;
                end
            end
        end
    end

    // --- Combinatorial FSM Control Trees ---
    always_comb begin
        next_state = state;
        step_pulse = 0;
        
        case(state)
            IDLE: begin
                if (start) next_state = RUN;
            end
            
            RUN: begin
                if (early_stop) begin
                    next_state = DONE_STATE; // Immediate early termination cut-off
                end else begin
                    step_pulse = 1; // Advance the encoder timeline every single clock cycle
                    
                    // Stop once the tracking counter hits its parameter step limit
                    if (time_counter == TIME_STEP - 1) begin
                        next_state = DONE_STATE;
                    end
                end
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end

    // Output Status Flags
    assign busy = (state != IDLE);
    assign done = (state == DONE_STATE);

endmodule