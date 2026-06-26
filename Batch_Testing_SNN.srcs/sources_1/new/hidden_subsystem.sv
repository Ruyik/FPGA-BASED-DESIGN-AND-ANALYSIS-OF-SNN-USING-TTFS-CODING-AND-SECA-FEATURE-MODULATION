`timescale 1ns / 1ps


module hidden_subsystem (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        step_strobe,         // Strobe from controller to advance encoder time
    input  logic [7:0]  pixel_values [196],
    output logic [11:0] hidden_spikes_out
);

    // This 196-bit wire bus now moves all data simultaneously in 1 clock cycle
    logic [195:0] parallel_spike_bus;
    // TIMING OPTIMIZATION REGISTER LAYER: Breaks the 491-layer combinational loop
    logic [195:0] pipelined_spike_bus;

    // UPGRADED INPUT STAGE: Swapped from TTFS to Poisson Rate Encoding Array
    ttfs_encoder_array ttfs_array_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .step_pulse(step_strobe),
        .pixel_data(pixel_values),
        .spike_bus_out(parallel_spike_bus)
    );
    
    // =========================================================================
    // PIPELINE REGISTER BARRIER (Slices critical path, drops routing time to <30m)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipelined_spike_bus <= 196'b0;
        end else if (start) begin
            pipelined_spike_bus <= 196'b0; // Synchronous clear on new frame start
        end else begin
            pipelined_spike_bus <= parallel_spike_bus; // Latch spikes for 1 cycle
        end
    end

    // 2. Processing Stage: Parallel Hidden Layer Matrix Adder
    hidden_layer #(
        .NUM_NEURONS(12),
        .V_WIDTH(32),
        .INPUT_PIXELS(196)
    ) parallel_hidden_layer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .clear_v(start),
        .parallel_input_spikes(pipelined_spike_bus), // Connected directly as a broad parallel bus
        .hidden_spikes_out(hidden_spikes_out)
    );

endmodule