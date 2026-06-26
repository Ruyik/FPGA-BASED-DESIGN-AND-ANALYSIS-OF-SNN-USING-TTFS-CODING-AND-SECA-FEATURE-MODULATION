`timescale 1ns / 1ps

module snn_top_no_seca #(
    parameter H = 12,
    parameter OUT = 3,
    parameter TMAX = 31
)(
    input  logic          clk,
    input  logic          rst_n,
    input  logic          start,              // Triggers system evaluation
    input  logic [7:0]    pixel_values [196], // Broad parallel input image
    output logic [OUT-1:0] prediction,
    output logic          done,               // Driven globally by System Controller
    output logic          busy,               // Driven globally by System Controller
    output logic signed [31:0] v_mon [OUT]    // Monitors classifier voltages in testbench
);

    // --- Global Interconnect Control Wires ---
    logic [4:0] fsm_time_step; 
    logic        step_strobe;
    logic        early_termination; // Connects the classifier output back to the controller

    // --- Hidden Layer Interconnect Bus ---
    logic [H-1:0] hidden_spikes_bus;
    
    // Constant Scale Factors (Ablation Configuration: No SECA Module)
    // 8'h80 represents a stable 1.0 weight multiplier in your fixed-point logic
    logic [8:0] constant_scales [H];
    
    always_comb begin
        for(int i=0; i<H; i++) begin
            constant_scales[i] = 8'h80; 
        end
    end

    // =========================================================================
    // 1. GLOBAL SYSTEM CONTROLLER (Parallel Upgraded)
    // =========================================================================
    system_controller #(
        .TIME_STEP(32)
    ) global_controller_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .early_stop(early_termination), 
        .step_pulse(step_strobe),
        .time_counter(fsm_time_step),
        .busy(busy),
        .done(done)
    );

    // =========================================================================
    // 2. HIDDEN SUBSYSTEM (Fully Validated Parallel Core processing unit)
    // =========================================================================
    hidden_subsystem hidden_sys_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .step_strobe(step_strobe), // Receives a timing strobe every single clock cycle
        .pixel_values(pixel_values),
        .hidden_spikes_out(hidden_spikes_bus)
    );

    // =========================================================================
    // 3. OUTPUT LAYER CLASSIFIER (Monitors Accumulations & Handles Halted Predictions)
    // =========================================================================
    output_layer_TTFS #(
        .NUM_NEURONS(H), 
        .NUM_CLASSES(OUT), 
        .TIME_STEPS(TMAX+1)
    ) output_inst (
        .clk(clk),
        .rst_n(rst_n),
        .hidden_spikes(hidden_spikes_bus),
        .scale_factors(constant_scales),
        .time_step(fsm_time_step),   
        .clear_v(start),             
        .v_out(v_mon),
        .prediction(prediction),
        .done_prediction(early_termination) // Pulses high to instantly halt the FSM loop
    );

endmodule