`timescale 1ns / 1ps

module snn_top #(
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
    output logic        early_termination,
    output logic signed [31:0] v_mon [OUT]    // Monitors classifier voltages in testbench
);

    // --- Global Interconnect Control Wires ---
    logic [4:0] fsm_time_step; 
    logic        step_strobe;
    // Connects the classifier output back to the controller
    
    // Matched 9-bit scale tracking bus to cleanly link SECA and your updated Output Layer
    logic [8:0]  seca_scales [H]; 
    
    // --- Hidden Layer Interconnect Bus ---
    logic [H-1:0] hidden_spikes_bus;
    

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
    // 3. SECA MODULE: The "Attention" (Matches your rolled-back 76% version)
    // =========================================================================
    seca_module #(
        .H(H)
    ) attention_inst (
        .clk(clk),
        .rst_n(rst_n),
        .enable_count(busy),         
        .hidden_spikes(hidden_spikes_bus),
        .clear_counts(start),  
        .time_step(fsm_time_step),    
        .scale_factors(seca_scales)
    );

    // =========================================================================
    // 4. OUTPUT LAYER CLASSIFIER (Port & Parameter Aligned)
    // =========================================================================
    output_layer_TTFS #(
        .H(H),       // FIX: Matched parameter name to your output_layer definition
        .OUT(OUT),   // FIX: Matched parameter name to your output_layer definition
        .TMAX(TMAX)  // FIX: Matched parameter name to your output_layer definition
    ) output_inst (
        .clk(clk),
        .rst_n(rst_n),
        .hidden_spikes(hidden_spikes_bus),
        .scale_factors(seca_scales), // Connected directly to the full 9-bit scale array
        .time_step(fsm_time_step),   
        .clear_v(start),             
        .v_out(v_mon),
        .prediction(prediction),
        .done_prediction(early_termination) // Pulses high to instantly halt the FSM loop
    );

endmodule