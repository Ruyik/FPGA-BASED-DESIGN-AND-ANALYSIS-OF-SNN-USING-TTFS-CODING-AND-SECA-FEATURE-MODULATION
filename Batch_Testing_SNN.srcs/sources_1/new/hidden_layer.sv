`timescale 1ns / 1ps

module hidden_layer #(
    parameter NUM_NEURONS = 12,
    parameter WEIGHT_WIDTH = 16,
    parameter V_WIDTH = 32,
    parameter INPUT_PIXELS = 196
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clear_v,
    input  logic [INPUT_PIXELS-1:0] parallel_input_spikes, // 196-bit parallel input bus
    output logic [NUM_NEURONS-1:0]  hidden_spikes_out
);
    
    // 2D Matrix Interconnect Array
    logic signed [WEIGHT_WIDTH-1:0] global_weights [NUM_NEURONS][INPUT_PIXELS];

    // 1. Instantiate the updated Parallel ROM
    weight_rom #(
        .NUM_NEURON(NUM_NEURONS),
        .INPUT_PIXELS(INPUT_PIXELS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH)
    ) parallel_rom_inst (
        .parallel_weights(global_weights)
    );

    // 2. Parallel Combinatorial Summation Array
    logic signed [V_WIDTH-1:0] total_neuron_input_charge [NUM_NEURONS];

    always_comb begin
        for (int n = 0; n < NUM_NEURONS; n++) begin
            total_neuron_input_charge[n] = 32'sd0;
            for (int p = 0; p < INPUT_PIXELS; p++) begin
                if (parallel_input_spikes[p]) begin
                    // FIX: Remove manual concat brackets. 
                    // Directly adding the signed 16-bit value to the signed 32-bit register 
                    // forces the compiler to run a flawless automatic sign-extension.
                    total_neuron_input_charge[n] = total_neuron_input_charge[n] + {{16{global_weights[n][p][15]}}, global_weights[n][p]};
                end
            end
        end
    end
    
    // --- SIGNED BIT-VALIDATION MONITOR ---

    
    // 3. Instantiate your 12 parallel LIF Neurons
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : gen_parallel_neurons
            lif_neuron #(.WEIGHT_WIDTH(V_WIDTH), .V_WIDTH(V_WIDTH)) neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .weight(total_neuron_input_charge[i]), // Fed with parallel-summed signed charge
                .spike_in(1'b1),                       // Evaluation trigger pulse
                .clear_v(clear_v),
                .spike_out(hidden_spikes_out[i]),
                .v_state() 
            );
        end
    endgenerate

endmodule