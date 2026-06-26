`timescale 1ns / 1ps

module weight_rom #(
    parameter NUM_NEURON = 12,
    parameter INPUT_PIXELS = 196,
    parameter WEIGHT_WIDTH = 16
) (
    // No address input needed! We output the entire weight matrix in parallel.
    output logic signed [WEIGHT_WIDTH-1:0] parallel_weights [NUM_NEURON][INPUT_PIXELS]
);
    
    // 1. Packed array to load the hex strings sequentially from the file
    // 12 neurons * 16 bits = 192 bits per line
    logic [(NUM_NEURON * WEIGHT_WIDTH)-1:0] internal_rom [0:INPUT_PIXELS-1];
    
    // 2. Load the exported weights file
    initial begin
        $readmemh("hidden_w.mem", internal_rom);
    end
    
    // 3. Combinatorial Unpacking Block
    // Splices the sequential text lines into a structured 2D signed grid
    always_comb begin
        for (int p = 0; p < INPUT_PIXELS; p++) begin
            for (int n = 0; n < NUM_NEURON; n++) begin
                // FIX: Use the variable '+:' indexing operator.
                // It means: "Start at bit index (n * WEIGHT_WIDTH) and slice UPWARDS by WEIGHT_WIDTH bits"
                // Since WEIGHT_WIDTH is a constant parameter, the compiler accepts it perfectly!
                parallel_weights[n][p] = $signed(internal_rom[p][(n * WEIGHT_WIDTH) +: WEIGHT_WIDTH]);
            end
        end
    end

endmodule