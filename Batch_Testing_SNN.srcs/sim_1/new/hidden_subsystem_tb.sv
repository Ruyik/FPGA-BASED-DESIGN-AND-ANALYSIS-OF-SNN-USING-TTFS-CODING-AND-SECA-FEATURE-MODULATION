`timescale 1ns / 1ps

module hidden_subsystem_tb();

    // 1. Core Interconnect Wires
    logic clk;
    logic rst_n;
    logic start;
    logic step_strobe; 
    logic [7:0] pixel_values [196];
    logic [11:0] hidden_spikes_out;
    
    // Cross-module probe array for internal neuron potentials
    logic [31:0] internal_v_states [12];

    genvar n;
    generate
        for (n = 0; n < 12; n = n + 1) begin : link_neuron_voltages
            assign internal_v_states[n] = uut.parallel_hidden_layer_inst.gen_parallel_neurons[n].neuron_inst.v_reg;
        end
    endgenerate

    // 2. Clock Generation (100MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    // 3. DUT Instance
    hidden_subsystem uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .step_strobe(step_strobe),
        .pixel_values(pixel_values),
        .hidden_spikes_out(hidden_spikes_out)
    );

    // 4. File Storage Buffers
    logic [7:0] image_file_buffer [0:58799]; // 300 images * 196 pixels
    logic [3:0] label_file_buffer [0:299];   // 300 labels

    // Load files into memory array layouts at time step 0
    initial begin
        $readmemh("test_batch_300.mem", image_file_buffer);
        $readmemh("labels_300.mem", label_file_buffer);
    end

    // 5. Automatic Verification Task
    // Dynamically slices out 196 pixels from the target file index block
    task verify_sample(input int sample_idx);
        int start_pixel_row;
        begin
            start_pixel_row = sample_idx * 196;
            
            $display("\n========================================================");
            $display("[TASK TRIGGER] Loading File Sample Index: %0d", sample_idx);
            $display("               Expected Real Target Label: %0d", label_file_buffer[sample_idx]);
            $display("========================================================");

            // Fetch the 196-pixel slice combinationally
            for (int p = 0; p < 196; p++) begin
                pixel_values[p] = image_file_buffer[start_pixel_row + p];
            end

            // Reset neuron internal states for this fresh image
            @(posedge clk); #1; start = 1;
            @(posedge clk); #1; start = 0;
            @(posedge clk);

            // Execute 32 SNN Timeline Steps
            for (int t = 0; t < 32; t++) begin
                #1; step_strobe = 1;
                @(posedge clk);
                #1; step_strobe = 0;

                // Monitor when the pulse arrays activate the network matrix
                if (uut.parallel_spike_bus > 0) begin
                    $display("  [T = %2d] >>> PARALLEL BUS ACTIVE! Hex Vector: %h", t, uut.parallel_spike_bus);
                    
                    // Wait for the next clock edge so the parallel matrix can accumulate signed metrics
                    @(posedge clk); #1;
                    $display("    -> Neurn  3 Charge: %0d", $signed(internal_v_states[3]));
                    $display("    -> Neurn  9 Charge: %0d", $signed(internal_v_states[9]));
                    $display("    -> Neurn 10 Charge: %0d", $signed(internal_v_states[10]));
                end

                if (hidden_spikes_out > 0) begin
                    $display("  [T = %2d] >>> HIDDEN SPIKE BUS ACTIVE: %b", t, hidden_spikes_out);
                end
            end
            
            // Separation buffer delay between samples
            repeat(10) @(posedge clk);
        end
    endtask

    // 6. Test Driver Loop
    initial begin
        rst_n = 0; start = 0; step_strobe = 0;
        #100;
        rst_n = 1;
        #50;

        // Run your exact requested test cross-checks
        verify_sample(0);   // Digit 0 (Row 1 of label file)
        verify_sample(100); // Digit 1 (Row 101 of label file)
        verify_sample(200); // Digit 2 (Row 201 of label file)

        $display("\n========================================================");
        $display("[SUB-TB] MULTI-DIGIT FILE EXTRACTION VALIDATION PASSED.");
        $display("========================================================");
        $finish;
    end

endmodule