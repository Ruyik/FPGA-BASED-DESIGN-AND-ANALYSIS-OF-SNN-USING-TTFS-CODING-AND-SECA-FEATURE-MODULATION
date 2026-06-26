`timescale 1ns / 1ps

module snn_top_tb_no_SECA();

    // 1. Parameters
    localparam H = 12;
    localparam OUT = 3;
    localparam TMAX = 31;
    localparam IMG_PIXELS = 196;
    localparam TOTAL_IMAGES = 300; 

    // 2. Signals
    logic clk, rst_n, start;
    logic [7:0] pixel_values [IMG_PIXELS];
    logic [OUT-1:0] prediction;
    logic done, busy;
    logic signed [31:0] v_mon [OUT];

    // 3. Batch Memories
    logic [7:0] full_test_set [0:(TOTAL_IMAGES * IMG_PIXELS)-1]; 
    logic [3:0] golden_labels [0:TOTAL_IMAGES-1];

    // 4. Debugging Counters
    int confusion_matrix [3][3];
    int pass_count = 0;
    int total_tested = 0;
    
    // --- LATENCY TRACKING VARIABLES ---
    longint total_latency_cycles = 0; // Cumulative cycles for all images
    int current_sample_cycles = 0;    // Cycles for the current image

    // 5. DUT Instance
    //snn_top #(H, OUT, TMAX) uut (.*);
    snn_top_no_seca #(H, OUT, TMAX) uut (.*);
    
    // 6. Clock (10ns Period = 100MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    // --- Updated Task: Run Inference with Latency Tracking ---
    task run_inference(input int img_idx);
        begin
            current_sample_cycles = 0; // Reset for this image

            // Load pixels
            for (int p = 0; p < IMG_PIXELS; p++) begin
                pixel_values[p] = full_test_set[img_idx * IMG_PIXELS + p];
            end

            // Print Sample Header before execution starts
            $display("[Sample %0d] [Real Digit: %0d] Processing...", img_idx, golden_labels[img_idx]);

            // Trigger FSM
            @(posedge clk); start = 1;
            @(posedge clk); start = 0;

            wait(busy);
            
            // Count cycles while the SNN is busy
            while (busy && !done) begin
                @(posedge clk);
                current_sample_cycles++;
            end
            
            total_latency_cycles += current_sample_cycles;

            repeat(5) @(posedge clk); // Settle time

            // Print Threshold/Prediction Status immediately after circuit finishes
            $display("    -> Finished at cycle %0d | Pred: %0d, Real: %0d %s", 
                     current_sample_cycles, prediction, golden_labels[img_idx],
                     (prediction == golden_labels[img_idx]) ? "[MATCH]" : "[MISMATCH!]");

            // Record into Confusion Matrix
            if (prediction < 3) begin
                confusion_matrix[golden_labels[img_idx]][prediction]++;
                if (prediction == golden_labels[img_idx]) pass_count++;
                total_tested++;
            end
        end
    endtask

    // 7. Main Control logic
    initial begin
        // Reset Matrix
        for(int r=0; r<3; r++) for(int c=0; c<3; c++) confusion_matrix[r][c] = 0;

        $display("[TB] Loading 300-sample batch data...");
        $readmemh("test_batch_300.mem", full_test_set);
        $readmemh("labels_300.mem", golden_labels);
        
        rst_n = 0; start = 0; #100; rst_n = 1; #50;

        $display("\n>>> STARTING BATCH TEST (300 SAMPLES)...");
        
        for (int i=0; i < TOTAL_IMAGES; i++) begin
             // Added visual separator blocks between different digits to break up the dense logs
             if (i == 0)       $display("\n--- DIGIT 0 SAMPLES ---");
             else if (i == 100) $display("\n--- DIGIT 1 SAMPLES ---");
             else if (i == 200) $display("\n--- DIGIT 2 SAMPLES ---");

             run_inference(i); 
        end

        // --- FINAL RESULTS ---
        $display("\n=============================================");
        $display("FINAL BATCH DEBUG SUMMARY (N=%0d)", total_tested);
        $display("=============================================");
        
        // RECOVERED CONFUSION MATRIX DISPLAY
        $display("Actual \\ Pred |  Digit 0  |  Digit 1  |  Digit 2  |");
        $display("---------------------------------------------");
        for (int r = 0; r < 3; r++) begin
            $display("  Digit %0d      |    %3d    |    %3d    |    %3d    |", 
                     r, confusion_matrix[r][0], confusion_matrix[r][1], confusion_matrix[r][2]);
        end
        
        $display("---------------------------------------------");
        $display("Total Hardware Accuracy With SECA: %0.2f%% (%0d/%0d)", 
                 (real'(pass_count)/total_tested)*100.0, pass_count, total_tested);
        
        // --- PERFORMANCE METRIC ---
        $display("Average Latency: %0.2f clock cycles", real'(total_latency_cycles)/total_tested);
        $display("=============================================");
        
        $finish;
    end

endmodule