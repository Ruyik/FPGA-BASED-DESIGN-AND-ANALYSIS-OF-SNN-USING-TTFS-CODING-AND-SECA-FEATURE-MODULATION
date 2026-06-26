
module snn_demo_tb();


    // 1. Parameters
    localparam H = 12;
    localparam OUT = 3;
    localparam TMAX = 31;
    localparam IMG_PIXELS = 196;
    localparam TOTAL_IMAGES = 3; 



    // 2. Signals
    logic clk, rst_n, start;
    logic [7:0] pixel_values [IMG_PIXELS];
    logic [OUT-1:0] prediction;
    logic done, busy;
    logic signed [31:0] v_mon [OUT];

    // 3. Batch Memories
    logic [7:0] full_test_set [0:(TOTAL_IMAGES * IMG_PIXELS)-1]; 
    logic [3:0] golden_labels [0:TOTAL_IMAGES-1];
    logic signed [31:0] decay_v [OUT];

    // 4. Debugging Counters
    int confusion_matrix [3][3];
    int pass_count = 0;
    int total_tested = 0;

    // --- LATENCY TRACKING VARIABLES ---
    longint total_latency_cycles = 0; // Cumulative cycles for all images
    int current_sample_cycles = 0;    // Cycles for the current image

    logic signed [31:0] hw_v_prev [OUT] = '{0,0,0};
    logic is_demo_sample;
    logic signed [31:0] hw_v0_prev = 0;
    logic signed [31:0] spike_impact = 0;
    // 5. DUT Instance
    // Replace uut (.*) with this:
    snn_top #(H, OUT, TMAX) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .pixel_values(pixel_values),
        .prediction(prediction),
        .done(done),
        .busy(busy),
        .early_termination(early_termination), // Map your TB signal here
        .v_mon(v_mon)
    );

 
    // 6. Clock (10ns Period = 100MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    // --- Updated Task: Run Inference with Latency Tracking ---
    // Enhanced Task with internal monitoring
    task run_inference(input int img_idx);
        begin
            current_sample_cycles = 0;
            is_demo_sample = (img_idx >= 0 && img_idx < TOTAL_IMAGES);
            
            // Reset virtual monitors
            for(int j=0; j<OUT; j++) begin
                decay_v[j] = 0;
                hw_v_prev[j] = 0;
            end
    
            for (int p = 0; p < IMG_PIXELS; p++) 
                pixel_values[p] = full_test_set[img_idx * IMG_PIXELS + p];
    
            @(posedge clk); start = 1;
            @(posedge clk); start = 0;
    
            wait(busy);
            
            while (busy && !done) begin
                @(posedge clk);
                current_sample_cycles++;
                
                // --- VIRTUAL DECAY LOGIC (More aggressive for demo: 200/256) ---
                for(int j=0; j<OUT; j++) begin
                    // Use a faster decay (200) to make it visually obvious in the log
                    decay_v[j] = ((decay_v[j] * 32'd252) >>> 8) + (uut.v_mon[j] - hw_v_prev[j]);
                    if (decay_v[j] < 0) decay_v[j] = 0;
                    hw_v_prev[j] = uut.v_mon[j]; 
                end
                
                if (is_demo_sample) begin
                    // Formatted to print all 3 neurons clearly
                    $display("T:%0d | V0: %8d | V1: %8d | V2: %8d", 
                            current_sample_cycles, decay_v[0], decay_v[1], decay_v[2]);
                end
            end 
            
            // --- CRITICAL FIX: Ensure these update correctly ---
            total_latency_cycles += current_sample_cycles;
            
            // Record results for the summary
            if (prediction < 3) begin
                confusion_matrix[golden_labels[img_idx]][prediction]++;
                if (prediction == golden_labels[img_idx]) pass_count++;
                total_tested++;
            end
            $display("    -> Finished at cycle %0d | Pred: %0d, Real: %0d", 
                     current_sample_cycles, prediction, golden_labels[img_idx]);
        end
    endtask

    // 7. Main Control logic
    initial begin
        // Reset Matrix
        for(int r=0; r<3; r++) for(int c=0; c<3; c++) confusion_matrix[r][c] = 0;
        $display("[TB] Loading 300-sample batch data...");
        $readmemh("demo_samples_3.mem", full_test_set);
        $readmemh("demo_labels_3.mem", golden_labels);      
        rst_n = 0; start = 0; #100; rst_n = 1; #50;

        $display("\n>>> STARTING BATCH TEST (3 SAMPLES)...");      
        for (int i=0; i < TOTAL_IMAGES; i++) begin
             // Added visual separator blocks between different digits to break up the dense logs
             if (i == 0)       $display("\n--- DIGIT 0 SAMPLES ---");
             else if (i == 1) $display("\n--- DIGIT 1 SAMPLES ---");
             else if (i == 2) $display("\n--- DIGIT 2 SAMPLES ---");
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