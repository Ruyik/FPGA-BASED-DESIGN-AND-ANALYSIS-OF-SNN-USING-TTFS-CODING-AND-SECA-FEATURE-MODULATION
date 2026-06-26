`timescale 1ns / 1ps

module tb_rate_encoder_array;

    // --- Parameters ---
    localparam NUM_ENCODERS = 196;
    localparam TMAX = 31;
    localparam CLK_PERIOD = 10; // 100MHz clock

    // --- Interface Wires ---
    logic clk;
    logic rst_n;
    logic start;
    logic step_pulse;
    logic [7:0] pixel_data [NUM_ENCODERS];
    logic [NUM_ENCODERS-1:0] spike_bus_out;

    // --- Trackers for Behavioral Analysis ---
    int spike_counts [NUM_ENCODERS];
    int total_sim_cycles;

    // --- UUT Instantiation ---
    rate_encoder_array #(
        .NUM_ENCODERS(NUM_ENCODERS),
        .TMAX(TMAX)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .step_pulse(step_pulse),
        .pixel_data(pixel_data),
        .spike_bus_out(spike_bus_out)
    );

    // --- Clock Generator Engine ---
    always #(CLK_PERIOD/2) clk = ~clk;

    // --- Spike Counting Monitor ---
    always_ff @(posedge clk) begin
        if (start) begin
            for(int i=0; i<NUM_ENCODERS; i++) spike_counts[i] <= 0;
        end else begin
            for(int i=0; i<NUM_ENCODERS; i++) begin
                if (spike_bus_out[i]) begin
                    spike_counts[i] <= spike_counts[i] + 1;
                end
            end
        end
    end

    // --- Main Test Stimulus Loop ---
    initial begin
        // 1. Initialize System State
        clk = 0;
        rst_n = 0;
        start = 0;
        step_pulse = 0;
        total_sim_cycles = 0;
        
        // Populate standard uniform baseline image array
        for(int i=0; i<NUM_ENCODERS; i++) pixel_data[i] = 8'd0;

        #(CLK_PERIOD * 2);
        rst_n = 1; // Release Reset
        #(CLK_PERIOD * 2);

        $display("\n==================================================");
        print_time_context();
        $display("   STARTING HARDWARE RATE ENCODER ARRAY BENCHMARK");
        $display("==================================================");

        // 2. Define Benchmark Receptive Profiles
        // Encoder 0: Bright Pixel (Direct Core Ink Feature) -> High Probability
        pixel_data[0] = 8'd220; 
        // Encoder 1: Mid-tone Pixel (Fractional Border Shade) -> Medium Probability
        pixel_data[1] = 8'd128;
        // Encoder 2: Dark Pixel (Background Backdrop Space)  -> Zero Spikes Expected
        pixel_data[2] = 8'd0;   

        // 3. Trigger FSM Execution Window
        @(posedge clk);
        start = 1; // Assert Start pulse
        @(posedge clk);
        start = 0;

        // 4. Run Timeline Stepping Simulation Matrix
        // Loop through TMAX steps, pulsing step_pulse each step to match your system controller behavior
        for (int t = 0; t <= TMAX; t++) begin
            $display("[TIMELINE TICK] Step T: %0d | Bus Spikes Status: %b %b %b", 
                     t, spike_bus_out[0], spike_bus_out[1], spike_bus_out[2]);
            
            // Wait for 1 clock cycle to process state transitions
            @(posedge clk);
            step_pulse = 1; // Strike Controller step strobe
            @(posedge clk);
            step_pulse = 0;
            total_sim_cycles++;
        end

        #(CLK_PERIOD * 5);

        // 5. Compile & Display Verification Metrics Report
        $display("\n==================================================");
        print_time_context();
        $display("          POISSON RATE BENCHMARK REPORT CARD");
        print_time_context();
        $display("==================================================");
        $display("Pixel Profile [0] (Bright, Input 220) -> Spike Count: %0d / %0d ticks", spike_counts[0], TMAX+1);
        $display("Pixel Profile [1] (Medium, Input 128) -> Spike Count: %0d / %0d ticks", spike_counts[1], TMAX+1);
        $display("Pixel Profile [2] (Dark,   Input 0  ) -> Spike Count: %0d / %0d ticks", spike_counts[2], TMAX+1);
        $display("--------------------------------------------------");
        
        // Behavioral Sanity Checks Validation
        if (spike_counts[0] >= spike_counts[1] && spike_counts[2] == 0) begin
            $display(">>> BENCHMARK STATUS: SUCCESS! VALIDATION MATCHED EXPECTED PROBABILITY DISTRIBUTIONS.");
        end else begin
            $display(">>> BENCHMARK STATUS: FAILED! UNEXPECTED TEMPORAL PROFILE DETECTED.");
        end
        $display("==================================================\n");

        $finish;
    end

    // Dynamic logging helper matching project specifications
    task print_time_context();
        $write("[%0t ns] ", $time);
    endtask

endmodule