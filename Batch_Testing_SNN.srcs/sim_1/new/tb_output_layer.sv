`timescale 1ns / 1ps

module tb_output_layer;

    // --- Parameters ---
    localparam H = 12;
    localparam OUT = 3;
    localparam TMAX = 31;
    localparam CLK_PERIOD = 10; // 100MHz

    // --- Interface Signals ---
    logic          clk;
    logic          rst_n;
    logic [H-1:0]  hidden_spikes;
    logic [8:0]    scale_factors [H];
    logic [4:0]    time_step;
    logic          clear_v;
    logic signed [31:0] v_out [OUT];
    logic [OUT-1:0]     prediction;
    logic               done_prediction;

    // --- DUT Instantiation ---
    output_layer #(
        .H(H),
        .OUT(OUT),
        .TMAX(TMAX)
    ) uut (.*);

    // --- Clock Generator Engine ---
    always #(CLK_PERIOD/2) clk = ~clk;

    // --- Main Stimulus ---
    initial begin
        // 1. Initialize State
        clk = 0;
        rst_n = 0;
        clear_v = 0;
        time_step = 0;
        hidden_spikes = '0;
        
        // Initialize SECA scale factors with a realistic nominal base of 128
        for (int i=0; i<H; i++) scale_factors[i] = 9'd128; 

        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        $display("\n==================================================================");
        $display("[0 ns] STARTING REALISTIC HIGH-DENSITY ACCUMULATION TESTBENCH");
        $display("==================================================================");

        // Flash internal state registers to raw zero parameters
        clear_v = 1;
        hidden_spikes = '0;
        time_step = 0;
        @(posedge clk);
        #2; // Let reset values settle completely across delta-cycles
        clear_v = 0;
        @(posedge clk);
        #2;

        // -----------------------------------------------------------------
        // REALISTIC INFERENCE TEMPORAL STREAM EVENT MATRIX
        // -----------------------------------------------------------------
        for (int t = 1; t <= 25; t++) begin
            // 1. Apply driving inputs immediately on the clock edge
            time_step = t;
            
            case (t)
                1: hidden_spikes = 12'b0000_0000_1001; 
                2: hidden_spikes = 12'b0000_0100_0000; 
                3: hidden_spikes = 12'b0000_1000_0010; 
                4: hidden_spikes = 12'b0001_0000_0100; 
                5: begin 
                    hidden_spikes = 12'b0000_0000_1011; 
                    scale_factors[0] = 9'd224; scale_factors[1] = 9'd224; scale_factors[2] = 9'd224; 
                   end
                6: hidden_spikes = 12'b0000_0000_0111; 
                7: hidden_spikes = 12'b0000_0000_1111; 
                8: hidden_spikes = 12'b0000_0000_1011;
                9: begin 
                    hidden_spikes = 12'b0000_0000_1111; 
                    scale_factors[0] = 9'd256; scale_factors[1] = 9'd256; scale_factors[2] = 9'd256; 
                   end
                10: hidden_spikes = 12'b0000_0000_0111;
                11: hidden_spikes = 12'b0000_0000_1111;
                12: hidden_spikes = 12'b0000_0000_1011;
                13: hidden_spikes = 12'b0000_0100_0000;
                14: hidden_spikes = 12'b0000_0000_0001;
                default: hidden_spikes = '0; 
            endcase

            // 2. Wait for the clock edge to pass AND let combinational signals settle
            @(posedge clk);
            #2; // FIXED: Settling delay protects against premature early stop triggers
            
            $display("Clock: %2d | Active Bus: %b | Probe Counter[0]: %0d | Potentials -> V0: %7d | V1: %7d | V2: %7d", 
                     t, uut.s1_spikes, uut.spike_counters[0], v_out[0], v_out[1], v_out[2]);
                     
            // 3. Now sample the output status line safely
            if (done_prediction) begin
                $display("\n>>> POISSON ACCELERATOR EARLY TERMINATION CALLED EARLY!");
                $display(">>> Winner-Take-All Registered Pred Class: %0d at Cycle %0d", prediction, t);
                break;
            end
        end

        // -----------------------------------------------------------------
        // FINAL COMPLIANCE VERIFICATION CHECK
        // -----------------------------------------------------------------
        $display("\n==================================================================");
        $display("                FINAL BENCHMARK EVALUATION LOG");
        $display("==================================================================");
        if (done_prediction && prediction == 3'd1) begin
            $display(">>> BENCHMARK STATUS: SUCCESS! REALISTIC BURST INTEGRATION FULLY ACCURATE.");
        end else begin
            $display(">>> BENCHMARK STATUS: FAILED! ACCUMULATOR STRUCTURAL MISMATCH.");
        end
        $display("==================================================================\n");

        $finish;
    end

endmodule