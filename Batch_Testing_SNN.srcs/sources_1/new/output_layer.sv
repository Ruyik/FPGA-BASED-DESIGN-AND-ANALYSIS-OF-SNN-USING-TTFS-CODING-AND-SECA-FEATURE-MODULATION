`timescale 1ns / 1ps

module output_layer # (
    parameter H = 12,
    parameter OUT = 3,
    parameter TMAX = 31
)(
    input  logic          clk,
    input  logic          rst_n,
    input  logic [H-1:0]  hidden_spikes,    
    input  logic [8:0]    scale_factors [H], 
    input  logic [4:0]    time_step,        
    input  logic          clear_v,            
    output logic signed [31:0] v_out [OUT], 
    output logic [OUT-1:0]     prediction,      
    output logic               done_prediction   
);

    // 1. Weights ROM
    logic signed [15:0] out_weights_rom [72];
    initial $readmemh("output_w.mem", out_weights_rom);

    // Benchmarked rate coding ceiling matches your 25,000 threshold choice
    localparam signed [31:0] THRESHOLD = 32'd31000; 

    // --- RATE CODING MULTI-SPIKE ACCUMULATORS ---
    // Replaced single-bit 'already_spiked' flag with 4-bit saturation counters
    logic [3:0] spike_counters [H]; 
    
    // --- PIPELINE REGISTERS ---
    logic [H-1:0] s1_spikes;
    logic [8:0]   s1_scales [H];
    logic [4:0]   s1_time;

    logic signed [31:0] s2_freq_inc [H][OUT];
    logic signed [31:0] s2_lat_inc  [H][OUT];
    logic [H-1:0]       s2_valid;

    logic signed [31:0] s3_sub_sum [OUT][3];

    // ---------------------------------------------------------
    // STAGE 1: Latch Inputs (1 Cycle)
    // ---------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n || clear_v) begin
            s1_spikes <= '0;
            s1_time   <= '0;
        end else begin
            s1_spikes <= hidden_spikes;
            s1_time   <= time_step;
            s1_scales <= scale_factors;
        end
    end

    // =========================================================================
    // STAGE 2: Explicitly Signed Multi-Spike Arithmetic Pipeline
    // =========================================================================
    // Standalone signed reference array to force proper hardware sign-extensions
    logic signed [5:0] signed_counters [H];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n || clear_v) begin
            s2_valid <= '0;
            for (int i=0; i<H; i++) begin
                spike_counters[i] <= 4'd0;
                signed_counters[i] <= 6'sd0; // Clear signed registers
                for (int j=0; j<OUT; j++) begin
                    s2_freq_inc[i][j] <= 32'sd0;
                    s2_lat_inc[i][j]  <= 32'sd0;
                end
            end
        end else begin
            s2_valid <= s1_spikes;
            
            for (int i=0; i<H; i++) begin
                // Track spike frequency history
                if (s1_spikes[i] && (spike_counters[i] != 4'hF)) begin
                    spike_counters[i] <= spike_counters[i] + 4'd1;
                end

                // Secure sign-extension by using a clean, standalone signed assignment
                signed_counters[i] <= $signed({2'b00, spike_counters[i]});

                for (int j=0; j<OUT; j++) begin
                    if (s1_spikes[i]) begin
                        // Frequency Component: Stays signed automatically
                        s2_freq_inc[i][j] <= out_weights_rom[(j*12)+i] * $signed({1'b0, s1_scales[i]});
                        
                        // Latency Component: Multiplied by the clean signed register block
                        // Both operands are explicitly signed, ensuring a proper signed product
                        s2_lat_inc[i][j]  <= (out_weights_rom[36+(j*12)+i] * signed_counters[i]) >>> 1;
                    end else begin
                        s2_freq_inc[i][j] <= 32'sd0;
                        s2_lat_inc[i][j]  <= 32'sd0;
                    end
                end
            end
        end
    end

    // ---------------------------------------------------------
    // STAGE 3: Full-Precision Sub-Summation (Timing Break)
    // ---------------------------------------------------------
    always_ff @(posedge clk) begin
        for (int j=0; j<OUT; j++) begin
            s3_sub_sum[j][0] <= (s2_valid[0] ? (s2_freq_inc[0][j] + s2_lat_inc[0][j]) : 32'sd0) +
                                (s2_valid[1] ? (s2_freq_inc[1][j] + s2_lat_inc[1][j]) : 32'sd0) +
                                (s2_valid[2] ? (s2_freq_inc[2][j] + s2_lat_inc[2][j]) : 32'sd0) +
                                (s2_valid[3] ? (s2_freq_inc[3][j] + s2_lat_inc[3][j]) : 32'sd0);

            s3_sub_sum[j][1] <= (s2_valid[4] ? (s2_freq_inc[4][j] + s2_lat_inc[4][j]) : 32'sd0) +
                                (s2_valid[5] ? (s2_freq_inc[5][j] + s2_lat_inc[5][j]) : 32'sd0) +
                                (s2_valid[6] ? (s2_freq_inc[6][j] + s2_lat_inc[6][j]) : 32'sd0) +
                                (s2_valid[7] ? (s2_freq_inc[7][j] + s2_lat_inc[7][j]) : 32'sd0);

            s3_sub_sum[j][2] <= (s2_valid[8] ? (s2_freq_inc[8][j] + s2_lat_inc[8][j]) : 32'sd0) +
                                (s2_valid[9] ? (s2_freq_inc[9][j] + s2_lat_inc[9][j]) : 32'sd0) +
                                (s2_valid[10]? (s2_freq_inc[10][j] + s2_lat_inc[10][j]): 32'sd0) +
                                (s2_valid[11]? (s2_freq_inc[11][j] + s2_lat_inc[11][j]): 32'sd0);
        end
    end

    // ---------------------------------------------------------
    // STAGE 4: Membrane Potentials Accumulation & Early Stop Check
    // ---------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n || clear_v) begin
            for (int j=0; j<OUT; j++) v_out[j] <= 32'sd0;
            prediction <= '0;
            done_prediction <= (clear_v) ? 1'b0 : 1'b1;
        end else if (!done_prediction) begin
            
            for (int j=0; j<OUT; j++) begin
                logic signed [31:0] total_cycle_inc;
                total_cycle_inc = (s3_sub_sum[j][0] + s3_sub_sum[j][1] + s3_sub_sum[j][2]) >>> 7;

                if (total_cycle_inc < 0 && (v_out[j] < -total_cycle_inc))
                    v_out[j] <= 32'sd0;
                else
                    v_out[j] <= v_out[j] + total_cycle_inc;
            end

            // Early termination tracking boundary check
            if (time_step >= 3 && s1_time >= 12 && (v_out[0] >= THRESHOLD || v_out[1] >= THRESHOLD || v_out[2] >= THRESHOLD)) begin
                done_prediction <= 1'b1;
                if (v_out[0] >= v_out[1] && v_out[0] >= v_out[2])      prediction <= 3'd0;
                else if (v_out[1] >= v_out[0] && v_out[1] >= v_out[2]) prediction <= 3'd1;
                else                                                   prediction <= 3'd2;
                
                $display("\n>>> POISSON ACCELERATOR EARLY STOP TRIGGERED!");
                $display(">>> THRESHOLD CROSSED at Clock:%0d (Algorithmic T:%0d)! V0:%0d, V1:%0d, V2:%0d\n", 
                         time_step, s1_time, v_out[0], v_out[1], v_out[2]);
            
            end else if (time_step == TMAX) begin
                done_prediction <= 1'b1;
                if (v_out[0] >= v_out[1] && v_out[0] >= v_out[2])      prediction <= 3'd0;
                else if (v_out[1] >= v_out[0] && v_out[1] >= v_out[2]) prediction <= 3'd1;
                else                                                   prediction <= 3'd2;

                $display(">>> TMAX REACHED. Prediction made by highest voltage.");
                $display(">>> FINAL POTENTIALS: V0:%0d, V1:%0d, V2:%0d", v_out[0], v_out[1], v_out[2]);
            end
        end
    end
endmodule