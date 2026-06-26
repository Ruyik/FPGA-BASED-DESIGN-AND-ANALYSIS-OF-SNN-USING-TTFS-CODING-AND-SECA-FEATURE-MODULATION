`timescale 1ns / 1ps

module rate_encoder #(
    parameter TMAX = 31
)(
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,
    input  logic       step_pulse,
    input  logic [7:0] pixel_val,
    output logic       spike_out
);

    logic [15:0] lfsr_reg;
    logic        running;
    logic [4:0]  counter;
    
    // NEW: Register block to hold shifted attenuated probability ceiling
    logic [7:0]  attenuated_pixel;

    // Standard 16-bit Galois LFSR with maximal feedback taps (Max info density)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr_reg <= 16'hACE1; // Seed value must be non-zero
        end else if (start) begin
            lfsr_reg <= 16'hACE1;
        end else if (running) begin
            lfsr_reg <= (lfsr_reg >> 1) ^ (lfsr_reg[0] ? 16'hB400 : 16'h0000);
        end
    end

    // Runtime Control FSM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter          <= 5'd0;
            running          <= 1'b0;
            spike_out        <= 1'b0;
            attenuated_pixel <= 8'd0;
        end else if (start) begin
            counter          <= 5'd0;
            running          <= 1'b1;
            spike_out        <= 1'b0;
            
            // ATTENUATOR INJECTION: Approximates 0.375 * pixel_val with zero hardware multipliers
            attenuated_pixel <= (pixel_val >> 1) + (pixel_val >> 2); // (0.50 + 0.125) = 0.625
        end else if (running) begin
            // Probabilistic Threshold Comparison against attenuated ceiling floor
            if (attenuated_pixel > lfsr_reg[7:0] && attenuated_pixel > 8'd0) begin
                spike_out <= 1'b1;
            end else begin
                spike_out <= 1'b0;
            end

            if (step_pulse) begin
                if (counter == TMAX) begin
                    running   <= 1'b0;
                    spike_out <= 1'b0;
                end else begin
                    counter <= counter + 5'd1;
                end
            end
        end
    end

endmodule