`timescale 1ns / 1ps

module lif_neuron #(
    parameter WEIGHT_WIDTH = 16,
    parameter V_WIDTH = 32,
    parameter signed [V_WIDTH-1:0] THRESHOLD = 32'sd3000 
)(
    input  logic                               clk,
    input  logic                               rst_n,
    input  logic signed [WEIGHT_WIDTH-1:0]     weight,     
    input  logic                               spike_in,   
    input  logic                               clear_v,    
    output logic                               spike_out,  
    output logic signed [V_WIDTH-1:0]          v_state     
);

    logic signed [V_WIDTH-1:0] v_reg;
    logic signed [V_WIDTH-1:0] v_decayed;
    logic signed [V_WIDTH-1:0] v_next;
    logic signed [V_WIDTH-1:0] v_next_bounded;
    logic [1:0] refractory_counter;

    assign v_state = v_reg;

    always_comb begin
        // Cast 15 to match dynamic V_WIDTH to guarantee area synthesis stability
        v_decayed = (v_reg * ( (V_WIDTH)'(15) )) >>> 4;
        
        if (spike_in) begin
            v_next = v_decayed + {{ (V_WIDTH-WEIGHT_WIDTH){weight[WEIGHT_WIDTH-1]} }, weight}; 
        end else begin
            v_next = v_decayed;
        end

        if (v_next < 32'sd0) begin
            v_next_bounded = 32'sd0;
        end else begin
            v_next_bounded = v_next;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v_reg              <= '0;
            spike_out          <= 1'b0;
            refractory_counter <= 2'd0;
        end else if (clear_v) begin
            v_reg              <= '0;
            spike_out          <= 1'b0;
            refractory_counter <= 2'd0;
        end else begin
            if (refractory_counter > 2'd0) begin
                refractory_counter <= refractory_counter - 2'd1;
                v_reg              <= '0;
                spike_out          <= 1'b0;
            end else begin
                if (v_next_bounded >= THRESHOLD) begin
                    spike_out          <= 1'b1;         
                    v_reg              <= '0;       
                    refractory_counter <= 2'd2;         
                end else begin
                    spike_out          <= 1'b0;
                    v_reg              <= v_next_bounded; 
                end
            end
        end
    end
    
endmodule