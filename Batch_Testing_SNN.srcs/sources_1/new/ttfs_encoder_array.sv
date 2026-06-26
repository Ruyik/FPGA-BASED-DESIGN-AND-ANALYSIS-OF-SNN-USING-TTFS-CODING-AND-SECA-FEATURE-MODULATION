
module ttfs_encoder_array #(
    parameter NUM_ENCODERS = 196,
    parameter TMAX = 31
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic step_pulse,    // NEW: Receives the strobe from the Controller
    input  logic [7:0] pixel_data [NUM_ENCODERS],
    output logic [NUM_ENCODERS-1:0] spike_bus_out
);

    genvar i;
    generate
        for (i = 0; i < NUM_ENCODERS; i = i + 1) begin : gen_encoders
            ttfs_encoder #(
                .TMAX(TMAX)
            ) encoder_inst (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .step_pulse(step_pulse), // Pass the pulse to each encoder
                .pixel_val(pixel_data[i]),
                .spike_out(spike_bus_out[i]),
                .busy() 
            );
        end
    endgenerate

endmodule