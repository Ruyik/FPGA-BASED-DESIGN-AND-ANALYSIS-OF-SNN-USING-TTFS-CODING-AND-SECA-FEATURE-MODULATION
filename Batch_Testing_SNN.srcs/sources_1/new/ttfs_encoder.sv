module ttfs_encoder #(
    parameter TMAX = 31
)(
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,
    input  logic       step_pulse, // Pulse from controller at the end of each scan
    input  logic [7:0] pixel_val,
    output logic       spike_out,
    output logic       busy
);

    logic [4:0] counter;
    logic [4:0] trigger_time;
    logic       running;
    logic       has_fired; // NEW: Track if this pixel already sent its pulse

    assign trigger_time = TMAX - (pixel_val >> 3);
    assign busy = running;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            counter   <= 0;
            spike_out <= 0;
            running   <= 0;
            has_fired <= 0;
        end else begin
            if (start) begin
                counter   <= 0;
                spike_out <= 0;
                running   <= 1;
                has_fired <= 0; // Reset firing history for new image
            end else if (running) begin
                
                // --- CORRECTED PULSE-TRIGGERED LOGIC ---
                // Fire a spike out ONLY on the exact matching cycle, and only ONCE
                if ((counter == trigger_time) && !has_fired) begin
                    spike_out <= 1'b1;
                    has_fired <= 1'b1; // Permanently latch that this pixel is done
                end else begin
                    spike_out <= 1'b0; // Auto-clears to 0 on the very next clock cycle!
                end
                
                // Advance time when the Controller says a full scan is done
                if (step_pulse) begin
                    if (counter == TMAX) begin
                        running   <= 0;
                        spike_out <= 0; 
                    end else begin
                        counter <= counter + 1;
                    end
                end
            end
        end
    end
endmodule