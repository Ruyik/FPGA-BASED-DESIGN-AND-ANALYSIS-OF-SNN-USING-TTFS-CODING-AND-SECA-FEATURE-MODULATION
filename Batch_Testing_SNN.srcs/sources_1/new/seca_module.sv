module seca_module #(

    parameter H = 12

)(

    input  logic         clk,
    input  logic         rst_n,
    input  logic         enable_count,  // High during runtime active processing steps
    input  logic [H-1:0] hidden_spikes,
    input  logic         clear_counts,  // Tied to start to flush memory fields
    input  logic [4:0]   time_step,        
    output logic [8:0]   scale_factors [H] 

);

    logic [11:0] evidence_d1;
    logic [11:0] evidence_d0;
    logic [11:0] evidence_d2;

    // Direct combinational count of incoming spikes per group on this step
    logic [2:0] current_spikes_d1;
    logic [2:0] current_spikes_d0;
    logic [2:0] current_spikes_d2;

    always_comb begin
        current_spikes_d1 = hidden_spikes[0] + hidden_spikes[1] + hidden_spikes[2];
        current_spikes_d0 = hidden_spikes[3] + hidden_spikes[4] + hidden_spikes[5] + hidden_spikes[6];
        current_spikes_d2 = hidden_spikes[7] + hidden_spikes[8] + hidden_spikes[9] + hidden_spikes[10] + hidden_spikes[11];

    end

    // 2. Hardware Leak Memory Update (Simulates (Ev * 12) >>> 4)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            evidence_d1 <= 12'd0;
            evidence_d0 <= 12'd0;
            evidence_d2 <= 12'd0;
        end else if (clear_counts) begin
            evidence_d1 <= 12'd0;
            evidence_d0 <= 12'd0;
            evidence_d2 <= 12'd0;
        end else if (enable_count) begin
            // Apply leak multiplication factor (Ev * 12) / 16 + (Spike * 256)
            evidence_d1 <= (((evidence_d1 * 12) >> 4) + (current_spikes_d1 * 256));
            evidence_d0 <= (((evidence_d0 * 12) >> 4) + (current_spikes_d0 * 256));
            evidence_d2 <= (((evidence_d2 * 12) >> 4) + (current_spikes_d2 * 256));
        end
    end

    // 3. Normalized Score Computations (Score = Sum / Group Size)
    // Avoids floating point division using precise fractional multipliers
    // Score_D1 = Ev_D1 / 3  ==> (Ev_D1 * 85)  >> 8
    // Score_D0 = Ev_D0 / 4  ==> (Ev_D0 * 64)  >> 8
    // Score_D2 = Ev_D2 / 5  ==> (Ev_D2 * 51)  >> 8
    logic [11:0] score_d1;
    logic [11:0] score_d0;
    logic [11:0] score_d2;

    always_comb begin
        score_d1 = (evidence_d1 * 85)  >> 8;
        score_d0 = (evidence_d0 * 64)  >> 8;
        score_d2 = (evidence_d2 * 51)  >> 8;
    end

    // 4. Competitive Gate: Winner-Take-All Selector Node
    logic [1:0] group_winner;
    always_comb begin
        if ((score_d0 >= score_d1) && (score_d0 >= score_d2))      group_winner = 2'd0; 
        else if ((score_d1 >= score_d0) && (score_d1 >= score_d2)) group_winner = 2'd1; 
        else                                                       group_winner = 2'd2; 

    end

    // 5. Total System Evidence Gate (Matches Python SECA_EVIDENCE_THRESHOLD = 512)
    logic [13:0] total_system_evidence;
    assign total_system_evidence = evidence_d1 + evidence_d0 + evidence_d2;
    // ---------------------------------------------------------
    // 6. Excitation Stage: Calibrated Scale Drive Assignment
    // ---------------------------------------------------------
    // FIX: Add a 1-cycle pipeline register to step outputs down
    // to match the Stage 1 latch latency inside your output_layer.sv module.
    logic [8:0] next_scale_factors [H];

    always_comb begin
        if (time_step == 5'd0 || (total_system_evidence < 14'd512)) begin
            for (int i=0; i<H; i++) next_scale_factors[i] = 9'd128;
        end else begin
            for (int i=0; i<H; i++) next_scale_factors[i] = 9'd128;
            case (group_winner)
                2'd0: begin 
                    for (int i=3; i<7; i++)  next_scale_factors[i] = 9'd224; 
                    for (int i=0; i<3; i++)  next_scale_factors[i] = 9'd80;  
                    for (int i=7; i<12; i++) next_scale_factors[i] = 9'd127; 
                end

                2'd1: begin 
                    for (int i=0; i<3; i++)  next_scale_factors[i] = 9'd224; 
                    for (int i=3; i<7; i++)  next_scale_factors[i] = 9'd48;  
                    for (int i=7; i<12; i++) next_scale_factors[i] = 9'd40; 
                end

                2'd2: begin 
                    for (int i=7; i<12; i++) next_scale_factors[i] = 9'd256; 
                    for (int i=0; i<3; i++)  next_scale_factors[i] = 9'd42;  
                    for (int i=3; i<7; i++)  next_scale_factors[i] = 9'd110; 
                end

                default: begin
                    for (int i=0; i<H; i++) next_scale_factors[i] = 9'd128;
                end
            endcase
        end
    end

    // Clocked output buffer to stabilize routing paths
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i=0; i<H; i++) scale_factors[i] <= 9'd128;
        end else begin
            scale_factors <= next_scale_factors;
        end
    end
endmodule