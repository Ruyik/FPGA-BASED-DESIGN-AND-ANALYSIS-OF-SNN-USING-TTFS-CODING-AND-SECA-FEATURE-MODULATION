# =========================================================================
# CLOCK DEFINITION
# =========================================================================
# Adjusted period to 286.5ns to resolve the negative slack in the hidden layer
create_clock -period 286.500 -name clk [get_ports clk]

# =========================================================================
# INPUT DELAY CONSTRAINTS
# =========================================================================
# Constraints for external asynchronous push buttons and switches
set_input_delay -clock clk -max 10.000 [get_ports rst_n]
set_input_delay -clock clk -max 10.000 [get_ports start]
set_input_delay -clock clk -max 10.000 [get_ports view_sel[*]]

# =========================================================================
# OUTPUT DELAY CONSTRAINTS
# =========================================================================
# Constraints for external physical classification LEDs and status pins
set_output_delay -clock clk -max 10.000 [get_ports led_pred[*]]
set_output_delay -clock clk -max 10.000 [get_ports busy_led]
set_output_delay -clock clk -max 10.000 [get_ports done_led]
set_output_delay -clock clk -max 10.000 [get_ports debug_v_out[*]]