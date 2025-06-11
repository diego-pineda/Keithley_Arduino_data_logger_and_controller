
import pyvisa
import numpy as np
import time

# Connect to the keithley 2400 SourceMeter and set it to a variable named Sourcemeter
Sourcemeter = pyvisa.ResourceManager().open_resource('GPIB0::24::INSTR')
initial_voltage = 0.5  # Set your desired voltage in Volts
voltages_to_test = np.linspace(0.5, 0.4, 11)  # [1.00] #
time_interval = 60  # [s] Time interval between steps in flow rate

# Starting the pump

# Set source mode to voltage
Sourcemeter.write(":SOURCE:FUNC VOLTAGE")

# Define desired voltage and compliance current (optional)

# compliance_current = 1e-3  # Set compliance current limit in Amps (optional)
Sourcemeter.write(f":SOURCE:VOLT {initial_voltage}")  # Set source voltage
Sourcemeter.write(":OUTPUT:STATE ON")  # Enable source output
print(f'Sequence started. Initial voltage = {initial_voltage} V\n'
      f'Time interval between voltage values = {time_interval} seconds\n'
      f'Estimated experiment time = {time_interval*len(voltages_to_test)/60} minutes')
time.sleep(10)
print("Flow rate sweep will start now.\n")
for voltage in voltages_to_test:

    Sourcemeter.write(f":SOURCE:VOLT {voltage}")
    print('Current voltage is: {} V'.format(voltage))
    time.sleep(time_interval)
    # (Optional) Read and print measured voltage
    # measured_voltage = float(Sourcemeter.query(":READ?").split(",")[0])
    # print(f"Measured voltage: {measured_voltage} V")
# Set compliance current (optional, for safety)
# Sourcemeter.write(f":SOURCE:CURR:LIMIT {compliance_current}")

# Disable source output (recommended for safety)
# Sourcemeter.write(":OUTPUT:STATE OFF")

# Close connection
Sourcemeter.close()

print("Script finished.")

# Sourcemeter.write('OUTPUT ON')
# commandstring_prefix_Keithly = "SOURCE:VOLT"
# stringflowrate = str(Sourcemeterlist[0])
# combinedstring_Keithly = commandstring_prefix_Keithly + " " + stringflowrate
# Sourcemeter.write(combinedstring_Keithly)

#for v in range(len(Sourcemeterlist)):
#   if flag == 0:
#      break
# else:
#    stringflowrate = str(Sourcemeterlist[v])
#   combinedstring_Keithly = commandstring_prefix_Keithly + " " + stringflowrate
#  Sourcemeter.write(combinedstring_Keithly)  # Command to change flow rate of the pump

# time.sleep(60)


#Sourcemeter.write('OUTPUT OFF')  # Turn off the Pump