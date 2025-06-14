﻿#import gpib_ctypes
#gpib_ctypes.gpib.gpib._load_lib("C:/Users/michi/Anaconda 3/Lib/site-packages/gpib_ctypes")
import pyvisa
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from scipy.interpolate import CubicSpline
from datetime import date
import threading
from serial import Serial
import numpy as np
import sys

z = 0

PT100_data_frame = pd.read_csv('./PT100_Table.txt', sep='\t', header=None)
PT100_array = pd.DataFrame(PT100_data_frame).to_numpy()
PT100 = CubicSpline(PT100_array[:, 1], PT100_array[:, 0])


def is_valid_number(value):
    """Checks if a string represents a valid float (including + or - signs)."""
    try:
        float(value)  # Attempt conversion to float
        return True
    except ValueError:
        return False


def get_user_input(prompt):
    """Prompts the user for input and ensures it follows the underscore format."""
    while True:
        user_input = input(prompt)
        if " " not in user_input:
            return user_input
        else:
            print("Invalid input. Please use underscores between words and no spaces.")


def get_channel_input(prompt):
    """Prompts the user for input and ensures it follows the required format."""
    while True:
        user_input = input(prompt)
        if user_input.isdigit() and 0 < int(user_input) < 6:
            return user_input
        else:
            print("Invalid input. Please type only integer numbers between 1 and 5")

def get_y_or_n_input(prompt):
    while True:
        user_input = input(prompt).lower()
        if user_input == "y" or user_input == "n":
            return user_input
        else:
            print("Invalid input. Please type only y or n. It can be upper or lower case.")

block_ID = get_user_input("Enter block ID (use underscore between words, no spaces): ")
experiment_ID = get_user_input("\nEnter experiment name (use underscore between words, no spaces): ")
Julabo_channel = get_user_input('\nTurn ON the Julabo thermostatic bath. Connect the USB cable to your PC. Read the serial port from the device manager of your PC.'
                                ' \nEnter the serial port of the Julabo TB. The entered value should be something '
                                'like COM6. Julabo serial port: ')

# Asking the user to assign a channel to every RTD connected to multimeter 2

T_block_top_channel_index = int(get_channel_input('\nRTD sensing the temperature of top of the block is connected to channel: '))-1
T_block_bottom_channel_index = int(get_channel_input('\nRTD sensing the temperature of bottom of the block is connected to channel: '))-1
T_amb_channel_index = int(get_channel_input('\nRTD sensing the ambient temperature is connected to channel: '))-1
# T_block_left_channel_index = int(get_channel_input('\nRTD sensing the temperature of left part of the block is connected to channel: '))-1
# T_block_right_channel_index = int(get_channel_input('\nRTD sensing the temperature of right part of the block is connected to channel: '))-1
T_water_in_channel_index = int(get_channel_input('\nRTD sensing the temperature of water entering the block is connected to channel: '))-1
T_water_out_channel_index = int(get_channel_input('\nRTD sensing the temperature of water leaving the block is connected to channel: '))-1

# Verification of channels on multimeter 1

verification_prompt = "\nCheck the connection of the following signals on multimeter 1 (Older Keithley 2000):\n" \
                      "\nChannel 1: Voltage drop across the plates\n" \
                      "Channel 2: Voltage drop across the block\n" \
                      "Channel 3: Differential pressure sensor\n" \
                      "Channel 4: Current shunt\n" \
                      "\nAre these signals properly connected? If not the program will stop. Make the necessary adjustments please! (y/n): "

verification_channels_multimeter_1 = get_y_or_n_input(verification_prompt)

if verification_channels_multimeter_1 == 'n':
    print("\nProgram will stop now. Connect signals properly in Multimeter 1 (older Keithley 2000).")
    sys.exit()

output_file_name = './Sensor_data/' + str(date.today().strftime("%d%b%Y")) + '_' + block_ID + '_' + experiment_ID + '.csv'
ser = Serial(Julabo_channel)
print('Output file name: ' + output_file_name)



# **********************************    Initialize Multimeters    *******************************************

multimeter1 = pyvisa.ResourceManager().open_resource('GPIB0::16::INSTR')  # Connect to a keithley 2000 and set it to a variable named multimeter1.
multimeter2 = pyvisa.ResourceManager().open_resource('GPIB0::20::INSTR')  # Connect to the keithley 2000 and set it to a variable named multimeter2.
multimeter3 = pyvisa.ResourceManager().open_resource('GPIB0::22::INSTR')  # Connect to the Agilent and set it to a variable named multimeter.
multimeter4 = pyvisa.ResourceManager().open_resource('GPIB0::12::INSTR')  # Connect to the LakeShore 331 temperature controller and set it to a variable named multimeter4.
nanovoltmeter = pyvisa.ResourceManager().open_resource('GPIB0::07::INSTR')
#multimeter4 = pyvisa.ResourceManager().open_resource('GPIB0::18::INSTR')# Connect to the Keithly and set it to a variable named multimeter.
#Sourcemeter = pyvisa.ResourceManager().open_resource('GPIB0::24::INSTR')# Connect to the keithly and set it to a variable named sourcemeter
ser.write(b"OUT_mode_05 1\r\n")  # Starts remote control of the JULABO refrigerated circulator

# ******************************* Setting up Multimeter 1 *****************************************

# Define some parameters
dataElements = "READ"
#number_data_elements_per_chan = len(dataElements)  # saving only reading, time stamp apparently not possible
chanList = "(@1:4)"  # "(@1:5)"
number_channels_in_scan = 4  # 5
number_of_scans = 1
bufferSize = number_of_scans * number_channels_in_scan
# debug = 1

DMM1_cmd_list = ["*RST",
                 "*CLS",
                 "SYSTEM:PRESET",
                 "TRAC:CLE",
                 "INIT:CONT OFF",
                 "TRIG:COUN " + str(number_of_scans),
                 "SAMP:COUN " + str(number_channels_in_scan),
                 "TRIG:SOUR IMM",
                 "TRACE:POINTS " + str(bufferSize),
                 "TRACE:FEED SENS",
                 "TRACE:FEED:CONT NEXT",
                 "FORM:DATA ASCII",
                 "FORM:ELEM " + dataElements,
                 "FUNC 'VOLTage:DC'",
                 "VOLTage:DC:NPLC 7",
                 "VOLTage:DC:RANG 10",
                 "ROUT:SCAN " + chanList,
                 "ROUT:SCAN:LSEL INT",
                 "STAT:MEAS:ENAB 512",
                 "*SRE 1",
                 "*OPC?"]

for cmd in DMM1_cmd_list:
    multimeter1.write(cmd)

print(multimeter1.read())   #read the *OPC? response
print("*OPC received; finished setting up Keithley 2000 Multimeter 1")

# ************************************ Setting up Multimeter 2 **********************************

# Define some parameters
dataElements = "READ"
#number_data_elements_per_chan = len(dataElements)  # reading
chanList = "(@1:5)"
number_channels_in_scan = 5
number_of_scans = 1
bufferSize = number_of_scans * number_channels_in_scan
# debug = 1

# "ROUT:SCAN:TSO IMM", is not a valid statement for Keithley 2000
DMM2_cmd_list = ["*RST",
                 "*CLS",
                 "SYSTEM:PRESET",
                 "TRAC:CLE",
                 "INIT:CONT OFF",  # INITiate:CONTinuous OFF has to do with the start of the trigger model
                 "TRIG:COUN " + str(number_of_scans),
                 "SAMP:COUN " + str(number_channels_in_scan),
                 "TRIG:SOUR IMM",
                 "TRACE:POINTS " + str(bufferSize),
                 "TRACE:FEED SENS",
                 "TRACE:FEED:CONT NEXT",
                 "FORM:DATA ASCII",
                 "FORM:ELEM " + dataElements,
                 "FUNC 'FRESistance'",
                 "FRES:NPLC 7",  # Integration rate. From 0.1 to 10. 0.1 fast rate, 1 medium, 10 slow. Influences scan rate.
                 "FRES:RANG 1000",  # Sets the range to kOhm
                 "ROUT:SCAN " + chanList,
                 "ROUT:SCAN:LSEL INT",
                 "STAT:MEAS:ENAB 512",
                 "*SRE 1",
                 "*OPC?"]

for cmd2 in DMM2_cmd_list:
    multimeter2.write(cmd2)

print(multimeter2.read())   #read the *OPC? response
print("*OPC received; finished setting up Keithley 2000 Multimeter 2")

# *************************************************************

# multimeter1.write(":SENSe:FUNCtion 'VOLTage:DC'") # Set the keithley to measure Voltage DC
# multimeter2.write(":SENSe:FUNCtion 'FRESistance'") # Set the keithley  to measure 4-wire resistance
multimeter3.write(":SENSe:FUNCtion 'VOLTage:DC'")  # Set the keithley to measure Voltage DC
multimeter3.write("VOLTage:DC:NPLC 7")  # Set the keithley to measure Voltage DC
multimeter3.write("TRIG:SOUR IMM")

#********************************************************************

#multimeter4.write(":SENSe:FUNCtion 'FRESistance'") # Set the keithley to measure Voltage DC
#multimeter4.write("FRESistance:NPLC 7") # Set the keithley to measure Voltage DC
#multimeter4.write("TRIG:SOUR IMM")

# Variables multimeter 1

time_stamp_DMM1 = []
# time_volt_drop_block = []
# time_current_shunt = []
# time_diff_pressure = []
# time_diff_pressure_filter = []
# time_voltage_drop_plates = []

voltage_drop_block = []
current_shunt = []
diff_pressure = []
diff_pressure_filter = []
voltage_drop_plates = []

# Variables multimeter 3

time_flow_rate = []  # Create an empty list to store time values in.
flow_rate = []  # Create an empty list to store flow rate values in.

# Variables Julabo

time_water_in = []
T_water_in_julabo = []  # [°C]

# Variables multimeter 2

time_stamp_DMM2 = []
# time_water_out = []
# time_block_top = []
# time_block_bottom = []
# time_block_left = []
# time_block_right = []

T_water_in = []  # [°C]
T_water_out = []  # [°C]
T_block_top = []  # [°C]
T_block_bottom = []  # [°C]
T_ambient = []  # [°C]

# Variables multimeter 4

time_lakeshore = []

T_block_left = []  # [°C]
T_block_right = []  # [°C]

startTime = time.time()  # Create a variable that holds the starting timestamp.
flag = 1  # When flag is set to 0 by the user, the infinite while loop stops


def normal():
    global flag

# Create a while loop that continuously measures and plots data from the multimeters until user stops it
    while flag == 1:

        # -------------------- Getting voltages of multimeter 1 --------------------

        multimeter1.write("INIT")
        DMM1_data = multimeter1.query("TRAC:DATA?")
        DMM1_time_stamp = float(time.time() - startTime)
        # multimeter1.write("ABORt")
        multimeter1.write("TRAC:CLE")
        DMM1_data_array = [float(i) for i in DMM1_data.split(',')]

        # -------------------- Getting voltage of multimeter 3 --------------------

        voltageReading1 = float(multimeter3.query('READ?').split(' ')[0]) # [:-2]Read and process data from the keithley. # :SENSe:DATA:FRESh?  .....  DATA:LAST?
        flow_rate.append(voltageReading1)
        time_flow_rate.append(float(time.time() - startTime))
        # time.sleep(0.5)

        # ----------- Performing calculations and assigning values to variables with voltages of multimeter 1 ----------

        voltage_drop_plates.append(DMM1_data_array[0])  # CH1.Multimeter_1
        voltage_drop_block.append(DMM1_data_array[1])  # CH2.Multimeter_1
        diff_pressure.append((DMM1_data_array[2] / 5.07 - 0.04) / 0.009)  #  CH3.Multimeter_1 pressure drop across block
        current_shunt.append(DMM1_data_array[3] / (50e-3 / 200))  # CH4.Multimeter_1 divided by the resist of the shunt
        # diff_pressure_filter.append((DMM1_data_array[4] / 5.07 - 0.04) / 0.009)  # CH5.Multimeter_1

        time_stamp_DMM1.append(DMM1_time_stamp)
        # time_voltage_drop_plates.append(DMM1_time_stamp)
        # time_volt_drop_block.append(DMM1_time_stamp)
        # time_diff_pressure.append(DMM1_time_stamp)
        # time_current_shunt.append(DMM1_time_stamp)
        # time_diff_pressure_filter.append(DMM1_time_stamp)

        # -------------------- Getting 4W resistances of multimeter 2 ------------------------

        multimeter2.write("INIT")
        DMM2_data = multimeter2.query("TRAC:DATA?")
        DMM2_time_stamp = float(time.time() - startTime)
        # multimeter2.write("ABORt")
        multimeter2.write("TRAC:CLE")
        DMM2_data_array = [float(i) for i in DMM2_data.split(',')]

        # ----------------------- Getting temperature of inlet water from Julabo --------------------------

        ser.write(b"IN_pv_02\n")  # Read temperature from Julabo
        # data_T_water_in = ser.readline()  # Read the temperature of the heat bath
        # decoded_T_water_in = data_T_water_in.decode("utf-8")  # Decode the byte
        # T_water_in_julabo.append(float(decoded_T_water_in))

        try:
            data_T_water_in = ser.readline().decode("utf-8").strip()
            if not data_T_water_in:
                raise ValueError("Empty response from Julabo")
            T_water_in_julabo.append(float(data_T_water_in))
        except Exception as e:
            print(f"Error reading from Julabo: {e}")
            T_water_in_julabo.append(np.nan)


        water_in_time_stamp = float(time.time() - startTime)
        time_water_in.append(water_in_time_stamp)
        # print(decoded_T_water_in)

        # ---------------------- Calculating temperatures in Celsius with resistances from multimeter 2 ----------------

        T_water_in.append(float(PT100(DMM2_data_array[T_water_in_channel_index])))
        T_water_out.append(float(PT100(DMM2_data_array[T_water_out_channel_index])))
        T_block_top.append(float(PT100(DMM2_data_array[T_block_top_channel_index])))
        T_block_bottom.append(float(PT100(DMM2_data_array[T_block_bottom_channel_index])))
        T_ambient.append(float(PT100(DMM2_data_array[T_amb_channel_index])))

        time_stamp_DMM2.append(DMM2_time_stamp)

        # ------------------------------ Getting temperatures from multimeter 4 --------------------------------
        # temp_A = multimeter4.query("KRDG? A")
        # T_block_right.append(float(temp_A) - 273.15)
        # time.sleep(0.25)
        # temp_B = multimeter4.query("KRDG? B")
        # T_block_left.append(float(temp_B) - 273.15)

        try:
            temp_A = multimeter4.query("KRDG? A").strip()
            temp_B = multimeter4.query("KRDG? B").strip()

            # print(f"Raw LakeShore responses -> A: '{temp_A}', B: '{temp_B}'")  # Debugging

            # Validate response format before converting to float
            if not is_valid_number(temp_A) or not is_valid_number(temp_B):
                raise ValueError("LakeShore returned non-numeric data")

            T_block_right.append(float(temp_A) - 273.15)
            T_block_left.append(float(temp_B) - 273.15)

        except Exception as e:
            print(f"Error reading from LakeShore 331: {e}")
            T_block_right.append(np.nan)
            T_block_left.append(np.nan)


        DMM4_time_stamp = float(time.time() - startTime)
        time_lakeshore.append(DMM4_time_stamp)
        # Note: both temperature readings differ slightly in time but assumed collected simultaneosuly
        # time.sleep(0.4)
        # ***************************************************************************

        if flag == False:
            print('Data will be saved in the file: {}'.format(output_file_name))

        # time.sleep(1)
    # ********************************* Returning multimeters to idle state ****************************************

    multimeter1.write(":ROUTe:SCAN:LSEL NONE")
    multimeter2.write(":ROUTe:SCAN:LSEL NONE")
    multimeter4.close()
    multimeter3.close()

    # ****************** Writing data to file *****************************

    output_dataframe = pd.DataFrame({'t_temperatures': time_stamp_DMM2,
                                     'T_block_top': T_block_top,
                                     'T_block_bottom': T_block_bottom,
                                     'T_ambient': T_ambient,
                                     'T_water_out': T_water_out,
                                     'T_water_in': T_water_in,
                                     't_water_in': time_water_in,
                                     'T_julabo': T_water_in_julabo,
                                     't_lakeshore': time_lakeshore,
                                     'T_block_left': T_block_left,
                                     'T_block_right': T_block_right,
                                     't_flow': time_flow_rate,
                                     'V_flow': flow_rate,
                                     't_DMM1': time_stamp_DMM1,
                                     'Diff_press': diff_pressure,
                                     'V_block_prob': voltage_drop_block,
                                     'Current_shunt': current_shunt,
                                     'V_plates': voltage_drop_plates
                                     #'Diff_press_filter': diff_pressure_filter
                                     })

    save_data = get_y_or_n_input('Do you want to save the data (y/n)?: ')
    if save_data == 'y':
        output_dataframe.to_csv(output_file_name, sep="\t", index=False)
        print('\nData file has been saved.\nClose the figures to start saving them and exit the program.')
    else:
        print('\nData will not be saved!\nClose the figures to exit the program.')


def get_input():
    global flag
    keystrk=input('Press the enter key to stop recording data. \n')  # \n
    # thread doesn't continue until key is pressed
    print('The recording of data will stop now.', keystrk)
    flag = 0


n = threading.Thread(target=normal)
i = threading.Thread(target=get_input)
n.start()
i.start()

# **************************************** Plotting data in (almost) real time *****************************************

x_data, y_data = [], []

temperatures_figure = plt.figure(figsize=(6, 4))
plt.xlabel('Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Temperature (\u00b0C)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# plt.legend(loc='upper left', prop={'size': 6})# plt.legend(['T_water_in', 'T_water_out', 'T_block_top', 'T_block_bottom', 'T_block_left', 'T_block_right', 'T_amb'])
julabo_temp_line, = plt.plot(x_data, y_data, color='orange', linestyle='-')  # T water in from Julabo
water_in_temp_line, = plt.plot(x_data, y_data, 'b-')  # T water in
water_out_temp_line, = plt.plot(x_data, y_data, 'g-')  # T water out
block_top_temp_line, = plt.plot(x_data, y_data, 'k-')  # T block top
block_bottom_temp_line, = plt.plot(x_data, y_data, 'r-')  # T block bottom
block_left_temp_line, = plt.plot(x_data, y_data, 'm-')  # T block left
block_right_temp_line, = plt.plot(x_data, y_data, 'y-')  # T block right
amb_temp_line, = plt.plot(x_data, y_data, 'c-')  # T ambient

water_temperatures_figure = plt.figure(figsize=(6, 4))
plt.xlabel('Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Temperature (\u00b0C)')
julabo_temp_line2, = plt.plot(x_data, y_data, color='orange', linestyle='-')  # T water in from Julabo
water_in_temp_line2, = plt.plot(x_data, y_data, 'b-')  # T water in
water_out_temp_line2, = plt.plot(x_data, y_data, 'g-')  # T water out

flow_figure = plt.figure(figsize=(6, 4))
plt.xlabel('Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Flow_rate (Lpm)')  # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
flow_rate_line, = plt.plot(x_data, y_data, 'b-')

volt_drop_block_figure = plt.figure(figsize=(8, 4))
plt.xlabel('Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Voltage drop MCM block (V)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
block_volt_drop_line, = plt.plot(x_data, y_data, 'r-')

diff_press_figure = plt.figure(figsize=(8, 4))
plt.xlabel('Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Differential_pressure (kPa)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
differential_pressure_line, = plt.plot(x_data, y_data, 'r-')

heating_current_figure = plt.figure(figsize=(8, 4))
plt.xlabel('Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Heating_Current (A)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
heating_current_line, = plt.plot(x_data, y_data, 'k-')


# power_figure = plt.figure(figsize=(8, 4))
# plt.xlabel('Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# plt.ylabel('Power (W)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# electrical_power_line, = plt.plot(x_data, y_data, 'r-')
# water_absorbed_heat_line, = plt.plot(x_data, y_data, 'b-')
# figure6 = plt.figure(figsize=(8, 4))
# plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# plt.ylabel('Differential_pressure_filter (kPa)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# line10, = plt.plot(x_data, y_data, 'r-')

volt_plates_figure = plt.figure(figsize=(8, 4))
plt.xlabel('Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Voltage between plates (V)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
volt_plates_line, = plt.plot(x_data, y_data, 'r-')


def temperatures(frame):
    julabo_temp_line.set_data(time_water_in, T_water_in_julabo)
    water_in_temp_line.set_data(time_stamp_DMM2, T_water_in)
    water_out_temp_line.set_data(time_stamp_DMM2, T_water_out)
    block_top_temp_line.set_data(time_stamp_DMM2, T_block_top)
    block_bottom_temp_line.set_data(time_stamp_DMM2, T_block_bottom)
    amb_temp_line.set_data(time_stamp_DMM2, T_ambient)
    block_left_temp_line.set_data(time_lakeshore, T_block_left)
    block_right_temp_line.set_data(time_lakeshore, T_block_right)

    temperatures_figure.gca().relim()
    temperatures_figure.gca().autoscale_view()
    # temperatures_figure.legend(['T_water_in', 'T_water_out', 'T_block_top', 'T_block_bottom', 'T_block_left', 'T_block_right', 'T_amb'])
    return julabo_temp_line, water_in_temp_line, water_out_temp_line, block_top_temp_line, block_bottom_temp_line, amb_temp_line, block_left_temp_line, block_right_temp_line


def water_temperatures(frame):
    julabo_temp_line2.set_data(time_water_in, T_water_in_julabo)
    water_in_temp_line2.set_data(time_stamp_DMM2, T_water_in)
    water_out_temp_line2.set_data(time_stamp_DMM2, T_water_out)
    water_temperatures_figure.gca().relim()
    water_temperatures_figure.gca().autoscale_view()
    return julabo_temp_line2, water_in_temp_line2, water_out_temp_line2


def flow(frame):
    flow_rate_line.set_data(time_flow_rate, flow_rate)
    flow_figure.gca().relim()
    flow_figure.gca().autoscale_view()
    return flow_rate_line


def block_volt_drop(frame):
    block_volt_drop_line.set_data(time_stamp_DMM1, voltage_drop_block)
    volt_drop_block_figure.gca().relim()
    volt_drop_block_figure.gca().autoscale_view()
    return block_volt_drop_line


def differential_pressure(frame):
    differential_pressure_line.set_data(time_stamp_DMM1, diff_pressure)
    diff_press_figure.gca().relim()
    diff_press_figure.gca().autoscale_view()
    return differential_pressure_line


def heating_current(frame):
    heating_current_line.set_data(time_stamp_DMM1, current_shunt)
    heating_current_figure.gca().relim()
    heating_current_figure.gca().autoscale_view()
    return heating_current_line


# def update6(frame):
#     line10.set_data(time_diff_pressure_filter, diff_pressure_filter)
#     figure6.gca().relim()
#     figure6.gca().autoscale_view()
#     return line10


def volt_plates(frame):
    volt_plates_line.set_data(time_stamp_DMM1, voltage_drop_plates)
    volt_plates_figure.gca().relim()
    volt_plates_figure.gca().autoscale_view()
    return volt_plates_line


# def power(frame):
#     electrical_power_line.set_data(time_stamp_DMM1, np.array(voltage_drop_plates) * np.array(current_shunt))
#     water_absorbed_heat_line.set_data(time_stamp_DMM2, 4200 * (np.array(flow_rate)/60) * (np.array(T_water_out) - np.array(T_water_in)))
#     power_figure.gca().relim()
#     power_figure.gca().autoscale_view()
#     return electrical_power_line, water_absorbed_heat_line



animation1 = FuncAnimation(temperatures_figure, temperatures, interval=200)
animation2 = FuncAnimation(flow_figure, flow, interval=200)
animation3 = FuncAnimation(volt_drop_block_figure, block_volt_drop, interval=200)
animation4 = FuncAnimation(diff_press_figure, differential_pressure, interval=200)
animation5 = FuncAnimation(heating_current_figure, heating_current, interval=200)
# animation6 = FuncAnimation(figure6, update6, interval=200)
animation7 = FuncAnimation(volt_plates_figure, volt_plates, interval=200)
animation8 = FuncAnimation(water_temperatures_figure, water_temperatures, interval=200)
# animation9 = FuncAnimation(power_figure, power, interval=200)

plt.show()

# ****************************************** Saving the figures *******************************************************


figure1_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Temperatures' + '_' + str(block_ID) + '_' + str(experiment_ID) + '.pdf'
temperatures_figure.savefig(figure1_name)

figure2_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Flow_rate' + '_' + str(block_ID) + '_' + str(experiment_ID) + '.pdf'
flow_figure.savefig(figure2_name)

figure3_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Block_V_drop' + '_' + str(block_ID) + '_' + str(experiment_ID) + '.pdf'
volt_drop_block_figure.savefig(figure3_name)

figure4_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Pressure_drop' + '_' + str(block_ID) + '_' + str(experiment_ID) + '.pdf'
diff_press_figure.savefig(figure4_name)

figure5_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Heating_current' + '_' + str(block_ID) + '_' + str(experiment_ID) + '.pdf'
heating_current_figure.savefig(figure5_name)
#
# figure6_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Pressure_drop_filter' + '_' + str(experiment) + '.pdf'
# figure6.savefig(figure6_name)
#
figure7_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Voltage_plates' + '_' + str(block_ID) + '_' + str(experiment_ID) + '.pdf'
volt_plates_figure.savefig(figure7_name)

figure8_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Water_temperatures' + '_' + str(block_ID) + '_' + str(experiment_ID) + '.pdf'
water_temperatures_figure.savefig(figure8_name)

# figure9_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Power' + '_' + str(block_ID) + '_' + str(experiment_ID) + '.pdf'
# power_figure.savefig(figure9_name)

# **************************************** End of script ************************************************************

