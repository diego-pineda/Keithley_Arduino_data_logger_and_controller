
import pyvisa
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from scipy.interpolate import CubicSpline
from datetime import date
import threading
name = 'Diego_Pineda'

PT100_data_frame = pd.read_csv('./PT100_Table.txt', sep='\t', header=None)
PT100_array = pd.DataFrame(PT100_data_frame).to_numpy()
PT100 = CubicSpline(PT100_array[:, 1], PT100_array[:, 0])

print('Enter block ID (use underscore between words, no spaces): ')
block = input()
print('Enter experiment name (use underscore between words, no spaces):')
experiment = input()
output_file_name = './Sensor_data/' + str(date.today().strftime("%d%b%Y")) + '_' + block + '_' + experiment + '.csv'
print('Output file name: ' + output_file_name)

# **********************************    Initialize the Keithleys    *******************************************

multimeter1 = pyvisa.ResourceManager().open_resource('GPIB0::16::INSTR')# Connect to the keithley and set it to a variable named multimeter.
multimeter2 = pyvisa.ResourceManager().open_resource('GPIB0::20::INSTR')# Connect to the keithley and set it to a variable named multimeter.
multimeter3 = pyvisa.ResourceManager().open_resource('GPIB0::22::INSTR')# Connect to the Agilent and set it to a variable named multimeter.

# ******************************* Setting up Multimeter 1 *****************************************

# Define some parameters
dataElements = "READ"
#number_data_elements_per_chan = len(dataElements)  # saving only reading, time stamp apparently not possible
chanList = "(@1:5)"
number_channels_in_scan = 5
number_of_scans = 1
bufferSize = number_of_scans * number_channels_in_scan
# debug = 1

DMM1_cmd_list = ["*RST",
                 "*CLS",
                 "SYSTEM:PRESET",
                 "TRAC:CLE",
                 "INIT:CONT OFF",
                 "TRACE:POINTS " + str(bufferSize),
                 "TRACE:FEED SENS",
                 "TRACE:FEED:CONT NEXT",
                 "FORM:DATA ASCII",
                 "FORM:ELEM " + dataElements,
                 "TRIG:COUN" + str(number_of_scans),
                 "SAMP:COUN " + str(number_channels_in_scan),
                 "TRIG:SOUR IMM",
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
            "TRACE:POINTS " + str(bufferSize),
            "TRACE:FEED SENS",
            "TRACE:FEED:CONT NEXT",
            "FORM:DATA ASCII",
            "FORM:ELEM " + dataElements,
            "TRIG:COUN" + str(number_of_scans),
            "SAMP:COUN " + str(number_channels_in_scan),
            "TRIG:SOUR IMM",
            "FUNC 'FRESistance'",
            "FRES:NPLC 7",  # Integration rate. From 0.1 to 10. 0.1 fast rate, 1 medium, 10 slow. Influences scan rate.
            "FRES:RANG 1000",
            "ROUT:SCAN " + chanList,
            "ROUT:SCAN:LSEL INT",
            "STAT:MEAS:ENAB 512",
            "*SRE 1",
            "*OPC?"]

for cmd in DMM2_cmd_list:
    multimeter2.write(cmd)

print(multimeter2.read())   #read the *OPC? response
print("*OPC received; finished setting up Keithley 2000 Multimeter 2")

# *************************************************************

# multimeter1.write(":SENSe:FUNCtion 'VOLTage:DC'") # Set the keithley to measure Voltage DC
# multimeter2.write(":SENSe:FUNCtion 'FRESistance'") # Set the keithley  to measure 4-wire resistance
multimeter3.write(":SENSe:FUNCtion 'VOLTage:DC'") # Set the keithley to measure Voltage DC
multimeter3.write("VOLTage:DC:NPLC 7") # Set the keithley to measure Voltage DC
multimeter3.write("TRIG:SOUR IMM")


# Variables multimeter 1

time_flow_rate = []  # Create an empty list to store time values in.
time_volt_drop_block = []
time_current_shunt = []
time_diff_pressure = []
time_diff_pressure_filter = []

flow_rate = []  # Create an empty list to store flow rate values in.
voltage_drop_block = []
current_shunt = []
diff_pressure = []
diff_pressure_filter = []

# Variables multimeter 2

time_water_in = []
time_water_out = []
time_block_1 = []
time_block_2 = []
time_amb = []

T_water_in = []
T_water_out = []
T_block_1 = []
T_block_2 = []
T_amb = []

startTime = time.time()  # Create a variable that holds the starting timestamp.
flag = 1


def normal():
    global flag

# Create a while loop that continuously measures and plots data from the keithley forever.
    while flag == 1:

        # -------------------- Multimeter 1 --------------------

        # multimeter1.write(":ROUTe:CLOSe (@1)")  # Set the keithley to measure channel 1 of card 1
        # time.sleep(1)
        # voltageReading1 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        # flow_rate.append(voltageReading1)
        # time_flow_rate.append(float(time.time() - startTime))
        # time.sleep(0.5)
        #
        # multimeter1.write(":ROUTe:CLOSe (@2)")  # Set the keithley to measure channel 2 of card 1
        # time.sleep(1) # 0.05 Interval to wait between collecting data points.
        # voltageReading2 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        # voltage_drop_block.append(voltageReading2)
        # time_volt_drop_block.append(float(time.time() - startTime))
        # time.sleep(0.5)
        #
        # multimeter1.write(":ROUTe:CLOSe (@3)")  # Set the keithley to measure channel 3 of card 1
        # time.sleep(1)
        # voltageReading3 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        # current_shunt.append(voltageReading3 / (50e-3 / 200))  # Voltage reading is converted to current by using resistance of shunt
        # time_current_shunt.append(float(time.time() - startTime))
        # time.sleep(0.5)
        #
        # multimeter1.write(":ROUTe:CLOSe (@4)")  # Set the keithley to measure channel 4 of card 1
        # time.sleep(1) # 0.05 Interval to wait between collecting data points.
        # voltageReading4 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        # diff_pressure.append((voltageReading4 / 5 - 0.04) / 0.009)
        # time_diff_pressure.append(float(time.time() - startTime))
        # time.sleep(0.5)

        multimeter1.write("INIT")
        DMM1_data = multimeter1.query("TRAC:DATA?")
        DMM1_time_stamp = float(time.time() - startTime)
        multimeter1.write("ABORt")
        multimeter1.write("TRAC:CLE")
        DMM1_data_array = [float(i) for i in DMM1_data.split(',')]

        # -------------------- Multimeter 3 --------------------

        # multimeter3.write(":ROUTe:CLOSe (@1)")  # Set the keithley to measure channel 1 of card 1
        # time.sleep(1)

        voltageReading1 = float(multimeter3.query('READ?').split(' ')[0]) # [:-2]Read and process data from the keithley. # :SENSe:DATA:FRESh?  .....  DATA:LAST?
        print(voltageReading1)
        flow_rate.append(voltageReading1)
        time_flow_rate.append(float(time.time() - startTime))
        # time.sleep(0.5)

        # flow_rate.append(DMM1_data_array[0])  # CH1.Multimeter_1
        voltage_drop_block.append(DMM1_data_array[1])  # CH2.Multimeter_1
        diff_pressure_filter.append((DMM1_data_array[2] / 5.07 - 0.04) / 0.009)  # CH3.Multimeter_1
        diff_pressure.append((DMM1_data_array[3]/ 5.07 - 0.04) / 0.009)  #  CH4.Multimeter_1 pressure drop across block
        current_shunt.append(DMM1_data_array[4]/ (50e-3 / 200))  # CH5.Multimeter_1 divided by the resist of the shunt

        # time_flow_rate.append(DMM1_time_stamp)
        time_volt_drop_block.append(DMM1_time_stamp)
        time_diff_pressure_filter.append(DMM1_time_stamp)
        time_diff_pressure.append(DMM1_time_stamp)
        time_current_shunt.append(DMM1_time_stamp)


        # -------------------- Multimeter 2 ------------------------

        # multimeter2.write(":ROUTe:CLOSe (@1)")  # Set the keithley to measure channel 1 of card 1
        # time.sleep(1) # 0.05 Interval to wait between collecting data points.
        # RTD_4 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0])
        # T_water_out.append(float(PT100(RTD_4)))
        # time_water_out.append(float(time.time() - startTime))
        # time.sleep(0.5)
        #
        # multimeter2.write(":ROUTe:CLOSe (@2)")  # Set the keithley to measure channel 1 of card 1
        # time.sleep(1) # 0.05 Interval to wait between collecting data points.
        # RTD_2 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0])
        # T_water_in.append(float(PT100(RTD_2)))
        # time_water_in.append(float(time.time() - startTime))
        # time.sleep(0.5)
        #
        # multimeter2.write(":ROUTe:CLOSe (@3)")  # Set the keithley to measure channel 1 of card 1
        # time.sleep(1) # 0.05 Interval to wait between collecting data points.
        # RTD_3 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0])
        # T_block_2.append(float(PT100(RTD_3)))
        # time_block_2.append(float(time.time() - startTime))
        # time.sleep(0.5)
        #
        # multimeter2.write(":ROUTe:CLOSe (@4)")  # Set the keithley to measure channel 1 of card 1
        # time.sleep(1) # 0.05 Interval to wait between collecting data points.
        # RTD_1 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0])
        # T_block_1.append(float(PT100(RTD_1)))
        # time_block_1.append(float(time.time() - startTime))
        # time.sleep(0.5)
        #
        # multimeter2.write(":ROUTe:CLOSe (@5)")  # Set the keithley to measure channel 1 of card 1
        # time.sleep(1) # 0.05 Interval to wait between collecting data points.
        # RTD_5 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0])
        # T_amb.append(float(PT100(RTD_5)))
        # time_amb.append(float(time.time() - startTime))
        # time.sleep(0.5)

        multimeter2.write("INIT")
        DMM2_data = multimeter2.query("TRAC:DATA?")
        DMM2_time_stamp = float(time.time() - startTime)
        multimeter2.write("ABORt")
        multimeter2.write("TRAC:CLE")
        DMM2_data_array = [float(i) for i in DMM2_data.split(',')]

        T_water_in.append(float(PT100(DMM2_data_array[0])))  # CH1.Multimeter_2
        T_water_out.append(float(PT100(DMM2_data_array[1])))  # CH2.Multimeter_2
        T_block_1.append(float(PT100(DMM2_data_array[2])))  # CH3.Multimeter_2
        T_block_2.append(float(PT100(DMM2_data_array[3])))  # CH4.Multimeter_2
        T_amb.append(float(PT100(DMM2_data_array[4])))  # CH5.Multimeter_2

        time_water_out.append(DMM2_time_stamp)
        time_water_in.append(DMM2_time_stamp)
        time_block_2.append(DMM2_time_stamp)
        time_block_1.append(DMM2_time_stamp)
        time_amb.append(DMM2_time_stamp)


        # ***************************************************************************

        if flag == False:
            print('Data will be saved in the file: {}'.format(output_file_name))

        # time.sleep(1)
    # ********************************* Returning multimeters to idle state ****************************************

    multimeter1.write(":ROUTe:SCAN:LSEL NONE")
    multimeter2.write(":ROUTe:SCAN:LSEL NONE")

    # ****************** Writing data to file *****************************

    output_dataframe = pd.DataFrame({'t_block_1': time_block_1,
                                     'T_block_1': T_block_1,
                                     't_block_2': time_block_2,
                                     'T_block_2': T_block_2,
                                     't_water_in': time_water_in,
                                     'T_water_in': T_water_in,
                                     't_water_out': time_water_out,
                                     'T_water_out': T_water_out,
                                     't_amb': time_amb,
                                     'T_amb': T_amb,
                                     't_flow': time_flow_rate,
                                     'V_flow': flow_rate,
                                     't_block_prob': time_volt_drop_block,
                                     'V_block_prob': voltage_drop_block,
                                     't_current_shunt': time_current_shunt,
                                     'Current_shunt': current_shunt,
                                     't_diff_press': time_diff_pressure,
                                     'Diff_press': diff_pressure,
                                     't_diff_press_filter': time_diff_pressure_filter,
                                     'Diff_press_filter': diff_pressure_filter
                                     })

    save_data = input('Do you want to save the data (y/n)?: \n')
    if save_data == 'y':
        output_dataframe.to_csv(output_file_name, sep="\t", index=False)
        print('Data file has been saved.\nClose the figures to start saving them and exit the program.')
    else:
        print('Data will not be saved!')

    # **********************************************************************


def get_input():
    global flag
    keystrk=input('Press the enter key to stop recording data. \n')  # \n
    # thread doesn't continue until key is pressed
    print('The recording of data will stop now.', keystrk)
    flag = False


n = threading.Thread(target=normal)
i = threading.Thread(target=get_input)
n.start()
i.start()

# **************************************** Plotting data in (almost) real time *****************************************

x_data, y_data = [], []

figure = plt.figure(figsize=(6, 4))
plt.xlabel('Elapsed Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Temperature (C)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line1, = plt.plot(x_data, y_data, 'b-')  # T water in
line2, = plt.plot(x_data, y_data, 'g-')  # T water out
line3, = plt.plot(x_data, y_data, 'k-')  # T block 1
line4, = plt.plot(x_data, y_data, 'r-')  # T block 2
line7, = plt.plot(x_data, y_data, 'm-')  # Ambient temperature

figure2 = plt.figure(figsize=(6, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Flow_rate (Lpm)')  # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line5, = plt.plot(x_data, y_data, 'b-')

figure3 = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Voltage drop MCM block (mV)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line6, = plt.plot(x_data, y_data, 'r-')

figure4 = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Differential_pressure (kPa)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line8, = plt.plot(x_data, y_data, 'r-')

figure5 = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Current (A)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line9, = plt.plot(x_data, y_data, 'k-')

figure6 = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Differential_pressure_filter (kPa)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line10, = plt.plot(x_data, y_data, 'r-')


def update(frame):
    # line1.set_data(time_water_in, T_water_in)
    line2.set_data(time_water_out, T_water_out)
    line3.set_data(time_block_1, T_block_1)
    line4.set_data(time_block_2, T_block_2)
    line7.set_data(time_amb, T_amb)
    figure.gca().relim()
    figure.gca().autoscale_view()
    return line1, line2, line3, line4, line7


def update2(frame):
    line5.set_data(time_flow_rate, flow_rate)
    figure2.gca().relim()
    figure2.gca().autoscale_view()
    return line5


def update3(frame):
    line6.set_data(time_volt_drop_block, voltage_drop_block)
    figure3.gca().relim()
    figure3.gca().autoscale_view()
    return line6


def update4(frame):
    line8.set_data(time_diff_pressure, diff_pressure)
    figure4.gca().relim()
    figure4.gca().autoscale_view()
    return line8


def update5(frame):
    line9.set_data(time_current_shunt, current_shunt)
    figure5.gca().relim()
    figure5.gca().autoscale_view()
    return line9


def update6(frame):
    line10.set_data(time_diff_pressure_filter, diff_pressure_filter)
    figure6.gca().relim()
    figure6.gca().autoscale_view()
    return line10

animation = FuncAnimation(figure, update, interval=200)
animation2 = FuncAnimation(figure2, update2, interval=200)
animation3 = FuncAnimation(figure3, update3, interval=200)
animation4 = FuncAnimation(figure4, update4, interval=200)
animation5 = FuncAnimation(figure5, update5, interval=200)
animation6 = FuncAnimation(figure6, update6, interval=200)

plt.show()

# ****************************************** Saving the figures *******************************************************


figure1_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Temperatures' + '_' + str(experiment) + '.pdf'
figure.savefig(figure1_name)

figure2_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Flow_rate' + '_' + str(experiment) + '.pdf'
figure2.savefig(figure2_name)

figure3_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Sample_V_drop' + '_' + str(experiment) + '.pdf'
figure3.savefig(figure3_name)

figure4_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Pressure_drop' + '_' + str(experiment) + '.pdf'
figure4.savefig(figure4_name)

figure5_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Current_shunt' + '_' + str(experiment) + '.pdf'
figure5.savefig(figure5_name)

figure6_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Pressure_drop_filter' + '_' + str(experiment) + '.pdf'
figure6.savefig(figure6_name)
# **************************************** End of script ************************************************************

