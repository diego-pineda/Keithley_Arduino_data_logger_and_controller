
import pyvisa
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from scipy.interpolate import CubicSpline
from datetime import date
import threading


PT100_data_frame = pd.read_csv('./PT100_Table.txt', sep='\t', header=None)
PT100_array = pd.DataFrame(PT100_data_frame).to_numpy()
PT100 = CubicSpline(PT100_array[:, 1], PT100_array[:, 0])

block = 'block_1b'
experiment = 'heat_trans_Diff_Press'
output_file_name = './' + str(date.today()) + block + '-' + experiment + '.csv'

# Initialize the keithley and create some useful variables
multimeter1 = pyvisa.ResourceManager().open_resource('GPIB0::16::INSTR')# Connect to the keithley and set it to a variable named multimeter.
multimeter2 = pyvisa.ResourceManager().open_resource('GPIB0::20::INSTR')# Connect to the keithley and set it to a variable named multimeter.

# multimeter.write(":ROUTe:CLOSe (@1)")  # Set the keithley to measure channel 1 of card 1
multimeter1.write(":SENSe:FUNCtion 'VOLTage:DC'") # Set the keithley to measure Voltage DC
multimeter2.write(":SENSe:FUNCtion 'FRESistance'") # Set the keithley  to measure 4-wire resistance
# print(multimeter.query(':SENSe:DATA:FRESh?'))


# multimeter.write(":ROUTe:MULTiple:CLOSe (@1,2)")  # Set the keithley to measure channel 1 and 2 of card 1
# multimeter.write(":SENSe:FUNCtion 'VOLTage:DC'") # Set the keithley to measure Voltage DC.
# multimeter.write(':TRACe:CLEar')
# multimeter.write(':TRACe:POINts 4')
# multimeter.write(':TRACe:FEED SENS')
# multimeter.write("TRACE:FEED:CONT NEXT")
# multimeter.write("FORM:DATA ASCII")
# # print(multimeter.query(':TRACe:DATA?'))
# # print(multimeter.query(':SYSTem:ERRor?'))


time1List = []  # Create an empty list to store time values in.
time2List = []
time3List = []
time4List = []
voltageList1 = []  # Create an empty list to store temperature values in.
voltageList2 = []
voltageList3 = []
voltageList4 = []

T_water_in = []
time_water_in = []
T_water_out = []
time_water_out = []
T_block_1 = []
time_block_1 = []
T_block_2 = []
time_block_2 = []
T_amb = []
time_amb = []

# multimeter2.write(":ROUTe:CLOSe (@401)") # Set the keithley to measure channel 1 of card 1
# multimeter2.write(":SENSe:FUNCtion 'TEMPerature'") # Set the keithley to measure temperature.
# temperatureList2 = [] # Create an empty list to store temperature values in.

startTime = time.time()  # Create a variable that holds the starting timestamp.

# Setup the plot

# plt.figure(1, figsize=(5, 5)) # Initialize a matplotlib figure
# plt.xlabel('Elapsed Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# # plt.xticks(fontsize=18) # Set the font size of the x tick numbers to 18pt
# plt.ylabel('Temperature (C)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# plt.yticks(fontsize=18) # Set the font size of the y tick numbers to 18pt


# plt.figure(2, figsize=(5, 5)) # Initialize a matplotlib figure
# plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# # plt.xticks(fontsize=18) # Set the font size of the x tick numbers to 18pt
# plt.ylabel('Voltage2 (V)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# # plt.yticks(fontsize=18) # Set the font size of the y tick numbers to 18pt


flag = 1


def normal():
    global flag

# Create a while loop that continuously measures and plots data from the keithley forever.
    while flag == 1:

        multimeter1.write(":ROUTe:CLOSe (@1)")  # Set the keithley to measure channel 1 of card 1
        time.sleep(1)
        voltageReading1 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        voltageList1.append(voltageReading1)  # Append processed data to the temperature list
        time1List.append(float(time.time() - startTime)) # Append time values to the time list
        time.sleep(0.5)
        #
        multimeter1.write(":ROUTe:CLOSe (@2)")  # Set the keithley to measure channel 2 of card 1
        time.sleep(1) # 0.05 Interval to wait between collecting data points.
        voltageReading2 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        voltageList2.append(voltageReading2)  # Append processed data to the temperature list
        time2List.append(float(time.time() - startTime)) # Append time values to the time list
        time.sleep(0.5)

        multimeter1.write(":ROUTe:CLOSe (@3)")  # Set the keithley to measure channel 1 of card 1
        time.sleep(1)
        voltageReading3 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        voltageList3.append(voltageReading3)  # Append processed data to the temperature list
        time3List.append(float(time.time() - startTime)) # Append time values to the time list
        time.sleep(0.5)
        #
        multimeter1.write(":ROUTe:CLOSe (@4)")  # Set the keithley to measure channel 2 of card 1
        time.sleep(1) # 0.05 Interval to wait between collecting data points.
        voltageReading4 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        voltageList4.append((voltageReading4/5-0.04)/0.009)  # Append processed data to the temperature list
        time4List.append(float(time.time() - startTime)) # Append time values to the time list
        time.sleep(0.5)

        # # temperatureReading2 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0][:-2])  # Read and process data from the keithley.
        # # temperatureList2.append(temperatureReading2 * 10)  # Append processed data to the temperature list
        #
        # time.sleep(0.25)

        # Multimeter 2

        multimeter2.write(":ROUTe:CLOSe (@1)")  # Set the keithley to measure channel 1 of card 1
        time.sleep(1) # 0.05 Interval to wait between collecting data points.
        test_data_string = multimeter2.query(':SENSe:DATA:FRESh?')
        RTD_1 = float(test_data_string.split(',')[0])
        print(float(test_data_string.split(',')[0]))
        T_block_1.append(float(PT100(RTD_1)))
        time_block_1.append(float(time.time() - startTime))
        time.sleep(0.5)

        multimeter2.write(":ROUTe:CLOSe (@2)")  # Set the keithley to measure channel 1 of card 1
        time.sleep(1) # 0.05 Interval to wait between collecting data points.
        RTD_2 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0])
        T_water_in.append(float(PT100(RTD_2)))
        time_water_in.append(float(time.time() - startTime))
        time.sleep(0.5)

        multimeter2.write(":ROUTe:CLOSe (@3)")  # Set the keithley to measure channel 1 of card 1
        time.sleep(1) # 0.05 Interval to wait between collecting data points.
        RTD_3 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0])
        T_block_2.append(float(PT100(RTD_3)))
        time_block_2.append(float(time.time() - startTime))
        time.sleep(0.5)

        multimeter2.write(":ROUTe:CLOSe (@4)")  # Set the keithley to measure channel 1 of card 1
        time.sleep(1) # 0.05 Interval to wait between collecting data points.
        RTD_4 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0])
        T_water_out.append(float(PT100(RTD_4)))
        time_water_out.append(float(time.time() - startTime))
        time.sleep(0.5)

        multimeter2.write(":ROUTe:CLOSe (@5)")  # Set the keithley to measure channel 1 of card 1
        time.sleep(1) # 0.05 Interval to wait between collecting data points.
        RTD_5 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0])
        T_amb.append(float(PT100(RTD_5)))
        time_amb.append(float(time.time() - startTime))
        time.sleep(0.5)

        # voltageReading3 = float(multimeter2.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        # voltageList3.append(voltageReading3) # Append processed data to the temperature list
        # time3List.append(float(time.time() - startTime)) # Append time values to the time list



        # plt.plot(timeList, temperatureList2, color='blue', linewidth=2) # Plot the collected data with time on the x axis and temperature on the y axis.

        # plt.show()
        # plt.figure(2)
        # plt.plot(time2List, voltageList2, color='red',linewidth=2)  # Plot the collected data with time on the x axis and temperature on the y axis.
        # #

        if flag==False:
            print('Data will be saved in the file: {}'.format(output_file_name))

    # print(T_water_in)

    output_dataframe = pd.DataFrame({'t_block_1': time_block_1,
                                     'T_block_1': T_block_1,
                                     't_water_in': time_water_in,
                                     'T_water_in': T_water_in,
                                     't_block_2': time_block_2,
                                     'T_block_2': T_block_2,
                                     't_water_out': time_water_out,
                                     'T_water_out': T_water_out,
                                     't_amb': time_amb,
                                     'T_amb': T_amb,
                                     't_flow': time1List,
                                     'V_flow': voltageList1,
                                     't_block_prob': time2List,
                                     'V_block_prob': voltageList2
                                     })



    output_dataframe.to_csv(output_file_name, sep="\t", index=False)


def get_input():
    global flag
    keystrk=input('Press the enter key to stop recording data. \n')  # \n
    # thread doesn't continue until key is pressed
    print('The recording of data will stop now.', keystrk)
    flag = False
    # print('flag is now:', flag)


n=threading.Thread(target=normal)
i=threading.Thread(target=get_input)
n.start()
i.start()


    # plt.figure(1)
    # plt.xlabel('Elapsed Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
    # # # plt.xticks(fontsize=18) # Set the font size of the x tick numbers to 18pt
    # plt.ylabel('Temperature (C)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
    # # plt.yticks(fontsize=18) # Set the font size of the y tick numbers to 18pt
    # ln, = plt.plot([])
    # plt.ion()
    # plt.show()

    # plt.plot(time_water_in, T_water_in, color='blue', linewidth=2)  # Plot the collected data with time on the x axis and temperature on the y axis.
    # plt.plot(time_water_out, T_water_out, color='purple', linewidth=2)  # Plot the collected data with time on the x axis and temperature on the y axis.
    # plt.plot(time_block_1, T_block_1, color='orange', linewidth=2)  # Plot the collected data with time on the x axis and temperature on the y axis.
    # plt.plot(time_block_2, T_block_2, color='red', linewidth=2)  # Plot the collected data with time on the x axis and temperature on the y axis.
    # plt.pause(0.01) # This command is required for live plotting. This allows the code to keep running while the plot is shown.


    # while True:
    #     plt.pause(6)
    #     ln.set_xdata(time_water_in)
    #     ln.set_ydata(T_water_in)
    #     plt.draw()



x_data, y_data = [], []

figure = plt.figure(figsize=(6, 4))
plt.xlabel('Elapsed Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Temperature (C)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line1, = plt.plot(x_data, y_data, 'b-')
line2, = plt.plot(x_data, y_data, 'g-')
line3, = plt.plot(x_data, y_data, 'k-')
line4, = plt.plot(x_data, y_data, 'r-')
line7, = plt.plot(x_data, y_data, 'm-')

figure2 = plt.figure(figsize=(6, 4))
plt.xlabel('Elapsed Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Voltage (V)')  # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line5, = plt.plot(x_data, y_data, 'b-')

figure3 = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Voltage (mV)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line6, = plt.plot(x_data, y_data, 'r-')

figure4 = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Differential_pressure (kPa)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line8, = plt.plot(x_data, y_data, 'r-')

figure5 = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Voltage (mV)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
line9, = plt.plot(x_data, y_data, 'k-')

def update(frame):
    # x_data.append(time_water_in)
    # y_data.append(T_water_in)
    line1.set_data(time_water_in, T_water_in)
    line2.set_data(time_water_out, T_water_out)
    line3.set_data(time_block_1, T_block_1)
    line4.set_data(time_block_2, T_block_2)
    line7.set_data(time_amb, T_amb)
    figure.gca().relim()
    figure.gca().autoscale_view()
    return line1, line2, line3, line4, line7

def update2(frame):
    # x_data.append(time_water_in)
    # y_data.append(T_water_in)
    line5.set_data(time1List, voltageList1)
    # line6.set_data(time2List, voltageList2)
    figure2.gca().relim()
    figure2.gca().autoscale_view()
    return line5

def update3(frame):
    # x_data.append(time_water_in)
    # y_data.append(T_water_in)
    line6.set_data(time2List, voltageList2)
    figure3.gca().relim()
    figure3.gca().autoscale_view()
    return line6

def update4(frame):
    # x_data.append(time_water_in)
    # y_data.append(T_water_in)
    line8.set_data(time4List, voltageList4)
    figure4.gca().relim()
    figure4.gca().autoscale_view()
    return line8

def update5(frame):
    # x_data.append(time_water_in)
    # y_data.append(T_water_in)
    line9.set_data(time3List, voltageList3)
    figure5.gca().relim()
    figure5.gca().autoscale_view()
    return line9

animation = FuncAnimation(figure, update, interval=200)
animation2 = FuncAnimation(figure2, update2, interval=200)
animation3 = FuncAnimation(figure3, update3, interval=200)
animation4 = FuncAnimation(figure4, update4, interval=200)
animation5 = FuncAnimation(figure5, update5, interval=200)

plt.show()


#
# print(voltageList1)
# print(voltageList2)
# print(voltageList3)
# ------------------------------------------------------------- new code ________________________
#
#
# #**************************************
# #define some parameters
# dataElements = "TST, READ, CHAN"
# number_data_elements_per_chan = len(dataElements)  #timestamp, reading and channel tag
# chanList = "(@1:2)"
# number_channels_in_scan = 2
# number_of_scans = 6
# bufferSize = number_of_scans * number_channels_in_scan
# debug = 1
# #**************************************
#
#
# cmd_list = ["*RST",
#             "*CLS",
#             "SYSTEM:PRESET",
#             "TRAC:CLE",
#             "INIT:CONT OFF",
#             "TRACE:POINTS " + str(bufferSize),
#             "TRACE:FEED SENS",
#             "TRACE:FEED:CONT NEXT",
#             "FORM:DATA ASCII",
#             "FORM:ELEM " + dataElements,
#             "TRIG:COUN " + str(number_of_scans),
#             "SAMP:COUN " + str(number_channels_in_scan),
#             "TRIG:SOUR IMM",
#             "ROUT:SCAN:TSO IMM",
#             "FUNC 'VOLTage:DC', " + chanList,
#             "ROUT:SCAN " + chanList,
#             "ROUT:SCAN:LSEL INT",
#             "STAT:MEAS:ENAB 512",
#             "*SRE 1",
#             "*OPC?"]
#
# for cmd in cmd_list:
#     multimeter.write(cmd)
#
# print(multimeter.read())   #read the *OPC? response
# print("*OPC received; finished setting up 2700....starting the scan")
# multimeter.write("INIT")
#
# #detect the scan is finished
# #repeat until the SRQ bit is set
# still_running = True
# status_byte = 0
#
# while still_running:
#     status_byte = int(multimeter.query('*STB?'))
#     if debug: print(status_byte)
#     if (status_byte and 64) == 64:
#         still_running = False
#     time.sleep(0.250)  #250msec pause before asking again
#
# print("Scan is done, status byte: " + str(status_byte))
#
#
#
# #read the data to this PC
#
# #how many are in the buffer to send to us?
# num_data_pts = multimeter.query(":TRAC:POINTS?")
# print("Number of buffer pts: " + str(num_data_pts))
# print("***************")
# print()
#
# #ask for all the data
# raw_data = multimeter.query("TRAC:DATA?")
# print(raw_data)
# print("***************")
# #print()
#
# #split, parse, etc.
# """
# raw_data will be comma delimted string of
# timestamp, reading, chanTag, timestamp, reading, chanTag,... etc.
# """
# raw_data_array = raw_data.split(',')