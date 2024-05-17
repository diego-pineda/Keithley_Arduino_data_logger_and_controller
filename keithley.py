
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

# Initialize the Keithleys

multimeter1 = pyvisa.ResourceManager().open_resource('GPIB0::16::INSTR')# Connect to the keithley and set it to a variable named multimeter.
multimeter2 = pyvisa.ResourceManager().open_resource('GPIB0::20::INSTR')# Connect to the keithley and set it to a variable named multimeter.


# #**************************************
# Define some parameters
dataElements = "READ"
number_data_elements_per_chan = len(dataElements)  #timestamp, reading and channel tag
chanList = "(@1:5)"
number_channels_in_scan = 5
number_of_scans = 1
bufferSize = number_of_scans * number_channels_in_scan
debug = 1
# #**************************************
# TRACe is for controlling the storage in the buffer
# INIT has to do with the start of the trigger model
# FORM is to define the form of the data to transfer to the pc
# TRIG has to do with triggering a sequence or operation mode
cmd_list = ["*RST",
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
            "ROUT:SCAN:TSO IMM",
            "FUNC 'FRESistance'",
            "FRES:NPLC 1",
            "ROUT:SCAN " + chanList,
            "ROUT:SCAN:LSEL INT",
            "STAT:MEAS:ENAB 512",
            "*SRE 1",
            "*OPC?"]

for cmd in cmd_list:
    multimeter2.write(cmd)

print(multimeter2.read())   #read the *OPC? response
print("*OPC received; finished setting up 2000....starting the scan")

# *************************************************************

multimeter1.write(":SENSe:FUNCtion 'VOLTage:DC'") # Set the keithley to measure Voltage DC
# multimeter2.write(":SENSe:FUNCtion 'FRESistance'") # Set the keithley  to measure 4-wire resistance



# multimeter.write(":ROUTe:MULTiple:CLOSe (@1,2)")  # Set the keithley to measure channel 1 and 2 of card 1
# multimeter.write(":SENSe:FUNCtion 'VOLTage:DC'") # Set the keithley to measure Voltage DC.
# multimeter.write(':TRACe:CLEar')
# multimeter.write(':TRACe:POINts 4')
# multimeter.write(':TRACe:FEED SENS')
# multimeter.write("TRACE:FEED:CONT NEXT")
# multimeter.write("FORM:DATA ASCII")
# # print(multimeter.query(':TRACe:DATA?'))
# # print(multimeter.query(':SYSTem:ERRor?'))


time_flow_rate = []  # Create an empty list to store time values in.
time_volt_drop_block = []
time_current_shunt = []
time_diff_pressure = []
flow_rate = []  # Create an empty list to store flow rate values in.
voltage_drop_block = []
current_shunt = []
diff_pressure = []

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

startTime = time.time()  # Create a variable that holds the starting timestamp.

flag = 1

def normal():
    global flag

# Create a while loop that continuously measures and plots data from the keithley forever.
    while flag == 1:

        # -------------------- Multimeter 1 --------------------

        multimeter1.write(":ROUTe:CLOSe (@1)")  # Set the keithley to measure channel 1 of card 1
        time.sleep(1)
        voltageReading1 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        flow_rate.append(voltageReading1)
        time_flow_rate.append(float(time.time() - startTime))
        time.sleep(0.5)

        multimeter1.write(":ROUTe:CLOSe (@2)")  # Set the keithley to measure channel 2 of card 1
        time.sleep(1) # 0.05 Interval to wait between collecting data points.
        voltageReading2 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        voltage_drop_block.append(voltageReading2)
        time_volt_drop_block.append(float(time.time() - startTime))
        time.sleep(0.5)

        multimeter1.write(":ROUTe:CLOSe (@3)")  # Set the keithley to measure channel 3 of card 1
        time.sleep(1)
        voltageReading3 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        current_shunt.append(voltageReading3 / (50e-3 / 200))  # Voltage reading is converted to current by using resistance of shunt
        time_current_shunt.append(float(time.time() - startTime))
        time.sleep(0.5)

        multimeter1.write(":ROUTe:CLOSe (@4)")  # Set the keithley to measure channel 4 of card 1
        time.sleep(1) # 0.05 Interval to wait between collecting data points.
        voltageReading4 = float(multimeter1.query(':SENSe:DATA:FRESh?').split(',')[0]) # [:-2]Read and process data from the keithley.
        diff_pressure.append((voltageReading4 / 5 - 0.04) / 0.009)
        time_diff_pressure.append(float(time.time() - startTime))
        time.sleep(0.5)

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

        # ***************************** test *********************************
        cmd_list = ["*RST",
                    "*CLS",
                    "TRAC:CLE",
                    "INIT:CONT OFF",
                    "TRACE:POINTS " + str(bufferSize),
                    "TRACE:FEED SENS",
                    "TRACE:FEED:CONT NEXT",
                    "FORM:DATA ASCII",
                    "FORM:ELEM " + dataElements,
                    "TRIG:COUN " + str(number_of_scans),
                    "SAMP:COUN " + str(number_channels_in_scan),
                    "TRIG:SOUR IMM",
                    "ROUT:SCAN:TSO IMM",
                    "FUNC 'FRESistance'",
                    "FRES:NPLC 1",
                    "ROUT:SCAN " + chanList,
                    "ROUT:SCAN:LSEL INT",
                    "STAT:MEAS:ENAB 512",
                    "*SRE 1",
                    "*OPC?"]

        for cmd in cmd_list:
            multimeter2.write(cmd)


        print(multimeter2.read())   #read the *OPC? response
        print("*OPC received; finished setting up 2000....starting the scan")

        multimeter2.write("INIT")

        #detect the scan is finished
        #repeat until the SRQ bit is set
        still_running = True
        status_byte = 0

        while still_running:
            status_byte = int(multimeter2.query('*STB?'))
            if debug: print(status_byte)
            if (status_byte and 64) == 64:
                still_running = False
            time.sleep(0.250)  #250msec pause before asking again

        print("Scan is done, status byte: " + str(status_byte))



        #read the data to this PC

        #how many are in the buffer to send to us?
        num_data_pts = multimeter2.query(":TRAC:POINTS?")
        print("Number of buffer pts: " + str(num_data_pts))
        print("***************")
        print()

        #ask for all the data
        raw_data = multimeter2.query("TRAC:DATA?")
        print(raw_data)
        print("***************")
        #print()

        #split, parse, etc.
        """
        raw_data will be comma delimted string of
        timestamp, reading, chanTag, timestamp, reading, chanTag,... etc.
        """
        raw_data_array = raw_data.split(',')
        # time.sleep(3)
        multimeter2.write("ABORt")
        # ***************************************************************************


        if flag == False:
            print('Data will be saved in the file: {}'.format(output_file_name))

    # ****************** Writing data to file *****************************

    # output_dataframe = pd.DataFrame({'t_block_1': time_block_1,
    #                                  'T_block_1': T_block_1,
    #                                  't_water_in': time_water_in,
    #                                  'T_water_in': T_water_in,
    #                                  't_block_2': time_block_2,
    #                                  'T_block_2': T_block_2,
    #                                  't_water_out': time_water_out,
    #                                  'T_water_out': T_water_out,
    #                                  't_amb': time_amb,
    #                                  'T_amb': T_amb,
    #                                  't_flow': time_flow_rate,
    #                                  'V_flow': flow_rate,
    #                                  't_block_prob': time_volt_drop_block,
    #                                  'V_block_prob': voltage_drop_block,
    #                                  't_current_shunt': time_current_shunt,
    #                                  'Current_shunt': current_shunt,
    #                                  't_diff_press': time_diff_pressure,
    #                                  'Diff_press': diff_pressure
    #                                  })
    #
    # output_dataframe.to_csv(output_file_name, sep="\t", index=False)

    # **********************************************************************

def get_input():
    global flag
    keystrk=input('Press the enter key to stop recording data. \n')  # \n
    # thread doesn't continue until key is pressed
    print('The recording of data will stop now.', keystrk)
    flag = False
    # print('flag is now:', flag)


n = threading.Thread(target=normal)
i = threading.Thread(target=get_input)
n.start()
i.start()

# **************************************** Plotting data in (almost) real time *****************************************
#
# x_data, y_data = [], []
#
# figure = plt.figure(figsize=(6, 4))
# plt.xlabel('Elapsed Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# plt.ylabel('Temperature (C)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# line1, = plt.plot(x_data, y_data, 'b-')  # T water in
# line2, = plt.plot(x_data, y_data, 'g-')  # T water out
# line3, = plt.plot(x_data, y_data, 'k-')  # T block 1
# line4, = plt.plot(x_data, y_data, 'r-')  # T block 2
# line7, = plt.plot(x_data, y_data, 'm-')  # Ambient temperature
#
# figure2 = plt.figure(figsize=(6, 4))
# plt.xlabel('Elapsed Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# plt.ylabel('Flow_rate (Lpm)')  # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# line5, = plt.plot(x_data, y_data, 'b-')
#
# figure3 = plt.figure(figsize=(8, 4))
# plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# plt.ylabel('Voltage drop MCM block (mV)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# line6, = plt.plot(x_data, y_data, 'r-')
#
# figure4 = plt.figure(figsize=(8, 4))
# plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# plt.ylabel('Differential_pressure (kPa)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# line8, = plt.plot(x_data, y_data, 'r-')
#
# figure5 = plt.figure(figsize=(8, 4))
# plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# plt.ylabel('Current (A)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# line9, = plt.plot(x_data, y_data, 'k-')
#
# def update(frame):
#     line1.set_data(time_water_in, T_water_in)
#     line2.set_data(time_water_out, T_water_out)
#     line3.set_data(time_block_1, T_block_1)
#     line4.set_data(time_block_2, T_block_2)
#     line7.set_data(time_amb, T_amb)
#     figure.gca().relim()
#     figure.gca().autoscale_view()
#     return line1, line2, line3, line4, line7
#
# def update2(frame):
#     line5.set_data(time_flow_rate, flow_rate)
#     figure2.gca().relim()
#     figure2.gca().autoscale_view()
#     return line5
#
# def update3(frame):
#     line6.set_data(time_volt_drop_block, voltage_drop_block)
#     figure3.gca().relim()
#     figure3.gca().autoscale_view()
#     return line6
#
# def update4(frame):
#     line8.set_data(time_diff_pressure, diff_pressure)
#     figure4.gca().relim()
#     figure4.gca().autoscale_view()
#     return line8
#
# def update5(frame):
#     line9.set_data(time_current_shunt, current_shunt)
#     figure5.gca().relim()
#     figure5.gca().autoscale_view()
#     return line9
#
# animation = FuncAnimation(figure, update, interval=200)
# animation2 = FuncAnimation(figure2, update2, interval=200)
# animation3 = FuncAnimation(figure3, update3, interval=200)
# animation4 = FuncAnimation(figure4, update4, interval=200)
# animation5 = FuncAnimation(figure5, update5, interval=200)
#
# plt.show()

# ****************************************** Saving the figures *******************************************************
#
#
# figure1_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Temperatures' + '_' + str(experiment) + '.pdf'
# figure.savefig(figure1_name)
#
# figure2_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Flow_rate' + '_' + str(experiment) + '.pdf'
# figure2.savefig(figure2_name)
#
# figure3_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Sample_V_drop' + '_' + str(experiment) + '.pdf'
# figure3.savefig(figure3_name)
#
# figure4_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Pressure_drop' + '_' + str(experiment) + '.pdf'
# figure4.savefig(figure4_name)
#
# figure5_name = './Output_figures/' + str(date.today().strftime("%d%b%Y")) + '_' + 'Current_shunt' + '_' + str(experiment) + '.pdf'
# figure5.savefig(figure5_name)

# **************************************** End of script ************************************************************

