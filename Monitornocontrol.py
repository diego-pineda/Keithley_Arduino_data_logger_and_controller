#import gpib_ctypes
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

z = 0

PT100_data_frame = pd.read_csv('./PT100_Table.txt', sep='\t', header=None)
PT100_array = pd.DataFrame(PT100_data_frame).to_numpy()
PT100 = CubicSpline(PT100_array[:, 1], PT100_array[:, 0])


def get_user_input(prompt):
    """Prompts the user for input and ensures it follows the underscore format."""
    while True:
        user_input = input(prompt)
        if "_" in user_input and " " not in user_input:
            return user_input
        else:
            print("Invalid input. Please use underscores between words and no spaces.")


block_ID = get_user_input("Enter block ID (use underscore between words, no spaces): ")
experiment_ID = get_user_input("Enter experiment name (use underscore between words, no spaces): ")
Julabo_channel = get_user_input('Enter the serial port that the Julabo uses. The entered value should be something '
                                'like COM6. Julabo serial port: ')

output_file_name = './Sensor_data/' + str(date.today().strftime("%d%b%Y")) + '_' + block_ID + '_' + experiment_ID + '.csv'
ser = Serial(Julabo_channel)
print('Output file name: ' + output_file_name)
name = "michiel"


# **********************************    Initialize Multimeters    *******************************************

multimeter1 = pyvisa.ResourceManager().open_resource('GPIB0::16::INSTR')# Connect to a keithley 2000 and set it to a variable named multimeter1.
multimeter2 = pyvisa.ResourceManager().open_resource('GPIB0::20::INSTR')# Connect to the keithley 2000 and set it to a variable named multimeter2.
multimeter3 = pyvisa.ResourceManager().open_resource('GPIB0::22::INSTR')# Connect to the Agilent and set it to a variable named multimeter.
multimeter4 = pyvisa.ResourceManager().open_resource('GPIB0::12::INSTR')# Connect to the LakeShore 331 temperature controller and set it to a variable named multimeter4.
#multimeter4 = pyvisa.ResourceManager().open_resource('GPIB0::18::INSTR')# Connect to the Keithly and set it to a variable named multimeter.
#Sourcemeter = pyvisa.ResourceManager().open_resource('GPIB0::24::INSTR')# Connect to the keithly and set it to a variable named sourcemeter
ser.write(b"OUT_mode_05 1\r\n")

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

#********************************************************************

#multimeter4.write(":SENSe:FUNCtion 'FRESistance'") # Set the keithley to measure Voltage DC
#multimeter4.write("FRESistance:NPLC 7") # Set the keithley to measure Voltage DC
#multimeter4.write("TRIG:SOUR IMM")

# Variables multimeter 1

time_flow_rate = []  # Create an empty list to store time values in.
time_volt_drop_block = []
time_current_shunt = []
time_diff_pressure = []
time_diff_pressure_filter = []
time_voltage_drop_plates = []

flow_rate = []  # Create an empty list to store flow rate values in.
voltage_drop_block = []
current_shunt = []
diff_pressure = []
diff_pressure_filter = []
voltage_drop_plates = []

# Variables multimeter 2

time_water_in = []
time_water_out = []
time_block_top = []
time_block_bottom = []
time_block_left = []
time_block_right = []


T_water_in = []
T_water_out = []
T_block_top = []
T_block_bottom = []
T_block_left = []
T_block_right = []

time_ambient = []
T_ambient = []

startTime = time.time()  # Create a variable that holds the starting timestamp.
flag = 1

#Create a list for the temperature control
float_HB_thread_list = [0,1,2] #Elements are added so they can be compared in thread


def normal():
    global flag

# Create a while loop that continuously measures and plots data from the keithley forever.
    while flag == 1:

        # -------------------- Getting voltages of multimeter 1 --------------------

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

        # -------------------- Getting voltage of multimeter 3 --------------------

        # multimeter3.write(":ROUTe:CLOSe (@1)")  # Set the keithley to measure channel 1 of card 1
        # time.sleep(1)

        voltageReading1 = float(multimeter3.query('READ?').split(' ')[0]) # [:-2]Read and process data from the keithley. # :SENSe:DATA:FRESh?  .....  DATA:LAST?
        #print(voltageReading1)
        flow_rate.append(voltageReading1)
        time_flow_rate.append(float(time.time() - startTime))
        # time.sleep(0.5)

        voltage_drop_plates.append(DMM1_data_array[0])  # CH1.Multimeter_1
        voltage_drop_block.append(DMM1_data_array[1])  # CH2.Multimeter_1
        diff_pressure.append((DMM1_data_array[2]/ 5.07 - 0.04) / 0.009)  #  CH3.Multimeter_1 pressure drop across block
        current_shunt.append(DMM1_data_array[3]/ (50e-3 / 200))  # CH4.Multimeter_1 divided by the resist of the shunt
        # diff_pressure_filter.append((DMM1_data_array[4] / 5.07 - 0.04) / 0.009)  # CH5.Multimeter_1

        time_voltage_drop_plates.append(DMM1_time_stamp)
        time_volt_drop_block.append(DMM1_time_stamp)
        time_diff_pressure.append(DMM1_time_stamp)
        time_current_shunt.append(DMM1_time_stamp)
        # time_diff_pressure_filter.append(DMM1_time_stamp)


        # -------------------- Getting 4W resistances of multimeter 2 ------------------------

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

        # ----------------------- Getting temperature of inlet water from Julabo --------------------------

        ser.write(b"IN_pv_02\n") #Read temperature from julabo
        data_T_water_in = ser.readline() #Read the temperature of the heat bath
        decoded_T_water_in = data_T_water_in.decode("utf-8") #Decode the byte
        # print(decoded_T_water_in)

        # ------------------------ Calculating temperatures in Celsius -----------------------------------

        T_water_in.append(float(decoded_T_water_in))  # CH1.Multimeter_2
        T_water_out.append(float(PT100(DMM2_data_array[0])))  # CH1.Multimeter_2
        T_block_top.append(float(PT100(DMM2_data_array[1])))  # CH2.Multimeter_2
        T_block_bottom.append(float(PT100(DMM2_data_array[2])))  # CH3.Multimeter_2
        T_block_left.append(float(PT100(DMM2_data_array[3])))  # CH4.Multimeter_2
        T_block_right.append(float(PT100(DMM2_data_array[4])))  # CH5.Multimeter_2

        time_water_out.append(DMM2_time_stamp)
        time_water_in.append(DMM2_time_stamp)
        time_block_top.append(DMM2_time_stamp)
        time_block_bottom.append(DMM2_time_stamp)
        time_block_left.append(DMM2_time_stamp)
        time_block_right.append(DMM2_time_stamp)

        # ------------------------------ Getting ambient temperature from multimeter 4 --------------------------------

        #multimeter4.write("INIT")
        # DMM4_data = multimeter4.query("TRAC:DATA?")
        #print(DMM4_data)
        #DMM4_time_stamp = float(time.time() - startTime)
        #multimeter4.write("ABORt")
        #multimeter4.write("TRAC:CLE")
        #DMM4_data_array = [float(i) for i in DMM4_data.split(',')]

        #time_ambient.append(DMM4_time_stamp)
        #T_ambient.append(DMM4_data_array[0])

        DMM4_data = multimeter4.query("KRDG? B")
        Tamb = float(DMM4_data) - 273.15
        T_ambient.append(Tamb)
        DMM4_time_stamp = float(time.time() - startTime)
        time_ambient.append(DMM4_time_stamp)

        # ***************************************************************************

        if flag == False:
            print('Data will be saved in the file: {}'.format(output_file_name))

        # time.sleep(1)
    # ********************************* Returning multimeters to idle state ****************************************

    multimeter1.write(":ROUTe:SCAN:LSEL NONE")
    multimeter2.write(":ROUTe:SCAN:LSEL NONE")

    # ****************** Writing data to file *****************************

    output_dataframe = pd.DataFrame({'t_block_1': time_block_top,
                                     'T_block_1': T_block_top,
                                     't_block_2': time_block_bottom,
                                     'T_block_2': T_block_bottom,
                                     't_block_3' : time_block_left,
                                     'T_block_3' : T_block_left,
                                     't_block_4' : time_block_right,
                                     'T_block_4' : T_block_right,
                                     't_water_in': time_water_in,
                                     'T_water_in': T_water_in,
                                     't_water_out': time_water_out,
                                     'T_water_out': T_water_out,
                                     't_flow': time_flow_rate,
                                     'V_flow': flow_rate,
                                     't_block_prob': time_volt_drop_block,
                                     'V_block_prob': voltage_drop_block,
                                     't_current_shunt': time_current_shunt,
                                     'Current_shunt': current_shunt,
                                     't_diff_press': time_diff_pressure,
                                     'Diff_press': diff_pressure,
                                     #'t_diff_press_filter': time_diff_pressure_filter,
                                     #'Diff_press_filter': diff_pressure_filter,
                                     't_plates': time_voltage_drop_plates,
                                     'V_plates': voltage_drop_plates
                                     })

    # output_dataframe = pd.DataFrame({'t_block_1': time_block_1,
    #                                  'T_block_1': T_block_1,
    #                                  'T_block_2': T_block_2,
    #                                  'T_block_3' : T_block_3,
    #                                  'T_block_4' : T_block_4,
    #                                  'T_water_in': T_water_in,
    #                                  'T_water_out': T_water_out,
    #                                  't_flow': time_flow_rate,
    #                                  'V_flow': flow_rate,
    #                                  't_diff_press': time_diff_pressure,
    #                                  'Diff_press': diff_pressure
    #                                  })


    save_data = input('Do you want to save the data (y/n)?: \n')
    if save_data == 'y':
        output_dataframe.to_csv(output_file_name, sep="\t", index=False)
        print('Data file has been saved.\nClose the figures to start saving them and exit the program.')
    else:
        print('Data will not be saved!')

    # *******************************TEMPERATURE CONTROL THREAD***************************************
# def Julabo():
#     global flag
#
#     # Create a while loop that continuously measures and plots data from the keithley forever.
#     while flag == 1:
#
#
#         commandstring_prefix_Julabo = b"OUT_SP_00"  # Both used to create full command
#         commandstring_suffix_Julabo = b"\r\n"
#
#         HB_T = np.linspace(15, 35, 2001)
#         stringtempHB = str(HB_T[z])  # Turn float into string
#         bytetempHB = stringtempHB.encode()  # Turn string into byte
#         combinedstring_Julabo = commandstring_prefix_Julabo + bytetempHB + b" " + commandstring_suffix_Julabo  # Create full command
#         ser.write(combinedstring_Julabo)
#         time.sleep(1)
#         z = z+1

#def temp_and_flow_control():
 #   global flag

    #Starting the heat bath
  #  ser.write(b"OUT_mode_05 1\r\n")  # starting the bath
    #stringtempHB = str(HB_T[0])  # Turn float into string
    #bytetempHB = stringtempHB.encode()  # Turn string into byte
   # commandstring_prefix_Julabo = b"OUT_SP_00" #Both used to create full command
    #commandstring_suffix_Julabo = b"\r\n"
    #combinedstring_Julabo = commandstring_prefix_Julabo + bytetempHB + b" " + commandstring_suffix_Julabo  # Create full command
    #ser.write(combinedstring_Julabo)

    #Starting the pump
    #Sourcemeter.write('OUTPUT ON')
    #commandstring_prefix_Keithly = "SOURCE:VOLT"
    #stringflowrate = str(Sourcemeterlist[0])
    #combinedstring_Keithly = commandstring_prefix_Keithly + " " + stringflowrate
    #Sourcemeter.write(combinedstring_Keithly)



    #for z in range(len(HB_T)):
     #   if flag == 0: #To make sure it does not finish its sequence when the measurements are stopped
       #     break
      #  else:
            #ser.write(b"IN_pv_02\n") #Reading external temperature
            #HB_thread_data = ser.readline()

            #decoded_HB_thread_data = HB_thread_data.decode("utf-8") #Decode byte to string
            #stripped_HB_thread_data = decoded_HB_thread_data.strip() #Strip the letters from the string
            #float_HB_thread_data = float(stripped_HB_thread_data) #Turn stripped data into float

            #float_HB_thread_list.append(float_HB_thread_data) #Put most recent measurement in list
            #condition1 = abs(float_HB_thread_data - HB_T[z]) #Condition that temperature is close to set temperature
            #condition2 = abs(float_HB_thread_list[-1] - float_HB_thread_list[-2]) #Condition that temperature is stable

        #    stringtempHB = str(HB_T[z])  # Turn float into string
         #   bytetempHB = stringtempHB.encode()  # Turn string into byte
          #  combinedstring_Julabo = commandstring_prefix_Julabo + bytetempHB + b" " + commandstring_suffix_Julabo  # Create full command
           # ser.write(combinedstring_Julabo)
        #for v in range(len(Sourcemeterlist)):
         #   if flag == 0:
          #      break
           # else:
            #    stringflowrate = str(Sourcemeterlist[v])
             #   combinedstring_Keithly = commandstring_prefix_Keithly + " " + stringflowrate
              #  Sourcemeter.write(combinedstring_Keithly)  # Command to change flow rate of the pump

               # time.sleep(60)

                #ser.write(b"IN_pv_02\n")  # Reading external temperature
                #HB_thread_data = ser.readline()

                #decoded_HB_thread_data = HB_thread_data.decode("utf-8")  # Decode byte to string
           #     stripped_HB_thread_data = decoded_HB_thread_data.strip()  # Strip the letter from the string
            #    float_HB_thread_data = float(stripped_HB_thread_data)  # Turn stripped data into float

             #   float_HB_thread_list.append(float_HB_thread_data)  # Put most recent measurement in list
              #  condition1 = abs(float_HB_thread_data - HB_T[z])  # Condition that temperature is close to set temperature
               # condition2 = abs(float_HB_thread_list[-1] - float_HB_thread_list[-2])  # Condition that temperature is stable
                #print(float_HB_thread_data)  # Print data to see what temperature is measured

          #      sumnumerator = sum(flow_rate[-21:-11]) #Calculate the sum of the 9 values before
           #     sumdenominator = sum(flow_rate[-11:-1]) #Calculate the sum of the last 9 values
            #    condition3 = abs((sumnumerator/sumdenominator)-1) #Calculate the error
             #   print(condition3)






          #  while (condition1 > 0.3 or condition2 > 0.3 or condition3 > 0.01) and flag == 1:
           #     time.sleep(60)

            #    ser.write(b"IN_pv_02\n")  # Reading external temperature
             #   HB_thread_data = ser.readline()

              #  decoded_HB_thread_data = HB_thread_data.decode("utf-8")  # Decode byte to string
               # stripped_HB_thread_data = decoded_HB_thread_data.strip() #Strip the letter from the string
            #    float_HB_thread_data = float(stripped_HB_thread_data)  # Turn stripped data into float

             #   float_HB_thread_list.append(float_HB_thread_data)  # Put most recent measurement in list
              #  condition1 = abs(float_HB_thread_data - HB_T[z]) #Condition that temperature is close to set temperature
               # condition2 = abs(float_HB_thread_list[-1] - float_HB_thread_list[-2]) #Condition that temperature is stable
               # print(float_HB_thread_data) #Print data to see what temperature is measured

                #sumnumerator = sum(flow_rate[-21:-11])  # Calculate the sum of the 9 values before
                #sumdenominator = sum(flow_rate[-11:-1])  # Calculate the sum of the last 9 values
                #condition3 = abs((sumnumerator / sumdenominator) - 1)  # Calculate the error
                #print(condition3)


    #flag = False #Stop all the threads when the measurementroutine is done
    #ser.write(b"OUT_mode_05 0\r\n")  # Turn off Heat Bath
    #Sourcemeter.write('OUTPUT OFF')  # Turn off the Pump


def get_input():
    global flag
    keystrk=input('Press the enter key to stop recording data. \n')  # \n
    # thread doesn't continue until key is pressed
    print('The recording of data will stop now.', keystrk)
    flag = 0


n = threading.Thread(target=normal)
#m = threading.Thread(target=temp_and_flow_control)
i = threading.Thread(target=get_input)
n.start()
#m.start()
i.start()

# **************************************** Plotting data in (almost) real time *****************************************

x_data, y_data = [], []

temperatures_figure = plt.figure(figsize=(6, 4))
plt.xlabel('Elapsed Time (s)') # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Temperature (C)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
water_in_temp_line, = plt.plot(x_data, y_data, 'b-')  # T water in
water_out_temp_line, = plt.plot(x_data, y_data, 'g-')  # T water out
block_top_temp_line, = plt.plot(x_data, y_data, 'k-')  # T block 1
block_bottom_temp_line, = plt.plot(x_data, y_data, 'r-')  # T block 2
block_left_temp_line, = plt.plot(x_data, y_data, 'm-')  # T block 3
block_right_temp_line, = plt.plot(x_data, y_data, 'y-')  # T block 4

flow_figure = plt.figure(figsize=(6, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Flow_rate (Lpm)')  # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
flow_rate_line, = plt.plot(x_data, y_data, 'b-')

volt_drop_block_figure = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Voltage drop MCM block (mV)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
block_volt_drop_line, = plt.plot(x_data, y_data, 'r-')

diff_press_figure = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Differential_pressure (kPa)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
differential_pressure_line, = plt.plot(x_data, y_data, 'r-')

heating_current_figure = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Heating_Current (A)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
heating_current_line, = plt.plot(x_data, y_data, 'k-')

# figure6 = plt.figure(figsize=(8, 4))
# plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
# plt.ylabel('Differential_pressure_filter (kPa)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
# line10, = plt.plot(x_data, y_data, 'r-')

volt_plates_figure = plt.figure(figsize=(8, 4))
plt.xlabel('Elapsed Time (s)')  # , fontsize=24 Create a label for the x axis and set the font size to 24pt
plt.ylabel('Voltage between plates (V)') # , fontsize=24 Create a label for the y axis and set the font size to 24pt.. $^\circ$C
volt_plates_line, = plt.plot(x_data, y_data, 'r-')


def temperatures(frame):
    water_in_temp_line.set_data(time_water_in, T_water_in)
    water_out_temp_line.set_data(time_water_out, T_water_out)
    block_top_temp_line.set_data(time_block_top, T_block_top)
    block_bottom_temp_line.set_data(time_block_bottom, T_block_bottom)
    block_left_temp_line.set_data(time_block_left, T_block_left)
    block_right_temp_line.set_data(time_block_right,T_block_right)
    temperatures_figure.gca().relim()
    temperatures_figure.gca().autoscale_view()
    return water_in_temp_line, water_out_temp_line, block_top_temp_line, block_bottom_temp_line, block_left_temp_line, block_right_temp_line


def flow(frame):
    flow_rate_line.set_data(time_flow_rate, flow_rate)
    flow_figure.gca().relim()
    flow_figure.gca().autoscale_view()
    return flow_rate_line


def block_volt_drop(frame):
    block_volt_drop_line.set_data(time_volt_drop_block, voltage_drop_block)
    volt_drop_block_figure.gca().relim()
    volt_drop_block_figure.gca().autoscale_view()
    return block_volt_drop_line


def differential_pressure(frame):
    differential_pressure_line.set_data(time_diff_pressure, diff_pressure)
    diff_press_figure.gca().relim()
    diff_press_figure.gca().autoscale_view()
    return differential_pressure_line


def heating_current(frame):
    heating_current_line.set_data(time_current_shunt, current_shunt)
    heating_current_figure.gca().relim()
    heating_current_figure.gca().autoscale_view()
    return heating_current_line


# def update6(frame):
#     line10.set_data(time_diff_pressure_filter, diff_pressure_filter)
#     figure6.gca().relim()
#     figure6.gca().autoscale_view()
#     return line10


def volt_plates(frame):
    volt_plates_line.set_data(time_voltage_drop_plates, voltage_drop_plates)
    volt_plates_figure.gca().relim()
    volt_plates_figure.gca().autoscale_view()
    return volt_plates_line


animation1 = FuncAnimation(temperatures_figure, temperatures, interval=200)
animation2 = FuncAnimation(flow_figure, flow, interval=200)
animation3 = FuncAnimation(volt_drop_block_figure, block_volt_drop, interval=200)
animation4 = FuncAnimation(diff_press_figure, differential_pressure, interval=200)
animation5 = FuncAnimation(heating_current_figure, heating_current, interval=200)
# animation6 = FuncAnimation(figure6, update6, interval=200)
animation7 = FuncAnimation(volt_plates_figure, volt_plates, interval=200)

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
# **************************************** End of script ************************************************************

