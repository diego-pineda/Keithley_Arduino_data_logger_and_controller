Keithley_Arduino_data_logger_and_controller

This repository contains the Python scripts used in the research project presented in Chapter 5 of the TU Delft PhD thesis titled "Advanced Magnetocaloric Regenerators for Heat Pump Applications". The aim of the research project was to determine the flow and heat transfer characteristics of a new geometry used for active magnetocaloric regenerators. The geometry results from an extrusion-based additive manufacturing process whereby layers of parallel fibers are printed with controlled spacing between them, with each successive layer oriented at an alternating angle to create a porous block structure. A description of the contents of this repository is given in what follows.

PT100_Table.txt

This file contains a table with the characteristic curve of a PT100 temperature sensor. A PT100 is a resistive temperature detector (RTD) type of sensor based on platinum and exhibiting a nominal resistance of 100 Ohms at 0 Celcius. The first column of the table gives the temperature in Celcius and the second column gives the electrical resistance in Ohms. This table is used in the Monitornocontrol.py script to convert the resistance values read by the multimeters to temperatures. 

Monitornocontrol.py

This Python script was used to collect data from two Keithley 2000 multimeters, one Lakeshore 331 temperature controller, one Agilent 34410A multimeter. Together, these multimeters collected data on temperature, flow rate, pressure drop, current, and voltage differences necessary for the pressure drop and heat transfer characterization of this geometry. The directories Output_figures and Sensor_data are used to store the outputs of the Monitornocontrol.py script, i.e. the collected sensor data in the form of tables (stored in Sensor_data) and in the form of signal vs time figures (stored in Output_figures).

Pump_control_Keithley2400.py

This Python script was used to generate a series of discrete voltage steps—either increasing or decreasing—using a Keithley 2400 SourceMeter. The voltages served as control signals for a PWM controller, which regulated the speed—and therefore the flow rate—of a gear pump.

Together, the three previous files were used for the collection of the pressure drop, flow rate, and temperature data used for the characterization of the flow and heat transfer of the 3D printed blocks, including the derivation of friction factor and Nusselt number correlations. 

image_processing_script_refactored.py

This Python script was used for processing SEM images of the surfaces of the 3D printed blocks employed in this investigation. The aim of this script is to obtain average fiber diameters, inter-fiber spacings and the standard deviations of the these quantities.

sample_characterization.py 

This Python script provides a Python class called Sample(), which is used to create instances of the different samples used in this research. This class includes methods to compute important geometric parameters such as void fraction and specific surface area of the blocks based on measurable parameters such as the fiber diameter and the inter-fiber distance. Many helper functions are also included in this script.

compute_uncertainties.py

This Python script defines the methods used to compute the uncertainty of geometric parameters (such as void fraction, equivalent particle diameter, specific surface area), measured variables (such as volumetric flow rate, pressure drop, temperature), and computed variables (such as friction factor, and Reynolds number) using the method of propagation of uncertainties.

flow_heat_data_analysis.py

This Python script was used in conjunction with sample_characterization.py and compute_uncertainties.py to process the collected pressure drop and volumetric flow rate data, ultimately deriving a correlation between friction factor and Reynolds number for the specific geometry under study.

The files image_processing_script.py and keithley.py are obsolete and can be omitted.  
