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
from CoolProp.CoolProp import PropsSI
from scipy.optimize import curve_fit
from sympy import symbols, lambdify
from scipy.optimize import fsolve

# Inputs
porosity45 = 0.45  #
porosity50 = 0.50  #
fiberwidth = 400  # micron
porosity40 = 0.40


def water_density_coolprop(temperature):
    """
    Calculates the density of liquid water at atmospheric pressure based on its temperature.
    Uses CoolProp library.

    Args:
        temperature: fluid temperature in K

    Returns:
        Water density in kg/m^3
    """
    return PropsSI('D', 'T', temperature, 'P', 101325, 'INCOMP::Water')


def water_viscosity_coolprop(temperature):
    """
    Calculates the dynamic viscosity of liquid water at atmospheric pressure based on its temperature.
    Uses CoolProp library.

    Args:
        temperature: fluid temperature in K

    Returns:
        Dynamic viscosity in Pa*s (direct method, if available)
    """
    return PropsSI("V", "T", temperature, "P", 101325, "INCOMP::Water")


def area_block(Nlayers, rods_per_layer, rod_diam, rod_length):
    """
    Calculates an approximation of the surface area (also known as wet area) based on geometric and printing parameters
    of the 3D printed block

    Args:
        Nlayers: number of layers of rods that were printed
        rods_per_layer: number of rods that were printed per layer
        rod_diam: approximate diameter of rods in m
        rod_length: approximate length of the rods in m

    Returns:
         Wet area of the 3D printed block in m^2
    """

    return np.pi * rod_diam * rod_length * rods_per_layer * Nlayers - 2 * rods_per_layer**2 * (Nlayers-1) * rod_diam**2


def effective_diameter(vol_block, block_area):
    """
    Calculates the hydraulic or effective diameter of the bed. This would be the diameter of particles of an
    hydraulically equivalent packed bed of (perfectly) spherical particles.

    Args:
        vol_block: volume of solid in the bed in m3
        block_area: wet area of the block in m2

    Returns:
        Effective (also know as hydraulic) diameter of the 3D printed block
    """

    return 6 * vol_block / block_area


def Re(rho, mu, u, Dh, eps):
    """
    This function calculates a modified Reynolds number

    Args:
        rho: density of fluid in kg/m3
        mu: dynamic viscosity in Pa*s
        u: superficial velocity of fluid in m/s. Velocity of fluid in before entering the bed.
        Dh: characteristic length, hydraulic diameter or effective diameter, in m
    Returns:
        Modified Reynolds number
    """

    return (rho * u * Dh) / (mu * (1 - eps))  # (rho * u * Dh) / mu


def ff(dp, rho, u, Dh, L, eps):
    """
    Calculates a modified friction factor of the form proposed by Blake

    Args:
        dp: Pressure drop in kPa
        rho: density of fluid in kg/m3
        u: superficial velocity of fluid in m/s. Velocity of fluid in before entering the bed.
        Dh: hydraulic (effective) diameter in m
        L: length of block measured in the flow direction in m
        eps: bed porosity
    Returns:
        Modified friction factor. Non-dimensional number
    """
    return (1000 * dp * Dh) / (L * rho * u ** 2) * (eps**3/(1-eps)) #  (2 * 1000 * dp * Dh) / (L * rho * u ** 2) * SD


def HP(rho, h, g):
    """
    Calculates hidrostatic pressure

    Args:
        rho: fluid density in kg/m3
        h: height of fluid column in m
        g: gravity constant in m/s2

    Returns:
        Hidrostatic pressure in Pa
    """
    return rho * h * g


def reynolds_friction(data_file, x1, x2, Tw_in_col_index, Tw_out_col_index, flow_col_index, dp_col_index, block_ID):  # Dh, A_cs, eps, L,
    """
    This function calculates the Reynolds numbers and friction factors corresponding
    to the flow rates and pressure data collected.

    Args:
        data_file: path to the file containing the data from the measurements
        x1: starting row from which data will be used for the calculation
        x2: final row until which data will be used for the calculation
        Tw_in_col_index: index of column with the inlet water temperature data
        Tw_out_col_index: index of column with the outlet water temperature data
        flow_col_index: index of column with flow rate data. (Note: first column in data file is index 0)
        dp_col_index: index of column with pressure drop data. (Note: first column in data file is index 0)
        block_ID: instance of the python class sample that contains the geometric description of the 3D printed block

    Returns:
        A tuple containing the following two lists:
        Re_list: list of Reynolds numbers corresponding to the set of flow rates measured
        ff_list: list of the corresponding friction factors corresponding to the set of flow rates measured
    """

    Dh = block_ID.Dp_eq
    A_cs = block_ID.A_cs
    eps = block_ID.porosity
    L = block_ID.L_block

    data_frame = pd.read_csv(data_file, sep='\t')
    data_array = pd.DataFrame(data_frame).to_numpy()
    Re_list = []
    ff_list = []
    for i in range(x1, x2):

        temperature = np.mean([data_array[i, Tw_in_col_index], data_array[i, Tw_out_col_index]]) + 273.15  # [K] Average between Tw_in and Tw_out
        viscosity = water_viscosity_coolprop(temperature)
        density = water_density_coolprop(temperature)

        Vout_press_sensor = 5.07 * (0.009 * data_array[i, dp_col_index] + 0.04)
        DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
        pressure = DP_corrected - HP(density, 0.11, 9.8) / 1000  # [kPa] Pressure drop

        flowrate = data_array[i, flow_col_index] / 60000  # [m3/s] Vol. flow rate. Flow in data file must be in Lpm.
        velocity = flowrate / A_cs  # [m/s] Superficial velocity

        Re_list.append(Re(density, viscosity, velocity, Dh, eps))
        ff_list.append(ff(pressure, density, velocity, Dh, L, eps))

    return Re_list, ff_list


def fit_function(expr, xdata, ydata):
    """
    Fit a general mathematical expression to data.

    Parameters:
        expr (sympy expression): The mathematical expression to fit.
        xdata (array-like): Data for the independent variable.
        ydata (array-like): Data for the dependent variable.

    Returns:
        dict: Fitted parameters and their covariance matrix.
    """
    # Define the variables
    x = symbols('x')
    params = sorted(expr.free_symbols - {x}, key=lambda s: s.name)  # Get parameters in a consistent order
    print(params)
    # Convert sympy expression to a callable function
    func = lambdify((x, *params), expr, modules='numpy')

    # Define the fitting wrapper function for curve_fit
    def wrapper(x_vals, *param_vals):
        return func(x_vals, *param_vals)

    # Provide an initial guess for the parameters (required by curve_fit)
    initial_guess = [1.0] * len(params)

    # Perform the curve fitting
    popt, pcov = curve_fit(wrapper, xdata, ydata, p0=initial_guess, maxfev=10000)

    # Return results as a dictionary
    return {
        'params': dict(zip([p.name for p in params], popt)),
        'covariance': pcov
    }


def evaluate_function(expr, x_values, *param_values):
    """
    Evaluate a function defined by a symbolic expression
    :param expr: symbolic expression
    :param x_values: x values to evaluate
    :return: values of the evaluated function
    """
    # Define the variables
    x = symbols('x')
    params = sorted(expr.free_symbols - {x}, key=lambda s: s.name)  # Get parameters in a consistent order

    # Convert sympy expression to a callable function
    func = lambdify((x, *params), expr, modules='numpy')

    # Define the fitting wrapper function for curve_fit

    return func(x_values, *param_values)


def ff_vs_Re(x, a, b):
    return a / x + b


def temperature_profile(y, L, T_f, g_dot, h, beta, k, q1, q2):
    """
    Calculate the temperature profile T(y) in a system with heat flux boundary conditions.

    Parameters:
    y (float or numpy array): Position along the domain (0 to L) [m]
    q1 (float): Heat flux at y=0 [W/m^2]
    q2 (float): Heat flux at y=L [W/m^2]
    k (float): Thermal conductivity of the material [W/m·K]
    L (float): Length of the domain [m]
    T_f (float): Fluid temperature [K]
    g_dot (float): Uniform heat generation rate per unit volume [W/m^3]
    h (float): Convective heat transfer coefficient [W/m^2·K]
    beta (float): Surface area per unit volume of the matrix [1/m]

    Returns:
    float or numpy array: Temperature T(y) at position(s) y [K]
    """
    # Calculate m from h, beta, and k
    m = np.sqrt(h * beta / k)

    # Calculate constants C1 and C2
    C1 = (q1 * np.exp(-m * L) - q2) / (k * m * (np.exp(m * L) - np.exp(-m * L)))
    C2 = (q1 * np.exp(m * L) - q2) / (k * m * (np.exp(m * L) - np.exp(-m * L)))

    # General temperature profile
    T_y = C1 * np.exp(m * y) + C2 * np.exp(-m * y) + T_f + g_dot / (h * beta)

    return T_y


def solve_h(h_initial, x_known, T_known, L, T_f, g_dot, beta, k, q1, q2):
    """
    Solves for h (heat transfer coefficient) using the known temperature
    at an intermediate location and Neumann boundary conditions.

    Parameters:
    h_initial (float): Initial guess for h [W/m^2·K].
    x_known (float): Position where temperature is known [m].
    T_known (float): Known temperature at x_known [K].
    L (float): Length of the domain [m].
    T_f (float): Fluid temperature [K].
    g_dot (float): Heat generation rate per unit volume [W/m^3].
    beta (float): Surface area per unit volume [1/m].
    k (float): Thermal conductivity [W/m·K].
    q1 (float): Heat flux at y=0 [W/m^2].
    q2 (float): Heat flux at y=L [W/m^2].

    Returns:
    float: Heat transfer coefficient h [W/m^2·K].
    """
    def equation(h):
        # Calculate the temperature profile using the updated function
        T_calc = temperature_profile(x_known, L, T_f, g_dot, h, beta, k, q1, q2)
        return T_calc - T_known

    # Solve for h using the given equation
    h_solution = fsolve(equation, h_initial)
    return h_solution[0]


def calculate_heat_transfer_coefficients_with_sample(file_path, sample, k_s, k_f, alpha_k, R_total):
    """
    Calculate heat transfer coefficients using geometric parameters from a Sample instance,
    dynamically computing k_eff based on porosity, and heat generation rates from the "g_dot" column.

    Parameters:
    - file_path (str): Path to the file containing the data (CSV or Excel).
    - sample (Sample): Instance of the Sample class containing geometric parameters.
    - k_s (float): Thermal conductivity of the solid [W/(m·K)].
    - k_f (float): Thermal conductivity of the fluid [W/(m·K)].
    - alpha_k (float): Weighting factor for combining upper and lower bounds of k_eff.
    - R_total (float): total estimated thermal contact resistance between sample and copper plates

    Returns:
    - DataFrame: Data with additional heat transfer coefficients column.
    - list: List of heat transfer coefficients (h_values).
    - list: List of flow rates in Lpm corresponding to h_values.
    """
    # Load the data into a DataFrame
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    # Calculate k_eff based on porosity and user-defined parameters
    k_upper = sample.porosity * k_f + (1 - sample.porosity) * k_s
    k_lower = 1 / (sample.porosity / k_f + (1 - sample.porosity) / k_s)
    k_eff = alpha_k * k_upper + (1 - alpha_k) * k_lower

    h_values = []
    flow_rates = []  # To store flow rate values in Lpm
    densities, viscosities, conductivities, prandtl_numbers = [], [], [], []
    superficial_velocities, reynolds_numbers, nusselt_numbers = [], [], []

    for _, row in data.iterrows():
        T1_measured = row["T_block_bottom"] + 273  # Convert to Kelvin
        T2_measured = row["T_block_top"] + 273     # Convert to Kelvin
        Q_contact_resistances = row["Qcontacts"]  # Heat generated at block contacts
        T_f = (row["T_water_out"] + row["T_water_in"]) / 2 + 273  # Average fluid temperature (Kelvin)
        g_dot = row["g_dot"]  # Retrieve g_dot for the current row
        flow_rate = row["Vflow"]  # Flow rate in Lpm
        flow_rates.append(flow_rate)

        alpha_initial = np.array([0.5])  # Initial guess for alpha
        h_initial = 200  # Initial guess for heat transfer coefficient
        x_known = 0  # Known location (e.g., y=0)

        # Define the equation to solve for alpha
        def alpha_equation(alpha):
            q1 = (Q_contact_resistances / (sample.L_block * sample.W_block)) * alpha
            q2 = -(Q_contact_resistances / (sample.L_block * sample.W_block)) * (1 - alpha)
            h_solution = solve_h(h_initial, x_known, T1_measured, sample.H_block, T_f, g_dot, sample.beta, k_eff, q1, q2)
            T2_calculated = temperature_profile(sample.H_block, sample.H_block, T_f, g_dot, h_solution, sample.beta, k_eff, q1, q2)
            return T2_calculated - T2_measured

        # Solve for alpha
        alpha_solution = fsolve(alpha_equation, alpha_initial, xtol=1e-8)[0]

        # Calculate final q1 and q2 using the solved alpha
        q1_final = Q_contact_resistances / (sample.L_block * sample.W_block) * alpha_solution
        q2_final = -Q_contact_resistances / (sample.L_block * sample.W_block) * (1 - alpha_solution)

        

        # Solve for h
        h_solution = solve_h(h_initial, x_known, T1_measured, sample.H_block, T_f, g_dot, sample.beta, k_eff, q1_final, q2_final)

        # Calculate fluid properties
        density = PropsSI('D', 'T', T_f, 'P', 101325, 'Water')
        viscosity = PropsSI('V', 'T', T_f, 'P', 101325, 'Water')
        conductivity = PropsSI('L', 'T', T_f, 'P', 101325, 'Water')
        cp = PropsSI('C', 'T', T_f, 'P', 101325, 'Water')
        prandtl = (cp * viscosity) / conductivity

        # Calculate velocities, Reynolds, Nusselt
        superficial_velocity = flow_rate / 60000 / sample.A_cs
        reynolds_number = (density * superficial_velocity * sample.Dp_eq) / (viscosity * (1 - sample.porosity))
        nusselt_number = h_solution * sample.Dp_eq / conductivity

        densities.append(density)
        viscosities.append(viscosity)
        conductivities.append(conductivity)
        prandtl_numbers.append(prandtl)
        superficial_velocities.append(superficial_velocity)
        reynolds_numbers.append(reynolds_number)
        h_values.append(h_solution)
        nusselt_numbers.append(nusselt_number)

    # Assign lists back to DataFrame
    data["Density"] = densities
    data["Viscosity"] = viscosities
    data["Conductivity"] = conductivities
    data["Prandtl"] = prandtl_numbers
    data["Superficial_Velocity"] = superficial_velocities
    data["Reynolds_number"] = reynolds_numbers
    data["Heat_Transfer_Coefficient"] = h_values
    data["Nusselt_number"] = nusselt_numbers

    return h_values, flow_rates, reynolds_numbers, nusselt_numbers, data




# ------------------------- Samples ---------------
class Sample:
    def __init__(self, V_solid_archimedes, N_layers, N_fibers, D_fibers, L_fibers, porosity, L_block, W_block, H_block, A_inters):
        """
        Class sample describe the geometry of the 3D printed block

        :param V_solid_archimedes: [m3] Volume of the solid measured using the Archimedes method
        :param N_layers: Number of layers of fibers that were printed
        :param N_fibers: Number of rods that were printed per layer
        :param D_fibers: [m] Diameter of fibers
        :param L_fibers: [m] Length of fibers. Should be in principle the same as W_block and H_block
        :param porosity: Porosity of 3D printed block determined using the Archimedes method
        :param L_block: [m] Length of 3D printed block measured in flow direction
        :param W_block: [m] Width of 3D printed block measured perpendicular to flow direction
        :param H_block: [m] Heigth of 3D printed block measured perpendicular to flow direction
        """

        self.V_solid_archimedes = V_solid_archimedes
        self.N_layers = N_layers
        self.N_fibers = N_fibers
        self.D_fibers = D_fibers
        self.L_fibers = L_fibers
        self.porosity = porosity
        self.L_block  = L_block
        self.W_block  = W_block
        self.H_block  = H_block
        self.A_cs     = (W_block-0.001) * (H_block-0.001)  # [m2] Area of cross section of flow channel before block
        self.A_solid  = (np.pi * D_fibers * L_fibers * N_fibers * N_layers - 2 * N_fibers**2 * (N_layers-1) * A_inters)  # D_fibers**2
        self.Dp_eq       =  6 * V_solid_archimedes / self.A_solid  # W_block / (N_fibers-1) - D_fibers  # Equivalent particle diameter
        self.beta     = self.A_solid / (self.W_block * self.H_block * self.L_block)
        # self.V_solid_calc = np.pi * D_fibers**2 * L_fibers * N_fibers * N_layers / 4 - N_fibers**2 * (1-N_layers) * V_inters

# TODO define a better way to estimate wet area and V_solid_calc

DP_90deg_600um_40p_a = Sample(V_solid_archimedes=1.091E-6, N_layers=18, N_fibers=21, D_fibers=570e-6, L_fibers=0.016, porosity=0.29, L_block=0.0062, W_block=0.016, H_block=0.016, A_inters=371283E-12)
DP_90deg_600um_45p_a = Sample(V_solid_archimedes=1.058E-6, N_layers=18, N_fibers=19, D_fibers=570e-6, L_fibers=0.016, porosity=0.31, L_block=0.0062, W_block=0.016, H_block=0.016, A_inters=371283E-12)
DP_90deg_600um_50p_a = Sample(V_solid_archimedes=1.000E-6, N_layers=18, N_fibers=17, D_fibers=570e-6, L_fibers=0.016, porosity=0.35, L_block=0.0062, W_block=0.016, H_block=0.016, A_inters=371283E-12)

# DP_90deg_400um_40p_a = Sample(V_solid_archimedes=1.027E-6, N_layers=24, N_fibers=33, D_fibers=380e-6, L_fibers=0.0162, porosity=0.331, L_block=0.0062, W_block=0.016, H_block=0.016, A_inters=135682E-12)
DP_90deg_400um_45p_a = Sample(V_solid_archimedes=1.027E-6, N_layers=24, N_fibers=33, D_fibers=380e-6, L_fibers=0.0162, porosity=0.331, L_block=0.0062, W_block=0.016, H_block=0.016, A_inters=135682E-12)
DP_90deg_400um_50p_a = Sample(V_solid_archimedes=0.958E-6, N_layers=24, N_fibers=29, D_fibers=380e-6, L_fibers=0.0162, porosity=0.376, L_block=0.0062, W_block=0.016, H_block=0.016, A_inters=135682E-12)

print(f'Dh_600um_40p_a = {DP_90deg_600um_40p_a.Dp_eq}')
print(f'Dh_600um_45p_a = {DP_90deg_600um_45p_a.Dp_eq}')
print(f'Dh_600um_50p_a = {DP_90deg_600um_50p_a.Dp_eq}')
print(f'Dh_400um_45p_a = {DP_90deg_400um_45p_a.Dp_eq}')
print(f'Dh_400um_50p_a = {DP_90deg_400um_50p_a.Dp_eq}')

print(f'As_600um_40p_a = {DP_90deg_600um_40p_a.A_solid}')
print(f'As_600um_45p_a = {DP_90deg_600um_45p_a.A_solid}')
print(f'As_600um_50p_a = {DP_90deg_600um_50p_a.A_solid}')
print(f'As_400um_45p_a = {DP_90deg_400um_45p_a.A_solid}')
print(f'As_400um_50p_a = {DP_90deg_400um_50p_a.A_solid}')

print(f'Beta_600um_40p_a = {DP_90deg_600um_40p_a.beta}')
print(f'Beta_600um_45p_a = {DP_90deg_600um_45p_a.beta}')
print(f'Beta_600um_50p_a = {DP_90deg_600um_50p_a.beta}')
print(f'Beta_400um_45p_a = {DP_90deg_400um_45p_a.beta}')
print(f'Beta_400um_50p_a = {DP_90deg_400um_50p_a.beta}')

# samples = {"90deg_600um_40p_a": {"V_solid_archimedes": 1.091E-6,
#                                  "N_layers": 18,
#                                  "N_fibers": 21,
#                                  "D_fiber": 600e-6,
#                                  "L_fiber": 0.016},
#            "90deg_600um_45p_a": {"V_solid_archimedes": 1.058E-6,
#                                  "N_layers": 18,
#                                  "N_fibers": 19,
#                                  "D_fiber": 600e-6,
#                                  "L_fiber": 0.016},
#            "90deg_600um_50p_a": {"V_solid_archimedes": 1.000E-6,
#                                  "N_layers": 18,
#                                  "N_fibers": 17,
#                                  "D_fiber": 600e-6,
#                                  "L_fiber": 0.016}}
#
# # Volume of solid part of 3D printed blocks determined using Archimedes method. [m3]
#
# V_solid_archimedes_600um_40p_a = 1.091E-6  # [m3] 600um 40% theoretical porosity sample
# V_solid_archimedes_600um_45p_a = 1.058E-6  # [m3] 600um 45% theoretical porosity sample
# V_solid_archimedes_600um_50p_a = 1.000E-6  # [m3] 600um 50% theoretical porosity sample
#
# # Number of layers per block. [-]
#
# # Number of rods per layer. [-]
#
#
# # Diameter of fibers. [m]
#
# # Approximate length of rods. [m]
#
# # Surface area of 3D printed blocks
# A_solid_600um_40p_a = area_block(18, 21, 0.000600, 0.016)
# A_solid_600um_45p_a = area_block(18, 19, 0.000600, 0.016)
# A_solid_600um_50p_a = area_block(18, 17, 0.000600, 0.016)
#
# # Hydraulic diameter
#
# Dh_600um_40p_a = effective_diameter(V_solid_archimedes_600um_40p_a, A_solid_600um_40p_a)
# Dh_600um_45p_a = effective_diameter(V_solid_archimedes_600um_45p_a, A_solid_600um_45p_a)
# Dh_600um_50p_a = effective_diameter(V_solid_archimedes_600um_50p_a, A_solid_600um_50p_a)

# ------------------------- 600um 40 % porosity sample ---------------

Re_600um_40perc_10C, ff_600um_40perc_10C = reynolds_friction("./Sensor_data/09Oct2024_90deg_600um_40p_a_press_drop_10C.csv",
                                                             1, 658, 5, 7, 11, 13, DP_90deg_600um_40p_a)  # .Dh, 0.015 * 0.015, 0.29, 0.0062 area was 0.016 ** 2 * 0.29 and then 0.000180**2 * 21**2  Dh was 0.180
Re_600um_40perc_22C, ff_600um_40perc_22C = reynolds_friction("./Sensor_data/09Oct2024_90deg_600um_40p_a_press_drop_22C.csv",
                                                             1, 794, 5, 7, 11, 13, DP_90deg_600um_40p_a)  # .Dh, 0.015 * 0.015, 0.29, 0.0062
Re_600um_40perc_50C, ff_600um_40perc_50C = reynolds_friction("./Sensor_data/09Oct2024_90deg_600um_40p_a_press_drop_50C.csv",
                                                             1, 773, 5, 7, 11, 13, DP_90deg_600um_40p_a)

h_600um_40perc, Vflow_600um_40perc, Re_htc_600um_40perc, Nu_600um_40perc, data_600um_40perc = calculate_heat_transfer_coefficients_with_sample("data_htc_600um_40p.xlsx", DP_90deg_600um_40p_a, 6, 0.598, 0.5)



# ------------------------- 600um 45 % porosity sample ---------------
Re_600um_45perc_7C, ff_600um_45perc_7C = reynolds_friction("./Sensor_data/17Sep2024_90deg_600um_45p_a_press_drop_7C.csv",
                                                           1, 741, 5, 7, 11, 13, DP_90deg_600um_45p_a)  # Dh was 260  area was 0.000260**2 * (19*18)
Re_600um_45perc_21C, ff_600um_45perc_21C = reynolds_friction("./Sensor_data/17Sep2024_90deg_600um_45p_a_press_drop_21C.csv",
                                                             1, 966, 5, 7, 11, 13, DP_90deg_600um_45p_a)  # Dh_600um_45p_a, 0.015 * 0.015, 0.31, 0.0062
Re_600um_45perc_50C, ff_600um_45perc_50C = reynolds_friction("./Sensor_data/13Sep2024_90deg_600um_45p_a_press_drop_50C_a.csv",
                                                             1, 1185, 5, 7, 11, 13, DP_90deg_600um_45p_a)

# ------------------------- 600um 50 % porosity sample ---------------
Re_600um_50perc_7C, ff_600um_50perc_7C = reynolds_friction("./Sensor_data/29Aug2024_90deg_600um_50p_a_press_drop_7C.csv",
                                                           1, 800, 5, 7, 9, 11, DP_90deg_600um_50p_a)  # Dh was 340 area was  0.000340**2 * (17*16)
Re_600um_50perc_25C, ff_600um_50perc_25C = reynolds_friction("./Sensor_data/29Aug2024_90deg_600um_50p_a_press_drop_25C.csv",
                                                             1, 990, 5, 7, 9, 11, DP_90deg_600um_50p_a)  # Dh_600um_50p_a, 0.015 * 0.015, 0.35, 0.0062
Re_600um_50perc_50C, ff_600um_50perc_50C = reynolds_friction("./Sensor_data/30Aug2024_90deg_600um_50p_press_drop_50C_2.csv",
                                                             1, 917, 5, 7, 11, 13, DP_90deg_600um_50p_a)  # 19Aug2024_90deg_600um_50p_a_press_drop_50C.csv

h_600um_50perc, Vflow_600um_50perc, Re_htc_600um_50perc, Nu_600um_50perc, data_600um_50perc = calculate_heat_transfer_coefficients_with_sample("600um_50p_averages_output.xlsx", DP_90deg_600um_50p_a, 6, 0.598, 0.5)
#"data_htc_600um_50p_V2.xlsx"
# ------------------------- 400um 40 % porosity sample ---------------

# ------------------------- 400um 45 % porosity sample ---------------
Re_400um_45perc_low, ff_400um_45perc_low = reynolds_friction("./Sensor_data/26Jul2024_400um_45perc_pressure_6C.csv",
                                                           1, 580, 5, 6, 8, 10, DP_90deg_400um_45p_a)  # Dh was 260  area was 0.000260**2 * (19*18)
Re_400um_45perc_mid, ff_400um_45perc_mid = reynolds_friction("./Sensor_data/19Jul2024_45por400_flow25C.csv",
                                                             2, 1161, 5, 6, 8, 10, DP_90deg_400um_45p_a)  # Dh_600um_45p_a, 0.015 * 0.015, 0.31, 0.0062
Re_400um_45perc_hi, ff_400um_45perc_hi = reynolds_friction("./Sensor_data/26Jul2024_400um_45per_pressure_45C.csv",
                                                             2, 978, 5, 6, 8, 10, DP_90deg_400um_45p_a)

h_400um_45perc, Vflow_400um_45perc, Re_htc_400um_45perc, Nu_400um_45perc, data_400um_45perc = calculate_heat_transfer_coefficients_with_sample("data_htc_400um_45p.xlsx", DP_90deg_400um_45p_a, 6, 0.598, 0.5)

# ------------------------- 400um 50 % porosity sample ---------------
Re_400um_50perc_mid, ff_400um_50perc_mid = reynolds_friction("./Sensor_data/19Jun2024_50p_Ambientgapclosing.csv",
                                                             1, 205, 9, 11, 13, 19, DP_90deg_400um_50p_a)  # Dh_600um_45p_a, 0.015 * 0.015, 0.31, 0.0062
Re_400um_50perc_hi, ff_400um_50perc_hi = reynolds_friction("./Sensor_data/19Jun2024_50p_Hottestvertical.csv",
                                                           1, 445, 9, 11, 13, 19, DP_90deg_400um_50p_a)
# ----------------------------------------------------------------------------------------------------------------------
# Fitting data to a function of the form ff = a / Re + b, using modified friction factor and modified Re number

single_Re_data_list = Re_600um_40perc_10C + Re_600um_40perc_22C + Re_600um_40perc_50C + \
                      Re_600um_45perc_7C + Re_600um_45perc_21C + Re_600um_45perc_50C + \
                      Re_600um_50perc_7C + Re_600um_50perc_25C + Re_600um_50perc_50C + \
                      Re_400um_45perc_low + Re_400um_45perc_mid + Re_400um_45perc_hi + \
                      Re_400um_50perc_mid + Re_400um_50perc_hi

single_ff_data_list = ff_600um_40perc_10C + ff_600um_40perc_22C + ff_600um_40perc_50C + \
                      ff_600um_45perc_7C + ff_600um_45perc_21C + ff_600um_45perc_50C + \
                      ff_600um_50perc_7C + ff_600um_50perc_25C + ff_600um_50perc_50C + \
                      ff_400um_45perc_low + ff_400um_45perc_mid + ff_400um_45perc_hi + \
                      ff_400um_50perc_mid + ff_400um_50perc_hi

# Define your expression: y = a / x + b
x, a, b, c = symbols('x a b c')
expr = a / x + b / x ** c
expr2 = a / x + b
fit_result = fit_function(expr, single_Re_data_list, single_ff_data_list)
print(list(fit_result['params'].values()))

popt, pcov = curve_fit(ff_vs_Re, single_Re_data_list, single_ff_data_list)
a_fit, b_fit = popt

print(f"ff = {a_fit} / Re + {b_fit}")

# ----------------------------------------------------------------------------------------------------------------------
# Plotting results
plt.figure()
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(expr, np.linspace(1, 1000, 1000), *list(fit_result['params'].values())), '-k', label='Fit by KTA')
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(expr, np.linspace(1, 1000, 1000), *[160, 3, 0.1]), '-r', label='KTA (1981)')
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(expr2, np.linspace(1, 1000, 1000), *[150, 1.75]), '-b', label='Ergun (1952)')
# plt.plot(np.linspace(1, 1000, 1000), ff_vs_Re(np.linspace(1, 1000, 1000), a_fit, b_fit), '-k', label='Fit')
# plt.plot(np.linspace(1, 1000, 1000), ff_vs_Re(np.linspace(1, 1000, 1000), 150, 1.75), '-r', label='Ergun')

plt.plot(Re_600um_40perc_10C, ff_600um_40perc_10C, '.', color='c', label='600um_40%p 10C')
plt.plot(Re_600um_40perc_22C, ff_600um_40perc_22C, '.', color='m', label='600um_40%p 22C')
plt.plot(Re_600um_40perc_50C, ff_600um_40perc_50C, '.', color='brown', label='600um_40%p 50C')

plt.plot(Re_600um_45perc_7C, ff_600um_45perc_7C, '.', color='Olive', label='600um_45%p 7C')
plt.plot(Re_600um_45perc_21C, ff_600um_45perc_21C, '.', color='Purple', label='600um_45%p 21C')
plt.plot(Re_600um_45perc_50C, ff_600um_45perc_50C, '.', color='Violet', label='600um_45%p 50C')

plt.plot(Re_600um_50perc_7C, ff_600um_50perc_7C, '.', color='DarkGreen', label='600um_50%p 7C')
plt.plot(Re_600um_50perc_25C, ff_600um_50perc_25C, '.', color='blue', label='600um_50%p 25C')
plt.plot(Re_600um_50perc_50C, ff_600um_50perc_50C, '.', color='red', label='600um_50%p 50C')

plt.plot(Re_400um_45perc_low, ff_400um_45perc_low, '.', color='Gray', label='400um_45%p low temp')
plt.plot(Re_400um_45perc_mid, ff_400um_45perc_mid, '.', color='Maroon', label='400um_45%p mid temp')
plt.plot(Re_400um_45perc_hi, ff_400um_45perc_hi, '.', color='Navy', label='400um_45%p hi temp')

plt.plot(Re_400um_50perc_mid, ff_400um_50perc_mid, '.', color='Pink', label='400um_50%p mid temp')
plt.plot(Re_400um_50perc_hi, ff_400um_50perc_hi, '.', color='Teal', label='400um_50%p hi temp')

plt.xlabel("Modified Reynolds number [-]", fontsize=10)
plt.ylabel("Modified friction factor [-]", fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(labelsize=10)
plt.legend(fontsize=10)
plt.show()


plt.figure()
plt.scatter(Re_htc_600um_50perc, Nu_600um_50perc, label="600um 35 % porosity")
plt.scatter(Re_htc_600um_40perc, Nu_600um_40perc, label="600 um 29 % porosity")
plt.scatter(Re_htc_400um_45perc, Nu_400um_45perc, label="400 um 33 % porosity")
plt.legend()
plt.xlabel("Modified Reynolds number [-]", fontsize=10)
plt.ylabel("Nusselt number [-]", fontsize=10)

plt.figure()
plt.scatter(Vflow_600um_50perc, h_600um_50perc, label="600um 35 % porosity")
plt.scatter(Vflow_600um_40perc, h_600um_40perc, label="600um 29 % porosity")
plt.scatter(Vflow_400um_45perc, h_400um_45perc, label="400um 33 % porosity")
plt.legend()
plt.xlabel("Flow rate [Lpm]", fontsize=10)
plt.ylabel(r"Heat transfer coefficient [$\rm{W}/(\rm{m}^2 \rm{K})$]", fontsize=10)
plt.show()




# ------------------------------------------------


fulldata = pd.read_csv("18Jun2024_45p_Reynoldsfrictiontest45_1.csv", sep='\t', )  #
fulldata_array = pd.DataFrame(fulldata).to_numpy()

fulldata45por_6C = pd.read_csv("26Jul2024_400um_45perc_pressure_6C.csv",
                               sep='\t', )  # 18Jun2024_45p_Reynoldsfrictiontest45_1
fulldata_array_45por6C = pd.DataFrame(fulldata45por_6C).to_numpy()

fulldata45por_25C = pd.read_csv("19Jul2024_45por400_flow25C.csv", sep='\t', )  # 18Jun2024_45p_Reynoldsfrictiontest45_1
fulldata_array_45por25C = pd.DataFrame(fulldata45por_25C).to_numpy()

fulldata45por_45C = pd.read_csv("26Jul2024_400um_45per_pressure_45C.csv",
                                sep='\t', )  # 18Jun2024_45p_Reynoldsfrictiontest45_1
fulldata_array_45por45C = pd.DataFrame(fulldata45por_45C).to_numpy()

fulldata2 = pd.read_csv("18Jun2024_45p_Ambienttest.csv", sep='\t', )
fulldata2_array = pd.DataFrame(fulldata2).to_numpy()

fulldata50 = pd.read_csv("18Jun2024_50p_Coldtest.csv", sep='\t', )
fulldata50_array = pd.DataFrame(fulldata50).to_numpy()

fulldata502 = pd.read_csv("19Jun2024_50p_Hottest.csv", sep='\t', )
fulldata502_array = pd.DataFrame(fulldata502).to_numpy()

fulldata503 = pd.read_csv("19Jun2024_50p_Hottestvertical.csv", sep='\t', )
fulldata503_array = pd.DataFrame(fulldata503).to_numpy()

fulldata504 = pd.read_csv("19Jun2024_50p_Ambientgapclosing.csv", sep='\t', )
fulldata504_array = pd.DataFrame(fulldata504).to_numpy()

fulldata40 = pd.read_csv("19Jun2024_40p_VerticalRefsweep.csv", sep='\t', )
fulldata40_array = pd.DataFrame(fulldata40).to_numpy()

fulldata402 = pd.read_csv("19Jun2024_40p_Verticalhottest.csv", sep='\t', )
fulldata402_array = pd.DataFrame(fulldata402).to_numpy()

fulldata403 = pd.read_csv("19Jun2024_40p_Verticalmediumtemp.csv", sep='\t', )
fulldata403_array = pd.DataFrame(fulldata403).to_numpy()

fulldata_600um_50p_7C = pd.read_csv("29Aug2024_90deg_600um_50p_a_press_drop_7C.csv", sep='\t', )
fulldata_600um_50p_7C_array = pd.DataFrame(fulldata_600um_50p_7C).to_numpy()

fulldata_600um_50p_25C = pd.read_csv("29Aug2024_90deg_600um_50p_a_press_drop_25C.csv", sep='\t', )
fulldata_600um_50p_25C_array = pd.DataFrame(fulldata_600um_50p_25C).to_numpy()

fulldata_600um_50p_50C = pd.read_csv("19Aug2024_90deg_600um_50p_a_press_drop_50C.csv", sep='\t', )
fulldata_600um_50p_50C_array = pd.DataFrame(fulldata_600um_50p_50C).to_numpy()

fulldata_600um_50p_50C_2 = pd.read_csv("30Aug2024_90deg_600um_50p_press_drop_50C_2.csv", sep='\t', )
fulldata_600um_50p_50C_array_2 = pd.DataFrame(fulldata_600um_50p_50C_2).to_numpy()



actual_fiberwidth = fiberwidth * 1e-6
Dh45 = 140e-6  # (porosity45*actual_fiberwidth)/(1-porosity45)
Acs45 = 0.015 * 0.015  # porosity45*0.016*0.0164
Dh50 = 169e-6  # (porosity50*actual_fiberwidth)/(1-porosity50)  #
Acs50 = 0.015 * 0.015  # porosity50*0.015*0.0154  #
Dh40 = 112e-6  # (porosity40*actual_fiberwidth)/(1-porosity40)  #
Acs40 = 0.015 * 0.015  # porosity40*0.016*0.0164  #

Dh_600um_50p = 340e-6
Acs_600um_50p = 0.015 * 0.015

L = 0.0062  # m
gravity = 9.81  # m/s^2
height = 0.110  # m

Reynolds_list = []
Reynolds45_list = []
Reynolds50_list = []
Reynolds40_list = []
Re_600um_50p_7C_list = []
Re_600um_50p_25C_list = []
Re_600um_50p_50C_list = []
Re_600um_50p_50C_list_2 = []

f_list = []
f45_list = []
f50_list = []
f40_list = []
ff_600um_50p_7C_list = []
ff_600um_50p_25C_list = []
ff_600um_50p_50C_list = []
ff_600um_50p_50C_list_2 = []

for i in range(580):  # 45 % porosity. Test at 7 C. Corresponds to Sheet2 in Origin
    # Note: this data could be unreliable since trapped air could have been messing with the measurements.
    flowrate = fulldata_array[i, 13] / 60000
    pressure = fulldata_array[i, 19]
    temperature = np.mean([fulldata_array[i, 9], fulldata_array[i, 11]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs45
    Reynolds_list.append(Re(density, viscosity, flowrate, Dh45, Acs45))
    f_list.append(ff(pressure, density, velocity, Dh45))

for i in range(1, 1164):  # 45 % porosity. Test at 6 C.
    # Note: vertical arrangement.
    flowrate = fulldata_array_45por6C[i, 8] / 60000
    temperature = np.mean([fulldata_array_45por6C[i, 5], fulldata_array_45por6C[i, 6]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    Vout_press_sensor = 5.07 * (0.009 * fulldata_array_45por6C[i, 10] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata_array_45por6C[i, 10] - HP(density, height, gravity) / 1000
    velocity = flowrate / Acs45
    Reynolds45_list.append(Re(density, viscosity, flowrate, Dh45, Acs45))
    f45_list.append(ff(pressure, density, velocity, Dh45))

for i in range(1, 1161):  # 45 % porosity. Test at 25 C.
    # Note: vertical arrangement.
    flowrate = fulldata_array_45por25C[i, 8] / 60000
    temperature = np.mean([fulldata_array_45por25C[i, 5], fulldata_array_45por25C[i, 6]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    Vout_press_sensor = 5.07 * (0.009 * fulldata_array_45por25C[i, 10] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata_array_45por25C[i, 10] - HP(density, height, gravity) / 1000
    velocity = flowrate / Acs45
    Reynolds45_list.append(Re(density, viscosity, flowrate, Dh45, Acs45))
    f45_list.append(ff(pressure, density, velocity, Dh45))
#
for i in range(1, 978):  # 45 % porosity. Test at 45 C.
    # Note: vertical arrangement.
    flowrate = fulldata_array_45por45C[i, 8] / 60000
    temperature = np.mean([fulldata_array_45por45C[i, 5], fulldata_array_45por45C[i, 6]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    Vout_press_sensor = 5.07 * (0.009 * fulldata_array_45por45C[i, 10] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata_array_45por25C[i, 10] - HP(density, height, gravity) / 1000
    velocity = flowrate / Acs45
    Reynolds45_list.append(Re(density, viscosity, flowrate, Dh45, Acs45))
    f45_list.append(ff(pressure, density, velocity, Dh45))

for i in range(1, 360):  # 45 % porosity. Test at 28 C. Corresponds to Sheet1 in Origin
    # Note: this data could be unreliable since trapped air could have been messing with the measurements.
    flowrate = fulldata2_array[i, 13] / 60000
    pressure = fulldata2_array[i, 19]
    temperature = np.mean([fulldata2_array[i, 9], fulldata2_array[i, 11]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs45
    Reynolds_list.append(Re(density, viscosity, flowrate, Dh45, Acs45))
    f_list.append(ff(pressure, density, velocity, Dh45))

for i in range(990, 1500):  # 45 % porosity. Test at 50 C. Corresponds to Sheet2 in Origin
    # Note: this data could be unreliable since trapped air could have been messing with the measurements.
    flowrate = fulldata_array[i, 13] / 60000
    pressure = fulldata_array[i, 19]
    temperature = np.mean([fulldata_array[i, 9], fulldata_array[i, 11]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs45
    Reynolds_list.append(Re(density, viscosity, flowrate, Dh45, Acs45))
    f_list.append(ff(pressure, density, velocity, Dh45))

# for i in range(420):  # 50 % porosity. Test at 7 C. Corresponds to Sheet1 in Origin.
#     # Note: this data could be unreliable since trapped air could have been messing with the measurements.
#     flowrate = fulldata50_array[i, 13] / 60000
#     pressure = fulldata50_array[i, 19]
#     temperature = np.mean([fulldata50_array[i, 9] , fulldata50_array[i, 11]]) + 273.15
#     viscosity = water_viscosity_coolprop(temperature)
#     density = water_density_coolprop(temperature)
#     velocity = flowrate / Acs50
#     Vout_press_sensor = 5.07*(0.009*fulldata50_array[i, 19]+0.04)
#     DP_corrected = (Vout_press_sensor - 0.21057)/((0.2762-0.21056)/1.4764995)
#     pressure = DP_corrected - HP(density, height, gravity) / 1000
#     Reynolds50_list.append(Re(density, viscosity, flowrate, Dh50, Acs50))
#     f50_list.append(ff(pressure, density, velocity, Dh50))

# for i in range(480):  # 50 % porosity. Test at 50 C. Corresponds to Sheet3 in Origin
#    flowrate = fulldata502_array[i, 13] / 60000
#    pressure = fulldata502_array[i, 19]
#    temperature = np.mean([fulldata502_array[i, 9], fulldata502_array[i, 11]]) + 273.15
#    viscosity = water_viscosity_coolprop(temperature)
#    density = water_density_coolprop(temperature)
#    velocity = flowrate / Acs50
#    Reynolds50_list.append(Re(density, viscosity, flowrate, Dh50, Acs50))
#    f50_list.append(ff(pressure, density, velocity, Dh50))

for i in range(445):  # 50 % porosity. Test at 50 C. Corresponds to Sheet4 in Origin
    flowrate = fulldata503_array[i, 13] / 60000
    temperature = np.mean([fulldata503_array[i, 9], fulldata503_array[i, 11]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs50
    Vout_press_sensor = 5.07 * (0.009 * fulldata503_array[i, 19] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata503_array[i, 19] - HP(density, height, gravity) / 1000
    Reynolds50_list.append(Re(density, viscosity, flowrate, Dh50, Acs50))
    f50_list.append(ff(pressure, density, velocity, Dh50))

for i in range(205):  # 50 % porosity. Test at 24 C. Corresponds to Sheet2 in Origin
    flowrate = fulldata504_array[i, 13] / 60000
    temperature = np.mean([fulldata504_array[i, 9], fulldata504_array[i, 11]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs50
    Vout_press_sensor = 5.07 * (0.009 * fulldata504_array[i, 19] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata504_array[i, 19] - HP(density, height, gravity) / 1000
    Reynolds50_list.append(Re(density, viscosity, flowrate, Dh50, Acs50))
    f50_list.append(ff(pressure, density, velocity, Dh50))

pressure_list = []
HP_list = []
for i in range(460):  # 40 % porosity. Test at 7 C. Corresponds to Sheet5 in Origin
    flowrate = fulldata40_array[i, 13] / 60000
    temperature = np.mean([fulldata40_array[i, 9], fulldata40_array[i, 11]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs40
    Vout_press_sensor = 5.07 * (0.009 * fulldata40_array[i, 19] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata40_array[i, 19] - HP(density, height, gravity)/1000
    pressure_list.append(pressure)
    HP_list.append(HP(density, height, gravity) / 1000)
    Reynolds40_list.append(Re(density, viscosity, flowrate, Dh40, Acs40))
    f40_list.append(ff(pressure, density, velocity, Dh40))

for i in range(460):  # 40 % porosity. Test at 50 C. Corresponds to Sheet3 in Origin
    flowrate = fulldata402_array[i, 13] / 60000
    temperature = np.mean([fulldata402_array[i, 9], fulldata402_array[i, 11]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs40
    Vout_press_sensor = 5.07 * (0.009 * fulldata402_array[i, 19] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata402_array[i, 19] - HP(density, height, gravity)/1000
    pressure_list.append(pressure)
    HP_list.append(HP(density, height, gravity) / 1000)
    Reynolds40_list.append(Re(density, viscosity, flowrate, Dh40, Acs40))
    f40_list.append(ff(pressure, density, velocity, Dh40))

for i in range(125):  # 40 % porosity. Test at 30 C. Corresponds to Sheet4 in Origin
    flowrate = fulldata403_array[i, 13] / 60000
    temperature = np.mean([fulldata403_array[i, 9], fulldata403_array[i, 11]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs40
    Vout_press_sensor = 5.07 * (0.009 * fulldata403_array[i, 19] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata403_array[i, 19] - HP(density, height, gravity)/1000
    pressure_list.append(pressure)
    HP_list.append(HP(density, height, gravity) / 1000)
    Reynolds40_list.append(Re(density, viscosity, flowrate, Dh40, Acs40))
    f40_list.append(ff(pressure, density, velocity, Dh40))

for i in range(800):  # 600um, 50 % porosity. Test at 7 C. Vertical configuration
    flowrate = fulldata_600um_50p_7C_array[i, 9] / 60000
    temperature = np.mean([fulldata_600um_50p_7C_array[i, 5], fulldata_600um_50p_7C_array[i, 7]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs_600um_50p
    Vout_press_sensor = 5.07 * (0.009 * fulldata_600um_50p_7C_array[i, 11] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata504_array[i, 19] - HP(density, height, gravity) / 1000
    Re_600um_50p_7C_list.append(Re(density, viscosity, flowrate, Dh_600um_50p, Acs_600um_50p))
    ff_600um_50p_7C_list.append(ff(pressure, density, velocity, Dh_600um_50p))

for i in range(990):  # 600um, 50 % porosity. Test at 25 C. Vertical configuration
    flowrate = fulldata_600um_50p_25C_array[i, 9] / 60000
    temperature = np.mean([fulldata_600um_50p_25C_array[i, 5], fulldata_600um_50p_25C_array[i, 7]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs_600um_50p
    Vout_press_sensor = 5.07 * (0.009 * fulldata_600um_50p_25C_array[i, 11] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata504_array[i, 19] - HP(density, height, gravity) / 1000
    Re_600um_50p_25C_list.append(Re(density, viscosity, flowrate, Dh_600um_50p, Acs_600um_50p))
    ff_600um_50p_25C_list.append(ff(pressure, density, velocity, Dh_600um_50p))

for i in range(917):  # 600um, 50 % porosity. Test at 50 C. Vertical configuration
    flowrate = fulldata_600um_50p_50C_array[i, 9] / 60000
    temperature = np.mean([fulldata_600um_50p_50C_array[i, 5], fulldata_600um_50p_50C_array[i, 7]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs_600um_50p
    Vout_press_sensor = 5.07 * (0.009 * fulldata_600um_50p_50C_array[i, 11] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata504_array[i, 19] - HP(density, height, gravity) / 1000
    Re_600um_50p_50C_list.append(Re(density, viscosity, flowrate, Dh_600um_50p, Acs_600um_50p))
    ff_600um_50p_50C_list.append(ff(pressure, density, velocity, Dh_600um_50p))

for i in range(1083):  # 600um, 50 % porosity. Test at 50 C. Vertical configuration
    flowrate = fulldata_600um_50p_50C_array_2[i, 11] / 60000
    temperature = np.mean([fulldata_600um_50p_50C_array_2[i, 5], fulldata_600um_50p_50C_array_2[i, 7]]) + 273.15
    viscosity = water_viscosity_coolprop(temperature)
    density = water_density_coolprop(temperature)
    velocity = flowrate / Acs_600um_50p
    Vout_press_sensor = 5.07 * (0.009 * fulldata_600um_50p_50C_array_2[i, 13] + 0.04)
    DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
    pressure = DP_corrected - HP(density, height, gravity) / 1000
    # pressure = fulldata504_array[i, 19] - HP(density, height, gravity) / 1000
    Re_600um_50p_50C_list_2.append(Re(density, viscosity, flowrate, Dh_600um_50p, Acs_600um_50p))
    ff_600um_50p_50C_list_2.append(ff(pressure, density, velocity, Dh_600um_50p))

combined_Re_list = Reynolds_list + Reynolds50_list + Reynolds40_list
combined_f_list = f_list + f50_list + f40_list


def func(x, a):  # x, a, b, c
    return a / x  # a/(x**2)+b/x+c


def func40(x, b, c):
    return b * (1 - 0.078) / x + c  # a/(x**2)+b/x+c


def func45(x, b, c):
    return b * (1 - 0.155) / x + c  # a/(x**2)+b/x+c


def func50(x, b, c):
    return b * (1 - 0.232) / x + c  # a/(x**2)+b/x+c


# par, cov = curve_fit(func, np.log(Reynolds_list), np.log(f_list))
# print(par)
# fit_list = []
# Reynolds_list_sorted = sorted(Reynolds_list)
# x_fit = np.linspace(10, 400, 100)
# for i in range(100):
#     fit = func(x_fit[i], par[0], par[1])
#     fit_list.append(fit)

# ----------------------------------------------------40p fit ------------------------------------
logRe40 = np.log(Reynolds40_list)
logf40 = np.log(f40_list)
coef40 = np.polyfit(logRe40, logf40, 2)
poly40 = np.poly1d(coef40)
x40 = sorted(Reynolds40_list)
yfit40 = lambda Reynolds_list: np.exp(poly40(logRe40))

# fittie40, cov40 = curve_fit(func40, Reynolds40_list, yfit40(Reynolds40_list))
fittie40, cov40 = curve_fit(func40, Reynolds40_list, f40_list)
print(fittie40)
# Reynolds40_list_sorted = sorted(Reynolds40_list)
# yvalues40 = []
# for i in range(len(Reynolds40_list)):
#     yvalue = fittie40[0]/(Reynolds40_list_sorted[i]**2)+fittie40[1]/Reynolds40_list_sorted[i]+fittie40[2]
#     yvalues40.append(yvalue)


# ----------------------------------------- 45 p fit -------------------
logRe = np.log(Reynolds_list)
logf = np.log(f_list)
coef = np.polyfit(logRe, logf, 2)
poly = np.poly1d(coef)
x = sorted(Reynolds_list)
yfit = lambda Reynolds_list: np.exp(poly(logRe))

# fittie, cov = curve_fit(func45,Reynolds_list, yfit(Reynolds_list))
fittie, cov = curve_fit(func45, Reynolds45_list, f45_list)
print(fittie)
# Reynolds_list_sorted = sorted(Reynolds_list)
# yvalues = []
# for i in range(len(Reynolds_list)):
#     yvalue = fittie[0]/(Reynolds_list_sorted[i]**2)+fittie[1]/Reynolds_list_sorted[i]+fittie[2]
#     yvalues.append(yvalue)


# ----------------------------------------------------50p fit ------------------------------------
logRe50 = np.log(Reynolds50_list)
logf50 = np.log(f50_list)
coef50 = np.polyfit(logRe50, logf50, 2)
poly50 = np.poly1d(coef50)
x50 = sorted(Reynolds50_list)
yfit50 = lambda Reynolds_list: np.exp(poly50(logRe50))

# fittie50, cov50 = curve_fit(func50,Reynolds50_list,yfit50(Reynolds50_list))
fittie50, cov50 = curve_fit(func50, Reynolds50_list, f50_list)
print(fittie50)
# print(fittie50[2])
# Reynolds50_list_sorted = sorted(Reynolds50_list)
# yvalues50 = []
# for i in range(len(Reynolds50_list)):
#     yvalue = fittie50[0]/(Reynolds50_list_sorted[i]**2)+fittie50[1]/Reynolds50_list_sorted[i]+fittie50[2]
#     yvalues50.append(yvalue)


# ---------------------------------------------------
# def ff(Re,epsilon):
#     return (150*(1-epsilon))/Re+1.75
#
#
# Reynoldslist = np.linspace(7,350,100)
#
# Reynolds40_new= []
# for i in range(len(Reynolds40_list_sorted)):
#     Reynolds40_new.append(Reynolds40_list_sorted[i] / 0.6)
#
# Reynolds45_new= []
# for i in range(len(Reynolds_list_sorted)):
#     Reynolds45_new.append(Reynolds_list_sorted[i] / 0.55)
#
# Reynolds50_new= []
# for i in range(len(Reynolds50_list_sorted)):
#     Reynolds50_new.append(Reynolds50_list_sorted[i] / 0.5)

# ------------------------------------------------------------

# plt.plot(Reynoldslist,ff(Reynoldslist,0.4),color = 'yellow', label = "40% porosity packed bed", linewidth = 2)
# plt.plot(Reynoldslist,ff(Reynoldslist,0.45),color = 'darkred', label = "45% porosity packed bed", linewidth = 2)
# plt.plot(Reynoldslist,ff(Reynoldslist,0.5),color = 'blue', label = "50% porosity packed bed", linewidth = 2)

# plt.plot(Reynolds40_list, f40_list, '.', color = 'goldenrod', label = '40% porosity')
# plt.plot(Reynolds_list, f_list, '.' , color = 'red', label = '45% Michiel')
# plt.plot(Reynolds45_list, f45_list, '.' , color = 'green', label = '45% porosity')
# plt.plot(Reynolds50_list, f50_list, '.', color = 'steelblue', label = '50% porosity')

plt.plot(Re_600um_50p_7C_list, ff_600um_50p_7C_list, '.', color='c', label='600um_50%p 7C')
plt.plot(Re_600um_50p_25C_list, ff_600um_50p_25C_list, '.', color='m', label='600um_50%p 25C')
# plt.plot(Re_600um_50p_50C_list, ff_600um_50p_50C_list, '.', color='y', label='600um_50%p 50C')
plt.plot(Re_600um_50p_50C_list_2, ff_600um_50p_50C_list_2, '.', color='brown', label='600um_50%p 50C_2')

# plt.plot(Reynolds_list, yfit(Reynolds_list),color = "black")
# plt.plot(Reynolds40_new,yvalues40, color = 'goldenrod',label = "40% porosity fit", linewidth = 2)
# plt.plot(Reynolds45_new,yvalues, color = 'red',label = "45% porosity fit", linewidth = 2)
# plt.plot(Reynolds50_new,yvalues50, color = 'steelblue',label = "50% porosity fit", linewidth = 2)

plt.xlabel("Reynolds number [-]", fontsize=10)
plt.ylabel("Friction factor [-]", fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(labelsize=10)
plt.legend(fontsize=10)
plt.show()
# plt.plot(Reynolds_list, f_list, '.')
# plt.plot(x_fit, fit_list)
# plt.plot(Reynolds_list, yfit(Reynolds_list))
# plt.xlabel("Reynolds number [-]", fontsize=16)
# plt.ylabel("Friction factor [-]", fontsize=16)
# plt.tick_params(labelsize=16)
# plt.show()


# plt.plot(Reynolds40_list, f40_list, '.', color = 'goldenrod', label = '40% porosity experiment, 33% porosity used for calculation')
# plt.plot(Reynolds_list, f_list, '.' , color = 'red', label = '45% porosity experiment, 43% porosity used for calculation')
# plt.plot(Reynolds50_list, f50_list, '.', color = 'steelblue', label = '50% porosity experiment, 50% porosity used for calculation')
# plt.xlabel("Reynolds number [-]", fontsize=18)
# plt.ylabel("Friction factor [-]", fontsize=18)
# plt.xscale('log')
# plt.yscale('log')
# plt.tick_params(labelsize=18)
# plt.legend(fontsize = 20)
# plt.show()
