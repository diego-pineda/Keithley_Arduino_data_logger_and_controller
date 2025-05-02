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
from scipy.optimize import brentq

from compute_uncertainties import compute_flow_rate_uncertainty, compute_flow_area_uncertainty

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


def estimate_intersection_area(d, a):
    """
    Estimate the surface area of the intersection region between two orthogonal cylinders.

    Parameters:
        d (float): Diameter of the cylinders (must be > 0)
        a (float): Offset between the central axes of the two cylinders (must be >= 0 and <= d)

    Returns:
        float: Estimated intersection surface area [same unit as d]^2
    """
    if d <= 0:
        raise ValueError("Diameter 'd' must be positive.")
    if a < 0:
        raise ValueError("Offset 'a' must be non-negative.")
    if a > d:
        raise ValueError("Offset 'a' must be less than or equal to diameter 'd'.")

    ratio = a / d
    return d**2 * 2 * (1 - ratio**(8/5))


def estimate_intersection_volume(d, a):
    """
    Estimate the volume of the intersection region between two orthogonal cylinders.

    Parameters:
        d (float): Diameter of the cylinders (must be > 0)
        a (float): Offset between the central axes of the two cylinders (must be >= 0 and <= d)

    Returns:
        float: Estimated intersection volume [same unit as d]^3
    """
    if d <= 0:
        raise ValueError("Diameter 'd' must be positive.")
    if a < 0:
        raise ValueError("Offset 'a' must be non-negative.")
    if a > d:
        raise ValueError("Offset 'a' must be less than or equal to diameter 'd'.")

    ratio = a / d
    return d**3 * (2/3) * (1 - ratio**(8/5))**2


def calculate_specific_surface_area(d, s1, s2, L, N_L):
    """
    Calculate the specific surface area (beta) of a 3D printed block
    made of orthogonally stacked cylindrical fibers.

    Parameters:
        d (float): Fiber diameter (must be > 0)
        s (float): Fiber spacing in-plane (must be > 0)
        L (float): Total height of the printed block (must be > d) (in the flow direction)
        N_L (int): Number of printed layers (must be >= 2)

    Returns:
        float: Specific surface area (beta) in units of 1/length
    """

    # --- Input validation ---
    if d <= 0 or s1 <= 0 or s2 <= 0 or L <= d:
        raise ValueError("Invalid input: d and s must be > 0, and H must be > d.")
    if N_L < 2:
        raise ValueError("Number of layers N_L must be at least 2.")

    # --- Layer-to-layer spacing ---
    a = (L - d) / (N_L - 1)

    # --- Intersection area per layer pair ---
    A_int = estimate_intersection_area(d, a)

    # --- Total superficial area of unit cell ---
    A_sup_uc = np.pi * d * s1 * (N_L / 2) + np.pi * d * s2 * (N_L / 2) - 2 * (N_L - 1) * A_int

    # --- Specific surface area ---
    beta = A_sup_uc / (s1 * s2 * L)

    return beta


def calculate_porosity(d, s1, s2, L, N_L):
    """
    Calculate the porosity (void fraction) of a 3D printed block made of stacked cylindrical fibers.

    Parameters:
        d (float): Fiber diameter (must be > 0)
        s (float): Fiber spacing in-plane (must be > 0)
        L (float): Total height of the printed block (must be > d) (in the flow direction)
        N_L (int): Number of printed layers (must be >= 2)

    Returns:
        float: Porosity (between 0 and 1)
    """

    # --- Input validation ---
    if d <= 0 or s1 <= 0 or s2 <= 0 or L <= d:
        raise ValueError("Invalid input: d and s must be > 0, and H must be > d.")
    if N_L < 2:
        raise ValueError("Number of layers N_L must be at least 2.")

    # --- Layer-to-layer spacing ---
    a = (L - d) / (N_L - 1)

    # --- Volume of intersections ---
    V_int = estimate_intersection_volume(d, a)

    # --- Total solid volume in unit cell ---
    V_solid_uc = (np.pi * d**2 * s1 * (N_L / 2)) / 4 + (np.pi * d**2 * s2 * (N_L / 2)) / 4 - (N_L - 1) * V_int

    # --- Porosity (void fraction) ---
    eps = 1 - V_solid_uc / (s1 * s2 * L)

    return eps


def effective_diameter(vol_block, block_area):
    """
    Calculates the equivalent particle diameter of the matrix. This would be the diameter of particles of an
    hydraulically equivalent packed bed of (perfectly) spherical particles.

    Args:
        vol_block: volume of solid in the bed in m3
        block_area: wet area of the block in m2

    Returns:
        Equivalent particle diameter of the 3D printed block
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


def ergun_pressure_drop(flowrate, viscosity, density, Dp, L, eps, A_cs):
    """
    Calculates pressure drop in a packed bed using the Ergun equation.

    Args:
        flowrate (float): Volumetric flow rate [m³/s].
        viscosity (float): Fluid dynamic viscosity [Pa·s].
        density (float): Fluid density [kg/m³].
        Dp (float): Particle diameter [m].
        L (float): Length of the packed bed [m].
        eps (float): Porosity (dimensionless).
        A_cs (float): Cross-sectional area of the bed [m²].

    Returns:
        float: Pressure drop [kPa].
    """
    velocity = flowrate / A_cs  # Superficial velocity [m/s]
    term1 = (150 * (1 - eps)**2 / eps**3) * (viscosity * velocity / Dp**2)
    term2 = (1.75 * (1 - eps) / eps**3) * (density * velocity**2 / Dp)
    dp = (term1 + term2) * L  # Pressure drop in [Pa]
    return dp / 1000  # Convert to kPa


def pp_pressure_drop(flowrate, viscosity, density, Dp, L, eps, A_cs):
    """
    Calculates pressure drop in a parallel plate using equivalent correlation.

    Args:
        flowrate (float): Volumetric flow rate [m³/s].
        viscosity (float): Fluid dynamic viscosity [Pa·s].
        density (float): Fluid density [kg/m³].
        Dp (float): Particle diameter [m].
        L (float): Length of the packed bed [m].
        eps (float): Porosity (dimensionless).
        A_cs (float): Cross-sectional area of the bed [m²].

    Returns:
        float: Pressure drop [kPa].
    """
    velocity = flowrate / A_cs  # Superficial velocity [m/s]
    term1 = (108 * (1 - eps)**2 / eps**3) * (viscosity * velocity / Dp**2)
    term2 = 0
    dp = (term1 + term2) * L  # Pressure drop in [Pa]
    return dp / 1000  # Convert to kPa


def plot_pressure_drop_vs_flowrate(data_file, x1, x2, Tw_in_col_index, Tw_out_col_index, flow_col_index, dp_col_index, block_ID):
    """
    Reads flow and pressure data, calculates pressure drops and flow rates,
    and generates a plot comparing measured pressure drop to the prediction
    from the Ergun equation for a packed bed with the same geometric properties.

    Args:
        data_file (str): Path to the tab-separated data file.
        x1 (int): Starting row index (inclusive).
        x2 (int): Ending row index (exclusive).
        Tw_in_col_index (int): Column index for inlet water temperature [°C].
        Tw_out_col_index (int): Column index for outlet water temperature [°C].
        flow_col_index (int): Column index for volumetric flow rate [L/min].
        dp_col_index (int): Column index for differential pressure [kPa].
        block_ID (object): Instance of a class (like Sample) with geometry attributes.
            Required attributes:
                - Dp_eq (float): Equivalent particle diameter [m].
                - A_cs (float): Cross-sectional flow area [m²].
                - porosity (float): Porosity (dimensionless).
                - L_block (float): Length of the block [m].
    """
    Dh = block_ID.Dp_eq
    A_cs = block_ID.A_cs
    eps = block_ID.porosity
    L = block_ID.L_block
    block_name = block_ID.name

    # Read data
    data_frame = pd.read_csv(data_file, sep='\t')
    data_array = pd.DataFrame(data_frame).to_numpy()

    flowrates = []
    pressure_drops = []
    ergun_pressure_drops = []
    pp_pressure_drops = []

    for i in range(x1, x2):
        # Average fluid temperature
        temperature = np.mean([data_array[i, Tw_in_col_index], data_array[i, Tw_out_col_index]]) + 273.15

        # Calculate water properties
        viscosity = water_viscosity_coolprop(temperature)
        density = water_density_coolprop(temperature)

        # Correct the measured pressure drop
        Vout_press_sensor = 5.07 * (0.009 * data_array[i, dp_col_index] + 0.04)
        DP_corrected = (Vout_press_sensor - 0.21057) / ((0.2762 - 0.21056) / 1.4764995)
        pressure_drop = DP_corrected - HP(density, 0.11, 9.8) / 1000  # [kPa]

        # Convert flow rate to [m³/s]
        flowrate = data_array[i, flow_col_index] / 60000

        # Store data for plotting
        flowrates.append(flowrate * 60000)  # Convert back to L/min for the plot
        pressure_drops.append(pressure_drop)

        # Ergun prediction for the same flow rate and conditions
        ergun_dp = ergun_pressure_drop(flowrate, viscosity, density, Dh, L, eps, A_cs)
        ergun_pressure_drops.append(ergun_dp)

        pp_dp = pp_pressure_drop(flowrate, viscosity, density, Dh, L, eps, A_cs)
        pp_pressure_drops.append(pp_dp)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(flowrates, pressure_drops, 'o', label=f'Measured data block {block_name}', color='tab:blue')
    plt.plot(flowrates, ergun_pressure_drops, '^', label='Equivalent packed bed (Ergun equation)', color='tab:red')
    plt.plot(flowrates, pp_pressure_drops, 's', label=r'Equivalent parallel-plate stack ($f_D=96/Re$)', color='magenta')

    plt.xlabel('Volumetric Flow Rate [L/min]')
    plt.ylabel('Pressure Drop [kPa]')
    # plt.title('Pressure Drop vs Flow Rate')
    plt.legend()
    # plt.grid(True)
    plt.show()


def reynolds_friction(data_file, *, x1, x2, Tw_in_col_index, Tw_out_col_index, flow_col_index, dp_col_index, block_ID):  # Dh, A_cs, eps, L,
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

    A_cs = block_ID.A_cs
    Dp_eq = block_ID.Dp_eq
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
        dflow_rate = compute_flow_rate_uncertainty(data_array[i, flow_col_index]) # [m3/s] Error bars in flow rate data

        velocity = flowrate / A_cs  # [m/s] Superficial velocity

        Re_list.append(Re(density, viscosity, velocity, Dp_eq, eps))
        ff_list.append(ff(pressure, density, velocity, Dp_eq, L, eps))

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
    k (float): Thermal conductivity  of solid matrix [W/m·K].
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

            # R_thermal_2 = R_total / (abs(q1 / q2) + 1)
            # R_thermal_1 = R_total - R_thermal_2
            # T1_adjusted = T1_measured - R_thermal_1 * abs(q1)
            # T2_adjusted = T2_measured - R_thermal_2 * abs(q2)

            h_solution = solve_h(h_initial, x_known, T1_measured, sample.H_block, T_f, g_dot, sample.beta, k_eff, q1, q2)
            T2_calculated = temperature_profile(sample.H_block, sample.H_block, T_f, g_dot, h_solution, sample.beta, k_eff, q1, q2)
            return T2_calculated - T2_measured

        # Solve for alpha
        # try:
        #     alpha_solution = brentq(alpha_equation, 0, 1)  # Ensures alpha is between 0 and 1
        # except ValueError:
        #     alpha_solution = 0.5  # Fallback in case of failure
        alpha_solution = fsolve(alpha_equation, alpha_initial, xtol=1e-5)[0]

        # Calculate final q1 and q2 using the solved alpha
        q1_final = Q_contact_resistances / (sample.L_block * sample.W_block) * alpha_solution
        q2_final = -Q_contact_resistances / (sample.L_block * sample.W_block) * (1 - alpha_solution)

        # print("relations", T1_measured/T2_measured, abs(q1_final/q2_final), alpha_solution)

        # R_thermal_2 = R_total / (abs(q1_final / q2_final) + 1)
        # R_thermal_1 = R_total - R_thermal_2
        #
        # T1_adjusted = T1_measured - R_thermal_1 * abs(q1_final)
        # T2_adjusted = T2_measured - R_thermal_2 * abs(q2_final)
        # Solve for h
        h_solution = solve_h(h_initial, x_known, T1_measured, sample.H_block, T_f, g_dot, sample.beta, k_eff, q1_final, q2_final)
        T2_recalculated = temperature_profile(sample.H_block, sample.H_block, T_f, g_dot, h_solution, sample.beta, k_eff, q1_final, q2_final)

        # print(f"T1_measured = {T1_measured}", f"T1_adjusted = {T1_adjusted}")
        # print(f"T2_measured = {T2_measured}", f"T2_adjusted = {T2_adjusted}")
        # print(f"T2_recalculated = {T2_recalculated}", f"T2_adjusted = {T2_measured}")

        # Calculate fluid properties
        density = PropsSI('D', 'T', T_f, 'P', 101325, 'Water')
        viscosity = PropsSI('V', 'T', T_f, 'P', 101325, 'Water')
        conductivity = PropsSI('L', 'T', T_f, 'P', 101325, 'Water')
        cp = PropsSI('C', 'T', T_f, 'P', 101325, 'Water')
        prandtl = (cp * viscosity) / conductivity

        # Calculate velocities, Reynolds, Nusselt
        superficial_velocity = flow_rate / 60000 / sample.A_cs
        reynolds_number = (density * superficial_velocity * sample.Dp_eq) / (viscosity)
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
    def __init__(self, *, V_solid_archimedes, porosity_Arch, N_layers, N_fibers, D_fibers, S1_fibers, S2_fibers, L_block, W_block, H_block, H_flow, W_flow, name):
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
        :param A_inters: [m^2] Area of intersection points between fibers
        :param name: sample's name
        """

        self.V_solid_archimedes = V_solid_archimedes
        self.porosity_Arch      = porosity_Arch

        self.N_layers           = N_layers
        self.N_fibers           = N_fibers
        self.D_fibers           = D_fibers
        self.S1_fibers          = S1_fibers
        self.S2_fibers          = S2_fibers

        self.L_block            = L_block  # In the flow direction
        self.W_block            = W_block  # In the cross section
        self.H_block            = H_block  # In the cross section
        self.H_flow             = H_flow  # In the cross section
        self.W_flow             = W_flow  # In the cross section
        self.name               = name

        self.A_cs = W_flow * H_flow  # [m2] Area of cross section of the block available for flow

        # self.layer_height = (L_block - D_fibers)/(N_layers - 1)  # ************** Not really necessary ***************
        # self.A_inters = estimate_intersection_area(D_fibers, self.layer_height)  # ******* Not really necessary ********

        self.calc_porosity = calculate_porosity(D_fibers, S1_fibers, S2_fibers, L_block, N_layers)
        self.beta = calculate_specific_surface_area(D_fibers, S1_fibers, S2_fibers, L_block, N_layers)

        self.A_solid = self.beta * (W_flow * H_flow * L_block)
        self.calc_solid_volume = (1 - self.calc_porosity) * (W_flow * H_flow * L_block)

        self.Dp_eq = effective_diameter(self.calc_solid_volume, self.A_solid)  # Could be calculated with Arch volume
        self.porosity = self.calc_porosity  # This is to decide if using measured or calculated porosity in further calc
        self.Dh = 4 * self.calc_porosity / self.beta  # Hydraulic diameter

        # self.Dp_eq       =  6 * V_solid_archimedes / self.A_solid  # W_block / (N_fibers-1) - D_fibers  # Equivalent particle diameter
        # self.A_solid  = (np.pi * D_fibers * L_fibers * N_fibers * N_layers - 2 * N_fibers**2 * (N_layers-1) * self.A_inters)  # D_fibers**2
        # self.beta     = self.A_solid / (self.W_block * self.H_block * self.L_block)
        # self.V_solid_calc = np.pi * D_fibers**2 * L_fibers * N_fibers * N_layers / 4 - N_fibers**2 * (1-N_layers) * V_inters


# DP_90deg_600um_40p_a = Sample(V_solid_archimedes=1.091E-6, N_layers=18, N_fibers=21, D_fibers=530.53e-6, S1_fibers=740.53e-6, S2_fibers=728.22e-6, porosity_Arch=0.29, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.013108, H_flow=0.014070, name='90deg_600um_29p_a')
# DP_90deg_600um_45p_a = Sample(V_solid_archimedes=1.058E-6, N_layers=18, N_fibers=19, D_fibers=538.83e-6, S1_fibers=538.83e-6+243.53e-6, S2_fibers=538.83e-6+243.53e-6, porosity_Arch=0.31, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.015, H_flow=0.015, name='90deg_600um_31p_a')
# DP_90deg_600um_50p_a = Sample(V_solid_archimedes=1.000E-6, N_layers=18, N_fibers=17, D_fibers=530.88e-6, S1_fibers=893e-6, S2_fibers=893e-6, porosity_Arch=0.35, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.0134, H_flow=0.01468, name='90deg_600um_35p_a')
#
# DP_90deg_400um_40p_a = Sample(V_solid_archimedes=1.199E-6, N_layers=24, N_fibers=33, D_fibers=401.61e-6, S1_fibers=401.61e-6+88.56e-6, S2_fibers=401.61e-6+88.56e-6, porosity_Arch=0.27, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.014, H_flow=0.0144, name='90deg_400um_27p_a')
# # DP_90deg_400um_45p_a = Sample(V_solid_archimedes=1.027E-6, N_layers=24, N_fibers=31, D_fibers=368.32e-6, S1_fibers=368.32e-6+126.85e-6, S2_fibers=368.32e-6+126.85e-6, porosity_Arch=0.331, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.014, H_flow=0.0143, name='90deg_400um_33p_a')
# DP_90deg_400um_45p_a = Sample(V_solid_archimedes=1.027E-6, N_layers=24, N_fibers=31, D_fibers=390.38e-6, S1_fibers=536.45e-6, S2_fibers=537.84e-6, porosity_Arch=0.331, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.013984, H_flow=0.014484, name='90deg_400um_33p_a')
# DP_90deg_400um_50p_a = Sample(V_solid_archimedes=0.958E-6, N_layers=24, N_fibers=29, D_fibers=401.30e-6, S1_fibers=597.53e-6, S2_fibers=585.93e-6, porosity_Arch=0.376, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.014341, H_flow=0.014648, name='400 um fibers 37 % porosity')

DP_90deg_600um_40p_a = Sample(V_solid_archimedes=1.091E-6, N_layers=18, N_fibers=21, D_fibers=509.846e-6, S1_fibers=735.03e-6, S2_fibers=736.37e-6, porosity_Arch=0.29, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.013231, H_flow=0.013991, name='90deg_600um_29p_a')
DP_90deg_600um_45p_a = Sample(V_solid_archimedes=1.058E-6, N_layers=18, N_fibers=19, D_fibers=542.653e-6, S1_fibers=814.71e-6, S2_fibers=817.49e-6, porosity_Arch=0.31, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.013850, H_flow=0.013897, name='90deg_600um_31p_a')
DP_90deg_600um_50p_a = Sample(V_solid_archimedes=1.000E-6, N_layers=18, N_fibers=17, D_fibers=554.218e-6, S1_fibers=895.07e-6, S2_fibers=885.72e-6, porosity_Arch=0.35, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.013426, H_flow=0.013286, name='90deg_600um_35p_a')

DP_90deg_400um_40p_a = Sample(V_solid_archimedes=1.199E-6, N_layers=24, N_fibers=33, D_fibers=401.61e-6, S1_fibers=401.61e-6+88.56e-6, S2_fibers=401.61e-6+88.56e-6, porosity_Arch=0.27, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.014, H_flow=0.0144, name='90deg_400um_27p_a')
# DP_90deg_400um_45p_a = Sample(V_solid_archimedes=1.027E-6, N_layers=24, N_fibers=31, D_fibers=368.32e-6, S1_fibers=368.32e-6+126.85e-6, S2_fibers=368.32e-6+126.85e-6, porosity_Arch=0.331, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.014, H_flow=0.0143, name='90deg_400um_33p_a')
DP_90deg_400um_45p_a = Sample(V_solid_archimedes=1.027E-6, N_layers=24, N_fibers=31, D_fibers=384.58e-6, S1_fibers=536.45e-6, S2_fibers=542.74e-6, porosity_Arch=0.331, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.014111, H_flow=0.014484, name='90deg_400um_33p_a')
DP_90deg_400um_50p_a = Sample(V_solid_archimedes=0.958E-6, N_layers=24, N_fibers=29, D_fibers=402.114e-6, S1_fibers=597.53e-6, S2_fibers=595.29e-6, porosity_Arch=0.376, L_block=0.0062, W_block=0.016, H_block=0.016, W_flow=0.014341, H_flow=0.014882, name='400 um fibers 37 % porosity')



DP_45deg_250um_45p_a = Sample(V_solid_archimedes=0.869e-6,
                              porosity_Arch=0.43,
                              N_layers=34,
                              N_fibers=50,
                              D_fibers=213.8e-6,
                              S1_fibers=213.8e-6+140.67e-6,
                              S2_fibers=213.8e-6+140.67e-6,
                              L_block=0.006,
                              W_block=0.016,
                              H_block=0.016,
                              W_flow=0.0155,
                              H_flow=0.0155,
                              name="45deg_250um_45p_a")

print(f'D_eq_600um_40p_a = {DP_90deg_600um_40p_a.Dp_eq}')
print(f'D_eq_600um_45p_a = {DP_90deg_600um_45p_a.Dp_eq}')
print(f'D_eq_600um_50p_a = {DP_90deg_600um_50p_a.Dp_eq}')
print(f'D_eq_400um_40p_a = {DP_90deg_400um_40p_a.Dp_eq}')
print(f'D_eq_400um_45p_a = {DP_90deg_400um_45p_a.Dp_eq}')
print(f'D_eq_400um_50p_a = {DP_90deg_400um_50p_a.Dp_eq}')

print(f'\nAs_600um_40p_a = {DP_90deg_600um_40p_a.A_solid}')
print(f'As_600um_45p_a = {DP_90deg_600um_45p_a.A_solid}')
print(f'As_600um_50p_a = {DP_90deg_600um_50p_a.A_solid}')
print(f'As_400um_40p_a = {DP_90deg_400um_40p_a.A_solid}')
print(f'As_400um_45p_a = {DP_90deg_400um_45p_a.A_solid}')
print(f'As_400um_50p_a = {DP_90deg_400um_50p_a.A_solid}')

print(f'\nBeta_600um_40p_a = {DP_90deg_600um_40p_a.beta}')
print(f'Beta_600um_45p_a = {DP_90deg_600um_45p_a.beta}')
print(f'Beta_600um_50p_a = {DP_90deg_600um_50p_a.beta}')
print(f'Beta_400um_40p_a = {DP_90deg_400um_40p_a.beta}')
print(f'Beta_400um_45p_a = {DP_90deg_400um_45p_a.beta}')
print(f'Beta_400um_50p_a = {DP_90deg_400um_50p_a.beta}')

print(f'\neps_400um_40p_a = {DP_90deg_400um_40p_a.porosity}')
print(f'eps_400um_45p_a = {DP_90deg_400um_45p_a.porosity}')
print(f'eps_400um_50p_a = {DP_90deg_400um_50p_a.porosity}')
print(f'eps_600um_40p_a = {DP_90deg_600um_40p_a.porosity}')
print(f'eps_600um_45p_a = {DP_90deg_600um_45p_a.porosity}')
print(f'eps_600um_50p_a = {DP_90deg_600um_50p_a.porosity}')

# print(f'Beta_400um_50p_a = {calculate_specific_surface_area(DP_90deg_400um_50p_a.D_fibers, (DP_90deg_400um_50p_a.W_block - DP_90deg_400um_50p_a.D_fibers)/(DP_90deg_400um_50p_a.N_fibers - 1), DP_90deg_400um_50p_a.L_block, DP_90deg_400um_50p_a.N_layers)}')
# print(f'eps_400um_50p_a = {calculate_porosity(DP_90deg_400um_50p_a.D_fibers, (DP_90deg_400um_50p_a.W_block - DP_90deg_400um_50p_a.D_fibers)/(DP_90deg_400um_50p_a.N_fibers - 1), DP_90deg_400um_50p_a.L_block, DP_90deg_400um_50p_a.N_layers)}')


# # ------------------------- 600um 40 % porosity sample ---------------
#
Re_600um_40perc_10C, ff_600um_40perc_10C = reynolds_friction("./Sensor_data/09Oct2024_90deg_600um_40p_a_press_drop_10C.csv",
                                                             x1=1,
                                                             x2=658,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=7,
                                                             flow_col_index=11,
                                                             dp_col_index=13,
                                                             block_ID=DP_90deg_600um_40p_a)  # .Dh, 0.015 * 0.015, 0.29, 0.0062 area was 0.016 ** 2 * 0.29 and then 0.000180**2 * 21**2  Dh was 0.180
#
# # plot_pressure_drop_vs_flowrate("./Sensor_data/09Oct2024_90deg_600um_40p_a_press_drop_10C.csv",
# #                                1, 658, 5, 7, 11, 13, DP_90deg_600um_40p_a)
#
Re_600um_40perc_22C, ff_600um_40perc_22C = reynolds_friction("./Sensor_data/09Oct2024_90deg_600um_40p_a_press_drop_22C.csv",
                                                             x1=1,
                                                             x2=794,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=7,
                                                             flow_col_index=11,
                                                             dp_col_index=13,
                                                             block_ID=DP_90deg_600um_40p_a)  # .Dh, 0.015 * 0.015, 0.29, 0.0062
#
# # plot_pressure_drop_vs_flowrate("./Sensor_data/09Oct2024_90deg_600um_40p_a_press_drop_22C.csv",
# #                                1, 794, 5, 7, 11, 13, DP_90deg_600um_40p_a)
#
Re_600um_40perc_50C, ff_600um_40perc_50C = reynolds_friction("./Sensor_data/09Oct2024_90deg_600um_40p_a_press_drop_50C.csv",
                                                             x1=1,
                                                             x2=773,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=7,
                                                             flow_col_index=11,
                                                             dp_col_index=13,
                                                             block_ID=DP_90deg_600um_40p_a)
#
# # plot_pressure_drop_vs_flowrate("./Sensor_data/09Oct2024_90deg_600um_40p_a_press_drop_50C.csv",
# #                                1, 773, 5, 7, 11, 13, DP_90deg_600um_40p_a)
#
#
# h_600um_40perc, Vflow_600um_40perc, Re_htc_600um_40perc, Nu_600um_40perc, data_600um_40perc = calculate_heat_transfer_coefficients_with_sample("data_htc_600um_40p.xlsx", DP_90deg_600um_40p_a, 6, 0.598, 0.5, 0.1e-4)
#
#
#
# # ------------------------- 600um 45 % porosity sample ---------------
Re_600um_45perc_7C, ff_600um_45perc_7C = reynolds_friction("./Sensor_data/17Sep2024_90deg_600um_45p_a_press_drop_7C.csv",
                                                           x1=1,
                                                           x2=741,
                                                           Tw_in_col_index=5,
                                                           Tw_out_col_index=7,
                                                           flow_col_index=11,
                                                           dp_col_index=13,
                                                           block_ID=DP_90deg_600um_45p_a)  # Dh was 260  area was 0.000260**2 * (19*18)
#
# # plot_pressure_drop_vs_flowrate("./Sensor_data/17Sep2024_90deg_600um_45p_a_press_drop_7C.csv",
# #                                1, 741, 5, 7, 11, 13, DP_90deg_600um_45p_a)
#
Re_600um_45perc_21C, ff_600um_45perc_21C = reynolds_friction("./Sensor_data/17Sep2024_90deg_600um_45p_a_press_drop_21C.csv",
                                                             x1=1,
                                                             x2=966,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=7,
                                                             flow_col_index=11,
                                                             dp_col_index=13,
                                                             block_ID=DP_90deg_600um_45p_a)  # Dh_600um_45p_a, 0.015 * 0.015, 0.31, 0.0062
#
# # plot_pressure_drop_vs_flowrate("./Sensor_data/17Sep2024_90deg_600um_45p_a_press_drop_21C.csv",
# #                                1, 966, 5, 7, 11, 13, DP_90deg_600um_45p_a)
#
Re_600um_45perc_50C, ff_600um_45perc_50C = reynolds_friction("./Sensor_data/13Sep2024_90deg_600um_45p_a_press_drop_50C_a.csv",
                                                             x1=1,
                                                             x2=1185,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=7,
                                                             flow_col_index=11,
                                                             dp_col_index=13,
                                                             block_ID=DP_90deg_600um_45p_a)
#
# # plot_pressure_drop_vs_flowrate("./Sensor_data/13Sep2024_90deg_600um_45p_a_press_drop_50C_a.csv",
# #                                1, 1185, 5, 7, 11, 13, DP_90deg_600um_45p_a)
#
# # ------------------------- 600um 50 % porosity sample ---------------
Re_600um_50perc_7C, ff_600um_50perc_7C = reynolds_friction("./Sensor_data/29Aug2024_90deg_600um_50p_a_press_drop_7C.csv",
                                                           x1=1,
                                                           x2=750,
                                                           Tw_in_col_index=5,
                                                           Tw_out_col_index=7,
                                                           flow_col_index=9,
                                                           dp_col_index=11,
                                                           block_ID=DP_90deg_600um_50p_a)  # Dh was 340 area was  0.000340**2 * (17*16)
#
# # plot_pressure_drop_vs_flowrate("./Sensor_data/29Aug2024_90deg_600um_50p_a_press_drop_7C.csv",
# #                                1, 800, 5, 7, 9, 11, DP_90deg_600um_50p_a)
#
Re_600um_50perc_25C, ff_600um_50perc_25C = reynolds_friction("./Sensor_data/29Aug2024_90deg_600um_50p_a_press_drop_25C.csv",
                                                             x1=1,
                                                             x2=790,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=7,
                                                             flow_col_index=9,
                                                             dp_col_index=11,
                                                             block_ID=DP_90deg_600um_50p_a)  # Dh_600um_50p_a, 0.015 * 0.015, 0.35, 0.0062
#
# # plot_pressure_drop_vs_flowrate("./Sensor_data/29Aug2024_90deg_600um_50p_a_press_drop_25C.csv",
# #                                1, 990, 5, 7, 9, 11, DP_90deg_600um_50p_a)
#
Re_600um_50perc_50C, ff_600um_50perc_50C = reynolds_friction("./Sensor_data/30Aug2024_90deg_600um_50p_press_drop_50C_2.csv",
                                                             x1=1,
                                                             x2=917,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=7,
                                                             flow_col_index=11,
                                                             dp_col_index=13,
                                                             block_ID=DP_90deg_600um_50p_a)  # 19Aug2024_90deg_600um_50p_a_press_drop_50C.csv
#
# # plot_pressure_drop_vs_flowrate("./Sensor_data/30Aug2024_90deg_600um_50p_press_drop_50C_2.csv",
# #                                1, 917, 5, 7, 11, 13, DP_90deg_600um_50p_a)
#
# h_600um_50perc, Vflow_600um_50perc, Re_htc_600um_50perc, Nu_600um_50perc, data_600um_50perc = calculate_heat_transfer_coefficients_with_sample("600um_50p_averages_output.xlsx", DP_90deg_600um_50p_a, 6, 0.598, 0.5, 0.1e-4)
# #"data_htc_600um_50p_V2.xlsx"

# ------------------------- 400um 40 % porosity sample ------------------------

Re_400um_40perc_low, ff_400um_40perc_low = reynolds_friction("./Sensor_data/19Jun2024_40p_VerticalRefsweep.csv",
                                                             x1=1,
                                                             x2=460,
                                                             Tw_in_col_index=9,
                                                             Tw_out_col_index=11,
                                                             flow_col_index=13,
                                                             dp_col_index=19,
                                                             block_ID=DP_90deg_400um_40p_a)  # Dh was 260  area was 0.000260**2 * (19*18)
Re_400um_40perc_mid, ff_400um_40perc_mid = reynolds_friction("./Sensor_data/19Jun2024_40p_Verticalmediumtemp.csv",
                                                             x1=1,
                                                             x2=124,
                                                             Tw_in_col_index=9,
                                                             Tw_out_col_index=11,
                                                             flow_col_index=13,
                                                             dp_col_index=19,
                                                             block_ID=DP_90deg_400um_40p_a)  # Dh_600um_45p_a, 0.015 * 0.015, 0.31, 0.0062
Re_400um_40perc_hi, ff_400um_40perc_hi = reynolds_friction("./Sensor_data/19Jun2024_40p_Verticalhottest.csv",
                                                           x1=1,
                                                           x2=460,
                                                           Tw_in_col_index=9,
                                                           Tw_out_col_index=11,
                                                           flow_col_index=13,
                                                           dp_col_index=19,
                                                           block_ID=DP_90deg_400um_40p_a)
#
#
# h_400um_40perc, Vflow_400um_40perc, Re_htc_400um_40perc, Nu_400um_40perc, data_400um_40perc = calculate_heat_transfer_coefficients_with_sample("data_htc_400um_40p.xlsx", DP_90deg_400um_40p_a, 6, 0.598, 0.5, 1.2e-4)  #1.286e-4

# ------------------------- 400um 45 % porosity sample ---------------
Re_400um_45perc_low, ff_400um_45perc_low = reynolds_friction("./Sensor_data/26Jul2024_400um_45perc_pressure_6C.csv",
                                                             x1=1,
                                                             x2=580,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=6,
                                                             flow_col_index=8,
                                                             dp_col_index=10,
                                                             block_ID=DP_90deg_400um_45p_a)  # Dh was 260  area was 0.000260**2 * (19*18)
Re_400um_45perc_mid, ff_400um_45perc_mid = reynolds_friction("./Sensor_data/19Jul2024_45por400_flow25C.csv",
                                                             x1=2,
                                                             x2=1161,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=6,
                                                             flow_col_index=8,
                                                             dp_col_index=10,
                                                             block_ID=DP_90deg_400um_45p_a)  # Dh_600um_45p_a, 0.015 * 0.015, 0.31, 0.0062
Re_400um_45perc_hi, ff_400um_45perc_hi = reynolds_friction("./Sensor_data/26Jul2024_400um_45per_pressure_45C.csv",
                                                           x1=2,
                                                           x2=978,
                                                           Tw_in_col_index=5,
                                                           Tw_out_col_index=6,
                                                           flow_col_index=8,
                                                           dp_col_index=10,
                                                           block_ID=DP_90deg_400um_45p_a)

# h_400um_45perc, Vflow_400um_45perc, Re_htc_400um_45perc, Nu_400um_45perc, data_400um_45perc = calculate_heat_transfer_coefficients_with_sample("data_htc_400um_45p.xlsx", DP_90deg_400um_45p_a, 6, 0.598, 0.5, 1.2e-4)  #1.286e-4

# ------------------------- 400um 50 % porosity sample ---------------
Re_400um_50perc_mid, ff_400um_50perc_mid = reynolds_friction("./Sensor_data/19Jun2024_50p_Ambientgapclosing.csv",
                                                             x1=1,
                                                             x2=205,
                                                             Tw_in_col_index=9,
                                                             Tw_out_col_index=11,
                                                             flow_col_index=13,
                                                             dp_col_index=19,
                                                             block_ID=DP_90deg_400um_50p_a)  # Dh_600um_45p_a, 0.015 * 0.015, 0.31, 0.0062
# plot_pressure_drop_vs_flowrate("./Sensor_data/19Jun2024_50p_Ambientgapclosing.csv",
#                                1, 205, 9, 11, 13, 19, DP_90deg_400um_50p_a)
#
Re_400um_50perc_hi, ff_400um_50perc_hi = reynolds_friction("./Sensor_data/19Jun2024_50p_Hottestvertical.csv",
                                                           x1=1,
                                                           x2=445,
                                                           Tw_in_col_index=9,
                                                           Tw_out_col_index=11,
                                                           flow_col_index=13,
                                                           dp_col_index=19,
                                                           block_ID=DP_90deg_400um_50p_a)
#
# plot_pressure_drop_vs_flowrate("./Sensor_data/19Jun2024_50p_Hottestvertical.csv",
#                                1, 445, 9, 11, 13, 19, DP_90deg_400um_50p_a)
# ----------------------------------------------------------------------------------------------------------------------

Re_250um_45perc_hi, ff_250um_45perc_hi = reynolds_friction("./Sensor_data/20Mar2025_45deg_250um_45p_a_DP_45C_HiLo_flow.csv",
                                                           x1=1,
                                                           x2=1361,
                                                           Tw_in_col_index=5,
                                                           Tw_out_col_index=4,
                                                           flow_col_index=12,
                                                           dp_col_index=14,
                                                           block_ID=DP_45deg_250um_45p_a)

Re_250um_45perc_mid, ff_250um_45perc_mid = reynolds_friction("./Sensor_data/12Mar2025_45deg_250um_45p_a_DP_Tamb.csv",
                                                           x1=1,
                                                           x2=1054,
                                                           Tw_in_col_index=5,
                                                           Tw_out_col_index=4,
                                                           flow_col_index=12,
                                                           dp_col_index=14,
                                                           block_ID=DP_45deg_250um_45p_a)

Re_250um_45perc_low, ff_250um_45perc_low = reynolds_friction("./Sensor_data/20Mar2025_45deg_250um_45p_a_DP_6C.csv",
                                                             x1=1,
                                                             x2=1381,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=4,
                                                             flow_col_index=12,
                                                             dp_col_index=14,
                                                             block_ID=DP_45deg_250um_45p_a)

Re_250um_45perc_low_flow, ff_250um_45perc_low_flow = reynolds_friction("./Sensor_data/20Mar2025_45deg_250um_45p_a_DP_6C_low_flow.csv",
                                                             x1=1,
                                                             x2=525,
                                                             Tw_in_col_index=5,
                                                             Tw_out_col_index=4,
                                                             flow_col_index=12,
                                                             dp_col_index=14,
                                                             block_ID=DP_45deg_250um_45p_a)


# Fitting data to a function of the form ff = a / Re + b, using modified friction factor and modified Re number

# single_Re_data_list = Re_400um_40perc_low + Re_400um_40perc_mid + Re_400um_40perc_hi + \
#                       Re_400um_45perc_low + Re_400um_45perc_mid + Re_400um_45perc_hi + \
#                       Re_400um_50perc_mid + Re_400um_50perc_hi
# single_ff_data_list = ff_400um_40perc_low + ff_400um_40perc_mid + ff_400um_40perc_hi + \
#                       ff_400um_45perc_low + ff_400um_45perc_mid + ff_400um_45perc_hi + \
#                       ff_400um_50perc_mid + ff_400um_50perc_hi

single_Re_data_list = Re_600um_40perc_10C + Re_600um_40perc_22C + Re_600um_40perc_50C + \
                      Re_600um_45perc_7C + Re_600um_45perc_21C + Re_600um_45perc_50C + \
                      Re_600um_50perc_7C + Re_600um_50perc_25C + Re_600um_50perc_50C + \
                      Re_400um_40perc_low + Re_400um_40perc_mid + Re_400um_40perc_hi + \
                      Re_400um_45perc_low + Re_400um_45perc_mid + Re_400um_45perc_hi + \
                      Re_400um_50perc_mid + Re_400um_50perc_hi

single_ff_data_list = ff_600um_40perc_10C + ff_600um_40perc_22C + ff_600um_40perc_50C + \
                      ff_600um_45perc_7C + ff_600um_45perc_21C + ff_600um_45perc_50C + \
                      ff_600um_50perc_7C + ff_600um_50perc_25C + ff_600um_50perc_50C + \
                      ff_400um_40perc_low + ff_400um_40perc_mid + ff_400um_40perc_hi + \
                      ff_400um_45perc_low + ff_400um_45perc_mid + ff_400um_45perc_hi + \
                      ff_400um_50perc_mid + ff_400um_50perc_hi


# print(single_ff_data_list)
# print(single_Re_data_list)

# Define your expression: y = a / x + b
x, a, b, c = symbols('x a b c')
KTA_like = a / x + b / x ** c
Ergun_like = a / x + b
fit_result_KTA = fit_function(KTA_like, single_Re_data_list, single_ff_data_list)
fit_result_Ergun = fit_function(Ergun_like, single_Re_data_list, single_ff_data_list)
print(list(fit_result_KTA['params'].values()))
print(list(fit_result_Ergun['params'].values()))
print(list(fit_result_KTA['covariance']))
print(list(fit_result_Ergun['covariance']))

popt, pcov = curve_fit(ff_vs_Re, single_Re_data_list, single_ff_data_list)
a_fit, b_fit = popt

print(f"ff = {a_fit} / Re + {b_fit}")

# single_Re_for_htc_data_list = Re_htc_600um_40perc + Re_htc_600um_50perc
# single_Nu_data_list = Nu_600um_40perc + Nu_600um_50perc
# expr3 = a + b * x**c
# fit_result_Nu = fit_function(expr3, single_Re_for_htc_data_list, single_Nu_data_list)
# print(f"Nu correlation parameters = {list(fit_result_Nu['params'].values())}")

# ----------------------------------------------------------------------------------------------------------------------
# Plotting results
plt.figure()
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(KTA_like, np.linspace(1, 1000, 1000), *list(fit_result_KTA['params'].values())), '--r', label='Fit by KTA')
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(Ergun_like, np.linspace(1, 1000, 1000), *list(fit_result_Ergun['params'].values())), '--b', label='Fit by Ergun')
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(KTA_like, np.linspace(1, 1000, 1000), *[160, 3, 0.1]), '-r', label='KTA (1981)')
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(Ergun_like, np.linspace(1, 1000, 1000), *[150, 1.75]), '-b', label='Ergun (1952)')
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(a/x, np.linspace(1, 1000, 1000), *[108]), '-m', label='Parallel plate')
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(a/x, np.linspace(1, 1000, 1000), *[72]), '-c', label='Pure internal laminar flow')
# plt.plot(np.linspace(1, 1000, 1000), ff_vs_Re(np.linspace(1, 1000, 1000), a_fit, b_fit), '-k', label='Fit')
# plt.plot(np.linspace(1, 1000, 1000), ff_vs_Re(np.linspace(1, 1000, 1000), 150, 1.75), '-r', label='Ergun')

plt.plot(Re_600um_40perc_10C, ff_600um_40perc_10C, '.', color='c', label='600um_29%p 10 °C')
plt.plot(Re_600um_40perc_22C, ff_600um_40perc_22C, '.', color='m', label='600um_29%p 22 °C')
plt.plot(Re_600um_40perc_50C, ff_600um_40perc_50C, '.', color='brown', label='600um_29%p 50 °C')
#
plt.plot(Re_600um_45perc_7C, ff_600um_45perc_7C, '.', color='Olive', label='600um_31%p 7 °C')
plt.plot(Re_600um_45perc_21C, ff_600um_45perc_21C, '.', color='Purple', label='600um_31%p 21 °C')
plt.plot(Re_600um_45perc_50C, ff_600um_45perc_50C, '.', color='Violet', label='600um_31%p 50 °C')
#
plt.plot(Re_600um_50perc_7C, ff_600um_50perc_7C, '.', color='DarkGreen', label='600um_35%p 7 °C')
plt.plot(Re_600um_50perc_25C, ff_600um_50perc_25C, '.', color='blue', label='600um_35%p 25 °C')
plt.plot(Re_600um_50perc_50C, ff_600um_50perc_50C, '.', color='red', label='600um_35%p 50 °C')

plt.plot(Re_400um_40perc_low, ff_400um_40perc_low, '.', color='Gray', label='400um_27%p 7 °C')
plt.plot(Re_400um_40perc_mid, ff_400um_40perc_mid, '.', color='Maroon', label='400um_27%p 25 °C')
plt.plot(Re_400um_40perc_hi, ff_400um_40perc_hi, '.', color='Navy', label='400um_27%p 50 °C')

plt.plot(Re_400um_45perc_low, ff_400um_45perc_low, '.', color='Pink', label='400um_33%p 7 °C')
plt.plot(Re_400um_45perc_mid, ff_400um_45perc_mid, '.', color='Teal', label='400um_33%p 25 °C')
plt.plot(Re_400um_45perc_hi, ff_400um_45perc_hi, '.', color='blue', label='400um_33%p 50 °C')

plt.plot(Re_400um_50perc_mid, ff_400um_50perc_mid, '.', color='yellow', label='400um_37%p 25 °C')
plt.plot(Re_400um_50perc_hi, ff_400um_50perc_hi, '.', color='cyan', label='400um_37%p 50 °C')

# plt.plot(Re_250um_45perc_hi, ff_250um_45perc_hi, '.', color='red', label='250um_45%p 45 °C')
# plt.plot(Re_250um_45perc_mid, ff_250um_45perc_mid, '.', color='DarkGreen', label='250um_45%p 24 °C')
# plt.plot(Re_250um_45perc_low, ff_250um_45perc_low, '.', color='Olive', label='250um_45%p 6 °C')
# plt.plot(Re_250um_45perc_low_flow, ff_250um_45perc_low_flow, '.', color='Violet', label='250um_45%p 6 °C Low Flow')

plt.xlabel("Modified Reynolds number [-]", fontsize=10)
plt.ylabel("Modified friction factor [-]", fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(labelsize=10)
plt.legend(fontsize=10)
# plt.show()

# Comparison of pressure drop between different geometries
plt.figure()
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(Ergun_like, np.linspace(1, 1000, 1000), *[150, 1.75]), '--r', label='Packed bed of spheres. Ergun (1952)')
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(Ergun_like, np.linspace(1, 1000, 1000), *list(fit_result_Ergun['params'].values())), '-m', label='3D printed geometry')
# plt.plot(np.linspace(1, 1000, 1000), evaluate_function(KTA_like, np.linspace(1, 1000, 1000), *[160, 3, 0.1]), '-r', label='KTA (1981)')
plt.plot(np.linspace(1, 1000, 1000), evaluate_function(a/x, np.linspace(1, 1000, 1000), *[108]), ':b', label='Parallel plate')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Modified Reynolds number [-]", fontsize=10)
plt.ylabel("Modified friction factor [-]", fontsize=10)
plt.tick_params(labelsize=10)
plt.legend(fontsize=10)

# plt.figure()
# plt.scatter(Re_htc_600um_50perc, Nu_600um_50perc, label="600um 35 % porosity")
# plt.scatter(Re_htc_600um_40perc, Nu_600um_40perc, label="600 um 29 % porosity")
# # plt.scatter(Re_htc_400um_45perc, Nu_400um_45perc, label="400 um 33 % porosity")
# # plt.scatter(Re_htc_400um_40perc, Nu_400um_40perc, label="400 um 30 % porosity")
# # plt.plot(np.linspace(1, 200, 200), evaluate_function(expr3, np.linspace(1, 200, 200), *list(fit_result_Nu['params'].values())), '-k', label='Fit')
# # plt.plot(np.linspace(1, 200, 200), evaluate_function(expr3, np.linspace(1, 200, 200), *[2, 1.1*7**(1/3), 0.6]), '-r', label='Wakao and Kaguei')
# plt.legend()
# plt.tick_params(labelsize=12)
# plt.xlabel("Reynolds number [-]", fontsize=12)
# plt.ylabel("Nusselt number [-]", fontsize=12)

# plt.figure()
# plt.scatter(Vflow_600um_50perc, h_600um_50perc, label="600um 35 % porosity")
# plt.scatter(Vflow_600um_40perc, h_600um_40perc, label="600um 29 % porosity")
# # plt.scatter(Vflow_400um_45perc, h_400um_45perc, label="400um 33 % porosity")
# # plt.scatter(Vflow_400um_40perc, h_400um_40perc, label="400um 30 % porosity")
# plt.legend()
# plt.xlabel("Flow rate [Lpm]", fontsize=12)
# plt.ylabel(r"Heat transfer coefficient [$\rm{W}/(\rm{m}^2 \rm{K})$]", fontsize=12)
# plt.tick_params(labelsize=12)
plt.show()




