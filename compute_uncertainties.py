import numpy as np
import pandas as pd
from sample_characterization import calculate_porosity
from sample_characterization import calculate_specific_surface_area
from CoolProp.CoolProp import PropsSI
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# def compute_flow_rate_uncertainty(Q_lpm):
#     """
#     Compute the uncertainty in flow rate measurements.
#
#     Parameters:
#     Q_lpm : np.ndarray
#         Flow rate data in liters per minute (L/min).
#
#     Returns:
#     dQ_m3s : np.ndarray
#         Uncertainty in flow rate in cubic meters per second (m³/s).
#     """
#     # Compute uncertainty in L/min
#     dQ_lpm = 0.008 * Q_lpm + 0.05
#
#     # Convert to m³/s: 1 L/min = 1/60000 m³/s
#     dQ_m3s = dQ_lpm / 60000.0
#
#     return dQ_m3s


# def compute_flow_area_uncertainty(
#         W_block, H_block,
#         delta1, delta2, delta3, delta4,
#         dW_block=0.00005, dH_block=0.00005,
#         sd_delta1=0.0, sd_delta2=0.0, sd_delta3=0.0, sd_delta4=0.0,
#         calibration_error_SEM=0.005
# ):
#     """
#     Computes the uncertainty in the flow area A_flow = (W_block - delta1 - delta2)*(H_block - delta3 - delta4)
#     using error propagation.
#
#     Parameters:
#     - W_block, H_block: block dimensions (in meters)
#     - delta1 to delta4: wall thicknesses (in meters)
#     - dW_block, dH_block: uncertainties in block dimensions (default 0.05 mm)
#     - sd_deltaX: standard deviation of each wall thickness (in meters)
#     - calibration_error: relative SEM calibration error (default 0.5%)
#
#     Returns:
#     - dA_flow: propagated uncertainty in flow area (in m²)
#     """
#     # Compute effective flow width and height
#     W_flow = W_block - delta1 - delta2
#     H_flow = H_block - delta3 - delta4
#
#     # Total uncertainty for each wall thickness (random + systematic)
#
#
#     d_delta1 = dimension_from_SEM_error(delta1, sd_delta1, calibration_error_SEM)
#     d_delta2 = dimension_from_SEM_error(delta2, sd_delta2, calibration_error_SEM)
#     d_delta3 = dimension_from_SEM_error(delta3, sd_delta3, calibration_error_SEM)
#     d_delta4 = dimension_from_SEM_error(delta4, sd_delta4, calibration_error_SEM)
#
#     # Error propagation formula
#     dA_flow = np.sqrt(
#         (H_flow * dW_block)**2 +
#         (H_flow * d_delta1)**2 +
#         (H_flow * d_delta2)**2 +
#         (W_flow * dH_block)**2 +
#         (W_flow * d_delta3)**2 +
#         (W_flow * d_delta4)**2
#     )
#
#     return dA_flow




#
# def compute_sd_spacing(d_thick_1, d_thick_2, N_fibers, block_dim_unc=0.00005):
#     """
#     Compute standard deviation in fiber spacing S = (block_dim - delta1 - delta2) / N_fibers.
#
#     Parameters:
#     - block_dim_unc: uncertainty in block dimension (W_block or H_block), in meters
#     - d_thick_1: standard deviation of wall thickness 1 (in meters)
#     - d_thick_2: standard deviation of wall thickness 2 (in meters)
#     - N_fibers: number of fibers across the block dimension
#
#     Returns:
#     - sd_S: standard deviation in fiber spacing (in meters)
#     """
#
#     sd_S = (1 / N_fibers) * np.sqrt(block_dim_unc**2 + d_thick_1**2 + d_thick_2**2)
#     return sd_S

#
# def compute_sd_spacing(d_thick_1, d_thick_2, N_fibers, block_dim_unc=0.00005):
#     """
#     Compute standard deviation in fiber spacing S = (block_dim - delta1 - delta2) / N_fibers.
#
#     Parameters:
#     - block_dim_unc: uncertainty in block dimension (W_block or H_block), in meters
#     - d_thick_1: standard deviation of wall thickness 1 (in meters)
#     - d_thick_2: standard deviation of wall thickness 2 (in meters)
#     - N_fibers: number of fibers across the block dimension
#
#     Returns:
#     - sd_S: standard deviation in fiber spacing (in meters)
#     """
#
#     sd_S = (1 / N_fibers) * np.sqrt(block_dim_unc**2 + d_thick_1**2 + d_thick_2**2)
#     return sd_S


# def compute_sd_flow_dimens(d_thick_1, d_thick_2, block_dim_unc=0.00005):
#     """
#     Compute SD in a dimension of the region available for flow, i.e. W_flow = (W_block - wall_1 - wall_2)
#
#     Parameters:
#     - block_dim_unc: uncertainty in block dimension (W_block or H_block), in meters
#     - d_thick_1: standard deviation of wall thickness 1 (in meters)
#     - d_thick_2: standard deviation of wall thickness 2 (in meters)
#
#     Returns:
#     - sd_S: standard deviation in flow dimension (in meters)
#     """
#
#     sd_flow_dim =  np.sqrt(block_dim_unc**2 + d_thick_1**2 + d_thick_2**2)
#     return sd_flow_dim




def fluid_property_uncertainties_from_temperature(*, T, dT, P=101325, fluid='Water', dT_fd=0.1):
    """
    Estimate uncertainty in density and dynamic viscosity due to temperature uncertainty.

    Parameters:
    - T (float): Temperature in K
    - dT (float): Uncertainty in temperature in K (e.g., 0.15)
    - P (float): Pressure in Pa (default 101325)
    - fluid (str): Fluid name (default 'Water')
    - dT_fd (float): Step size for finite difference (default 0.1 K)

    Returns:
    - drho (float): Estimated uncertainty in density [kg/m³]
    - dmu (float): Estimated uncertainty in viscosity [Pa·s]
    """
    # Derivative of density using CoolProp
    try:
        drho_dT = PropsSI('d(D)/d(T)|P', 'T', T, 'P', P, fluid)
    except Exception:
        # Fallback to finite difference if needed
        rho_plus = PropsSI('D', 'T', T + dT_fd, 'P', P, fluid)
        rho_minus = PropsSI('D', 'T', T - dT_fd, 'P', P, fluid)
        drho_dT = (rho_plus - rho_minus) / (2 * dT_fd)

    # Derivative of viscosity via finite difference
    mu_plus = PropsSI('VISCOSITY', 'T', T + dT_fd, 'P', P, fluid)
    mu_minus = PropsSI('VISCOSITY', 'T', T - dT_fd, 'P', P, fluid)
    dmu_dT = (mu_plus - mu_minus) / (2 * dT_fd)

    drho = abs(drho_dT) * dT
    dmu = abs(dmu_dT) * dT

    return drho, dmu


def calculate_Re_uncertainty(rho, drho, mu, dmu, u, du, Dp_eq, dDp_eq, eps, deps):
    """
    Calculate the Reynolds number and its uncertainty using analytical error propagation.

    Parameters:
    - rho : float
        Fluid density (kg/m³)
    - drho : float
        Uncertainty in density (kg/m³)
    - mu : float
        Fluid viscosity (Pa·s)
    - dmu : float
        Uncertainty in viscosity (Pa·s)
    - u : float
        Superficial velocity (m/s)
    - du : float
        Uncertainty in velocity (m/s)
    - Dp_eq : float
        Equivalent particle diameter (m)
    - dDp_eq : float
        Uncertainty in equivalent diameter (m)
    - eps : float
        Porosity (–)
    - deps : float
        Uncertainty in porosity (–)

    Returns:
    - dRe : float
        Propagated uncertainty in Reynolds number
    """
    denom = mu * (1 - eps)

    dRe = np.sqrt(
        ((u * Dp_eq) / denom * drho) ** 2 +
        ((rho * Dp_eq) / denom * du) ** 2 +
        ((rho * u) / denom * dDp_eq) ** 2 +
        ((rho * u * Dp_eq) / (mu**2 * (1 - eps)) * dmu) ** 2 +
        ((rho * u * Dp_eq) / (mu * (1 - eps)**2) * deps) ** 2
    )

    return dRe


def calculate_friction_factor_uncertainty(dp_kPa, ddp_kPa, rho, drho, u, du, Dp_eq, dDp_eq, L, dL, eps, deps):
    """
    Calculate the modified friction factor and its uncertainty using analytical error propagation.

    Parameters:
    - dp_kPa : float
        Differential pressure in kPa
    - ddp_kPa : float
        Uncertainty in differential pressure in kPa
    - rho : float
        Fluid density in kg/m³
    - drho : float
        Uncertainty in density
    - u : float
        Superficial velocity in m/s
    - du : float
        Uncertainty in velocity
    - Dp_eq : float
        Equivalent particle diameter in m
    - dDp_eq : float
        Uncertainty in Dp_eq
    - L : float
        Block length in flow direction (m)
    - dL : float
        Uncertainty in L
    - eps : float
        Porosity
    - deps : float
        Uncertainty in porosity

    Returns:
    - df : float
        Propagated uncertainty in friction factor
    """
    # Convert pressure to Pa
    dp = 1000 * dp_kPa
    ddp = 1000 * ddp_kPa

    # Core components
    C = (dp * Dp_eq) / (L * rho * u**2)
    phi = (eps**3) / (1 - eps)

    # Derivatives
    dfdp = (Dp_eq / (L * rho * u**2)) * phi
    dfddp = dfdp * ddp

    dfdD = (dp / (L * rho * u**2)) * phi
    dfDd = dfdD * dDp_eq

    dfdL = -(dp * Dp_eq / (L**2 * rho * u**2)) * phi
    dfLd = dfdL * dL

    dfdrho = -(dp * Dp_eq / (L * rho**2 * u**2)) * phi
    dfrhod = dfdrho * drho

    dfdu = -2 * (dp * Dp_eq / (L * rho * u**3)) * phi
    dfud = dfdu * du

    dfdphi = C * ( (3 * eps**2 * (1 - eps) + eps**3) / (1 - eps)**2 )
    dfeps = dfdphi * deps

    # Total uncertainty
    df = np.sqrt(dfddp**2 + dfDd**2 + dfLd**2 + dfrhod**2 + dfud**2 + dfeps**2)

    return df


def calculate_dp_sensor_uncertainty(
        V_sensor,
        V_zero=0.21056,
        V_cal=0.27620,
        dp_cal=1.4765,
        dV_sensor=2.9e-5,
        dV_zero=2.9e-5,
        dV_cal=2.9e-5,
        ddp_cal=0.0098
):
    """
    Calculate the uncertainty in differential pressure derived from sensor voltage.

    Parameters:
    - V_sensor : float
        Voltage from sensor during measurement (V)
    - V_zero : float, optional
        Voltage at zero pressure (V), default: 0.21056
    - V_cal : float, optional
        Voltage during calibration test (V), default: 0.27620
    - dp_cal : float, optional
        Known calibration pressure (kPa), default: 1.4765
    - dV_sensor : float, optional
        Uncertainty in V_sensor (V), default: 29 μV
    - dV_zero : float, optional
        Uncertainty in V_zero (V), default: 29 μV
    - dV_cal : float, optional
        Uncertainty in V_cal (V), default: 29 μV
    - ddp_cal : float, optional
        Uncertainty in calibration pressure (kPa), default: 0.0098

    Returns:
    - ddp_sensor : float
        Propagated uncertainty in differential pressure (kPa)

    Notes:
    - The default uncertainty in dp_cal (0.0098 kPa) is based on a hydrostatic pressure
      head of 150 ± 1 mm, assuming ρ = 997 kg/m³ and g = 9.8 m/s²:
        ΔP = ρ·g·Δh ≈ 997 × 9.8 × 0.001 = 9.8 Pa = 0.0098 kPa
    - The influence of uncertainty in fluid density due to ±0.15 K uncertainty in water temperature
      is negligible compared to the geometric uncertainty of the height (±1 mm).
    """
    V_diff = V_sensor - V_zero
    V_cal_span = V_cal - V_zero

    # Partial derivatives
    d_dp_dVsensor = dp_cal / V_cal_span
    d_dp_dVzero = dp_cal * (V_sensor - V_cal) / V_cal_span**2
    d_dp_dVcal = -dp_cal * V_diff / V_cal_span**2
    d_dp_ddpcal = V_diff / V_cal_span

    # Total propagated uncertainty
    ddp_sensor = np.sqrt(
        (d_dp_dVsensor * dV_sensor) ** 2 +
        (d_dp_dVzero * dV_zero) ** 2 +
        (d_dp_dVcal * dV_cal) ** 2 +
        (d_dp_ddpcal * ddp_cal) ** 2
    )

    return ddp_sensor


def calculate_dp_dynamic_uncertainty(dp_sensor, ddp_sensor, rho, drho, h=0.109, dh=0.001):
    """
    Calculate the uncertainty in the dynamic differential pressure ΔP_dynamic,
    based on sensor pressure and hydrostatic correction.

    Parameters:
    - dp_sensor : float
        Calibrated differential pressure from the sensor (kPa)
    - ddp_sensor : float
        Uncertainty in dp_sensor (kPa)
    - rho : float
        Fluid density (kg/m³)
    - drho : float
        Uncertainty in fluid density (kg/m³)
    - h : float, optional
        Vertical height difference between pressure ports (m), default: 0.109
    - dh : float, optional
        Uncertainty in h (m), default: 0.001 (±1 mm)

    Returns:
    - ddp_dynamic : float
        Uncertainty in dynamic pressure drop (kPa)

    Notes:
    - Hydrostatic correction is ρ·g·h, with g = 9.8 m/s².
    """
    g = 9.8  # m/s²

    ddp_dynamic = np.sqrt(
        ddp_sensor**2 +
        ((g * h * drho) / 1000) ** 2 +
        ((rho * g * dh) / 1000) ** 2
    )

    return ddp_dynamic


def monte_carlo_fit_ff_vs_Re(
        Re, ff, Re_err, ff_err,
        N_sim=1000, Re_eval=None, conf_interval=0.95, seed=None
):
    """
    Perform Monte Carlo fitting for ff = a / Re + b, accounting for uncertainty in both Re and ff.

    Parameters:
    - Re : array-like
        Measured Reynolds numbers.
    - ff : array-like
        Measured friction factors.
    - Re_err : array-like
        Standard deviation (uncertainty) for each Reynolds number.
    - ff_err : array-like
        Standard deviation (uncertainty) for each friction factor.
    - N_sim : int
        Number of Monte Carlo simulations (default: 1000).
    - Re_eval : array-like or None
        Re values at which the model will be evaluated. If None, it uses a linspace over Re range.
    - conf_interval : float
        Confidence level for the output interval (e.g. 0.95 for 95%).
    - seed : int or None
        Random seed for reproducibility. Set to an integer for consistent results.

    Returns:
    - Re_eval : ndarray
        Re values at which the model is evaluated.
    - ff_mean : ndarray
        Mean predicted friction factor at each Re_eval point.
    - ff_lower : ndarray
        Lower bound of the confidence interval at each Re_eval.
    - ff_upper : ndarray
        Upper bound of the confidence interval at each Re_eval.
    """

    # Optional reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Convert all inputs to numpy arrays for indexing and broadcasting
    Re = np.asarray(Re)
    ff = np.asarray(ff)
    Re_err = np.asarray(Re_err)
    ff_err = np.asarray(ff_err)

    # If the user didn't specify Re_eval points, use 200 points over full range
    if Re_eval is None:
        Re_eval = np.linspace(Re.min(), Re.max(), 200)

    # To store the fitted parameters from each simulation
    a_samples = []
    b_samples = []
    residuals_all = []

    # Define the model function: ff = a / Re + b
    def model(Re, a, b):
        return a / Re + b

    # Main Monte Carlo loop
    for _ in range(N_sim):
        # For each point, generate a new synthetic value based on its uncertainty
        Re_synth = np.random.normal(Re, Re_err)  # shape: same as Re
        ff_synth = np.random.normal(ff, ff_err)  # shape: same as ff

        # --------------------------------

        if _ == 0:  # Only for the first iteration


            print("Original Re:", Re[:5])
            print("Original ff:", ff[:5])
            print("Synthetic Re (1st sim):", Re_synth[:5])
            print("Synthetic ff (1st sim):", ff_synth[:5])

            # Plot original vs synthetic data
            plt.figure(figsize=(6, 4))
            plt.errorbar(Re, ff, xerr=Re_err, yerr=ff_err, fmt='o', label='Original data')
            plt.scatter(Re_synth, ff_synth, c='red', label='Synthetic sample')
            plt.xlabel('Re')
            plt.ylabel('f')
            plt.xscale('log')
            plt.yscale('log')
            plt.title('Original vs Synthetic Data (1st Iteration)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show(block=False)
            # ---------------------------------

    # Fit model to the synthetic dataset using curve_fit
        try:
            popt, _ = curve_fit(
                model,               # function to fit
                Re_synth,            # synthetic x data
                ff_synth,            # synthetic y data
                p0=[100, 1],      # initial guess for [a, b]
                maxfev=10000         # increase max number of iterations to avoid RuntimeError
            )
            # Store the fitted parameters
            a_i = popt[0]
            b_i = popt[1]
            a_samples.append(a_i)
            b_samples.append(b_i)
        except RuntimeError:
            # curve_fit can fail if synthetic data is noisy or numerically unstable
            # we just skip such cases
            continue

        ff_model_synth = a_i / Re_synth + b_i
        residuals_all.extend(ff_synth - ff_model_synth)

    # Convert parameter samples to numpy arrays
    a_samples = np.array(a_samples)  # shape: (N_successful_fits,)
    b_samples = np.array(b_samples)
    residuals_all_std = np.std(residuals_all, ddof=1)
    # --------------------------------------
    # Compute statistics of fitted parameters
    a_mean = np.mean(a_samples)
    a_std = np.std(a_samples, ddof=1)

    b_mean = np.mean(b_samples)
    b_std = np.std(b_samples, ddof=1)

    # Print formatted result
    print("Fitted correlation (from Monte Carlo):")
    print(f"f = ({a_mean:.4f} ± {a_std:.4f}) / Re + ({b_mean:.5f} ± {b_std:.5f})")
    # -------------------------------------------

    # Evaluate the fitted model at each Re_eval point for every (a, b) pair
    # ff_all will be shape: (N_fits, len(Re_eval))
    ff_all = np.array([a / Re_eval + b for a, b in zip(a_samples, b_samples)])

    # Compute the mean fitted ff at each Re_eval point
    ff_mean = np.mean(ff_all, axis=0)

    # ------------
    # Fit model directly to ff_mean
    popt, _ = curve_fit(model, Re_eval, ff_mean)
    a_fit, b_fit = popt
    # Evaluate all curves
    ff_model_mean = a_mean / Re_eval + b_mean
    ff_model_fit = a_fit / Re_eval + b_fit

    # -----



    # Determine the bounds of the confidence interval
    alpha = 1.0 - conf_interval  # e.g., 0.05 for 95% CI
    ff_lower = np.percentile(ff_all, 100 * alpha / 2, axis=0)        # lower bound (e.g. 2.5th percentile)
    ff_upper = np.percentile(ff_all, 100 * (1 - alpha / 2), axis=0)  # upper bound (e.g. 97.5th percentile)

    # Prediction intervals

    # --- Prediction Interval Computation ---

    # Evaluate the mean model on your original data points
    ff_predicted_at_data = a_mean / Re + b_mean

    # Residuals between actual data and model prediction
    residuals = ff - ff_predicted_at_data

    # Estimate the standard deviation of those residuals
    residual_std = np.std(residuals, ddof=1)

    # Add residual noise to the confidence bounds to get prediction interval
    ff_pred_lower = ff_lower - residuals_all_std  # residual_std
    ff_pred_upper = ff_upper + residuals_all_std  # residual_std

    # Relative uncertainty of the correlation
    rel_uncertainty = (ff_pred_upper / ff_model_mean) - 1
    percent_uncertainty = rel_uncertainty * 100
    avg_rel_unc = np.mean(percent_uncertainty)
    max_rel_unc = np.max(percent_uncertainty)

    print(f"Average relative prediction uncertainty: ±{avg_rel_unc:.1f}%")
    print(f"Maximum relative prediction uncertainty: ±{max_rel_unc:.1f}%")
    plt.figure()
    plt.plot(rel_uncertainty)
    plt.show(block=False)
    # ------------------------------------------

    # Plotting parameter-based prediction bounds
    ff_upper_param_1 = (a_mean + a_std) / Re_eval + (b_mean + b_std)
    ff_upper_param_2 = (a_mean + a_std) / Re_eval + (b_mean - b_std)
    ff_lower_param_1 = (a_mean - a_std) / Re_eval + (b_mean + b_std)
    ff_lower_param_2 = (a_mean - a_std) / Re_eval + (b_mean - b_std)

    plt.figure(figsize=(8, 6))
    plt.plot(Re_eval, ff_model_mean, 'k-', label='Mean model')
    plt.plot(Re_eval, ff_upper_param_1, 'r--', label='(a+σa, b+σb)')
    plt.plot(Re_eval, ff_upper_param_2, 'g--', label='(a+σa, b−σb)')
    plt.plot(Re_eval, ff_lower_param_1, 'b--', label='(a−σa, b+σb)')
    plt.plot(Re_eval, ff_lower_param_2, 'm--', label='(a−σa, b−σb)')


    # You can also fill the area if desired:
    plt.fill_between(Re_eval,
                     np.minimum.reduce([ff_lower_param_1, ff_lower_param_2]),
                     np.maximum.reduce([ff_upper_param_1, ff_upper_param_2]),
                     color='gray', alpha=0.2, label='Param uncertainty envelope')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Re')
    plt.ylabel('f')
    plt.legend()
    plt.grid(True)
    plt.title('Parameter-Based Prediction Interval')
    plt.tight_layout()
    plt.show(block=False)


# -----------------------
    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.errorbar(Re, ff, xerr=Re_err, yerr=ff_err, fmt='o', label='Original data', alpha=0.05)
    plt.plot(Re_eval, ff_mean, label='Monte Carlo average (ff_mean)', linewidth=2)
    plt.plot(Re_eval, ff_model_mean, '--', label=f'a_mean / Re + b_mean\n(a={a_mean:.3f}, b={b_mean:.4f})')
    plt.plot(Re_eval, ff_model_fit, ':', label=f'Fit to ff_mean\n(a={a_fit:.3f}, b={b_fit:.4f})')

    plt.fill_between(Re_eval, ff_pred_lower, ff_pred_upper,
                     color='gray', alpha=0.2, label='95% prediction interval')
    # ------------------ trial of using a+sd_a/Re + (b+st_b) and a-sd_a/Re + (b-st_b) for prediction interval-----------
    plt.plot(Re_eval, ff_upper_param_1, 'r--', label='(a+σa, b+σb)')
    plt.plot(Re_eval, ff_lower_param_2, 'm--', label='(a−σa, b−σb)')
    # --------------------------------------------------------------------------------------------------
    plt.xlabel('Reynolds number')
    plt.ylabel('Friction factor')
    plt.title('Comparison of Model Curves')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    # ----------------------



    return Re_eval, ff_mean, ff_lower, ff_upper, ff_pred_lower, ff_pred_upper


def propagate_uncertainty(Re_eval, var_a, var_b, cov_ab):
    """
    Computes the propagated standard deviation of f = a / Re + b
    due to uncertainty in a and b, including their covariance.
    """
    Re_eval = np.asarray(Re_eval)
    df_da = 1 / Re_eval
    df_db = 1.0
    var_f = df_da**2 * var_a + df_db**2 * var_b + 2 * df_da * df_db * cov_ab
    return np.sqrt(var_f)


def monte_carlo_fit_ff_vs_Re_simplified(Re, ff, Re_err, ff_err, N_realizations=10, Re_eval=None, conf_interval=0.95, seed=None):
    """
    Simpler and statistically sounder version of the original function using a single pooled fit
    and covariance-based confidence bands.
    """

    if seed is not None:
        np.random.seed(seed)

    Re = np.asarray(Re)
    ff = np.asarray(ff)
    Re_err = np.asarray(Re_err)
    ff_err = np.asarray(ff_err)

    if Re_eval is None:
        Re_eval = np.logspace(np.log10(Re.min()), np.log10(Re.max()), 200)

    # --- Step 1: Create synthetic pooled dataset ---
    Re_all = []
    ff_all = []

    for _ in range(N_realizations):
        Re_synth = np.random.normal(Re, Re_err)
        ff_synth = np.random.normal(ff, ff_err)
        Re_all.extend(Re_synth)
        ff_all.extend(ff_synth)

    Re_all = np.array(Re_all)
    ff_all = np.array(ff_all)

    # --- Step 2: Fit pooled synthetic data ---
    def model(Re, a, b):
        return a / Re + b

    popt, pcov = curve_fit(model, Re_all, ff_all, p0=[100, 1], maxfev=10000)
    a_fit, b_fit = popt
    var_a = pcov[0, 0]
    var_b = pcov[1, 1]
    cov_ab = pcov[0, 1]

    print(f"Fitted model: f = ({a_fit:.3f} ± {np.sqrt(var_a):.3f}) / Re + ({b_fit:.4f} ± {np.sqrt(var_b):.4f})")

    # --- Step 3: Evaluate model and uncertainty ---
    ff_pred = a_fit / Re_eval + b_fit
    sigma_f = propagate_uncertainty(Re_eval, var_a, var_b, cov_ab)
    z = 1.96 if conf_interval == 0.95 else 2.576 if conf_interval == 0.99 else 1.0  # adjust if needed

    ff_upper = ff_pred + z * sigma_f
    ff_lower = ff_pred - z * sigma_f

    # --- Step 4: Plot ---
    plt.figure(figsize=(8, 5))
    plt.errorbar(Re, ff, xerr=Re_err, yerr=ff_err, fmt='o', label='Original data', alpha=0.3)
    plt.plot(Re_eval, ff_pred, 'k-', label='Fitted model')
    plt.fill_between(Re_eval, ff_lower, ff_upper, color='gray', alpha=0.3, label=f'{int(conf_interval*100)}% Confidence band')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Reynolds number')
    plt.ylabel('Friction factor')
    plt.title('Model Fit with Confidence Band')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    return Re_eval, ff_pred, ff_lower, ff_upper

# --------------------------------------


def generate_synthetic_data(Re, ff, Re_err, ff_err, n_realizations=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    Re_all_synth, ff_all_synth = [], []
    for _ in range(n_realizations):
        Re_synth = np.random.normal(Re, Re_err)
        ff_synth = np.random.normal(ff, ff_err)
        Re_all_synth.extend(Re_synth)
        ff_all_synth.extend(ff_synth)
    return np.array(Re_all_synth), np.array(ff_all_synth)


def generate_balanced_synthetic_data(Re, ff, Re_err, ff_err, n_bins=20, N_target=1000, seed=None):
    """
    Generates synthetic data such that each Re bin ends up with ~N_target points,
    by oversampling sparse bins and undersampling dense bins.
    """
    if seed is not None:
        np.random.seed(seed)

    Re = np.asarray(Re)
    ff = np.asarray(ff)
    Re_err = np.asarray(Re_err)
    ff_err = np.asarray(ff_err)

    bins = np.linspace(np.min(Re), np.max(Re), n_bins + 1)
    bin_indices = np.digitize(Re, bins) - 1

    Re_synth_all = []
    ff_synth_all = []

    for i in range(n_bins):
        mask = bin_indices == i
        Re_bin = Re[mask]
        ff_bin = ff[mask]
        Re_err_bin = Re_err[mask]
        ff_err_bin = ff_err[mask]

        n_points = len(Re_bin)
        if n_points == 0:
            continue

        reps_per_point = int(np.ceil(N_target / n_points))

        for j in range(n_points):
            Re_synth = np.random.normal(loc=Re_bin[j], scale=Re_err_bin[j], size=reps_per_point)
            ff_synth = np.random.normal(loc=ff_bin[j], scale=ff_err_bin[j], size=reps_per_point)
            Re_synth_all.extend(Re_synth)
            ff_synth_all.extend(ff_synth)

    return np.array(Re_synth_all), np.array(ff_synth_all)


def model_func(Re, a, b):
    return a / Re + b


def find_bounds_via_grid(Re_synth, ff_synth, a, b, coverage=0.5, n_steps=10):
    best_a_bound, best_b_bound, min_area = None, None, np.inf
    a_bounds = np.linspace(0.05, 0.5 * abs(a), n_steps)
    b_bounds = np.linspace(0.05, 0.5 * abs(b), n_steps)

    for da in a_bounds:
        for db in b_bounds:
            ff_hi = (a + da) / Re_synth + (b + db)
            ff_lo = (a - da) / Re_synth + (b - db)
            within = (ff_synth >= ff_lo) & (ff_synth <= ff_hi)
            # print(within)
            if np.mean(within) >= coverage:
                area = da * db
                if area < min_area:
                    best_a_bound, best_b_bound, min_area = da, db, area
    return best_a_bound, best_b_bound


def find_bounds_via_grid_per_bin(Re_synth, ff_synth, a, b, coverage=0.95, n_steps=20, n_bins=40):
    """
    Searches for (a_bound, b_bound) such that in each Re bin, at least `coverage` percent of
    synthetic data points fall within the prediction envelope defined by:
    ff = (a ± a_bound)/Re + (b ± b_bound)
    """
    Re_synth = np.asarray(Re_synth)
    ff_synth = np.asarray(ff_synth)

    # Create linear Re bins
    bins = np.linspace(np.min(Re_synth), np.max(Re_synth), n_bins + 1)
    bin_indices = np.digitize(Re_synth, bins) - 1
    fracs_best = []
    best_a_bound, best_b_bound, min_area = None, None, np.inf
    a_bounds = np.linspace(0.05, 0.95 * abs(a), n_steps)
    b_bounds = np.linspace(0.05, 0.95 * abs(b), n_steps)

    for da in a_bounds:
        for db in b_bounds:
            ff_hi = (a + da) / Re_synth + (b + db)
            ff_lo = (a - da) / Re_synth + (b - db)
            fracs = []
            all_bins_ok = True
            for i in range(n_bins):
                mask = bin_indices == i
                if np.sum(mask) == 0:
                    continue
                in_bounds = (ff_synth[mask] >= ff_lo[mask]) & (ff_synth[mask] <= ff_hi[mask])
                frac = np.sum(in_bounds) / np.sum(mask)
                fracs.append(frac)
                if frac < coverage:
                    all_bins_ok = False
                    break

            if all_bins_ok:
                area = da + db
                if area < min_area:
                    best_a_bound, best_b_bound = da, db
                    min_area = area
                    fracs_best = fracs
    print(f"Fractions are: \n {fracs_best}")
    return best_a_bound, best_b_bound


def stratified_sampling_by_Re_auto_linear(Re, ff, n_bins=10, seed=None):
    """
    Stratified sampling of synthetic data across linearly spaced Re bins,
    keeping the number of points in each bin equal to the size of the smallest bin.
    """
    if seed is not None:
        np.random.seed(seed)

    Re = np.asarray(Re)
    ff = np.asarray(ff)
    print(f"The size of the synthetic data set is {len(Re)}")
    bins = np.linspace(np.min(Re), np.max(Re), n_bins + 1)
    bin_indices = np.digitize(Re, bins) - 1  # bin index for each point

    # Count points per bin
    counts = np.array([np.sum(bin_indices == i) for i in range(n_bins)])
    print(f"Counts per bin are \n {counts}")
    min_count = np.min(counts[counts > 0])  # Exclude empty bins
    print(f"The minimum number of points in a bin is {min_count}")

    # Resample each bin
    Re_balanced, ff_balanced = [], []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            continue
        idx = np.random.choice(np.where(mask)[0], size=min_count, replace=False)
        Re_balanced.extend(Re[idx])
        ff_balanced.extend(ff[idx])

    return np.array(Re_balanced), np.array(ff_balanced)


def fit_and_predict_with_bounds(Re, ff, Re_err, ff_err, Re_eval=None, n_realizations=10, coverage=0.95, n_bins=40, seed=None, N_target=1000):
    if Re_eval is None:
        Re_eval = np.linspace(np.min(Re), np.max(Re), 200)

    # Generate synthetic data
    # Re_unbal, ff_unbal = generate_synthetic_data(Re, ff, Re_err, ff_err, n_realizations=n_realizations)
    # Re_all, ff_all = stratified_sampling_by_Re_auto_linear(Re_unbal, ff_unbal, n_bins=12)
    Re_all, ff_all = generate_balanced_synthetic_data(Re, ff, Re_err, ff_err, n_bins=n_bins, N_target=N_target, seed=seed)

    # Fit the model
    popt, _ = curve_fit(model_func, Re_all, ff_all, p0=[100, 1], maxfev=10000)
    a_fit, b_fit = popt

    # Find envelope bounds
    # a_bound, b_bound = find_bounds_via_grid(Re_all, ff_all, a_fit, b_fit, coverage=coverage)
    a_bound, b_bound = find_bounds_via_grid_per_bin(Re_all, ff_all, a_fit, b_fit, coverage=coverage, n_bins=n_bins)

    # Generate predictions
    ff_pred = model_func(Re_eval, a_fit, b_fit)
    ff_upper = model_func(Re_eval, a_fit + a_bound, b_fit + b_bound)
    ff_lower = model_func(Re_eval, a_fit - a_bound, b_fit - b_bound)

    # Plot results
    plt.figure(figsize=(6, 5))
    plt.scatter(Re, ff, marker='.', zorder=2, color='#1f77b4', alpha=0.5)
    plt.errorbar(Re, ff, xerr=Re_err, yerr=ff_err, fmt='o', color='#1f77b4', label='Original data', alpha=0.05, zorder=2)
    # plt.plot(Re, ff, '.')
    # Subsample 10% of synthetic points
    n_total = len(Re_all)
    n_sample = int(0.1 * n_total)  # This is to make the plot of synthetic points lighter
    np.random.seed(42)  # or any number
    sample_indices = np.random.choice(n_total, size=n_sample, replace=False)

    plt.plot(Re_all[sample_indices], ff_all[sample_indices], linestyle='none', marker='.', color='r', alpha=0.05, label="Synthetic data", zorder=1)
    plt.plot(Re_eval, ff_pred, 'k-', linewidth=1, label='Fitted model', zorder=3)
    plt.fill_between(Re_eval, ff_lower, ff_upper, color='gray', alpha=0.4,
                     label=f'{int(coverage * 100)}% Prediction envelope', zorder=1)
    # Plot the boundary lines
    plt.plot(Re_eval, ff_lower, 'k--', linewidth=1, zorder=3)  # dashed black line
    plt.plot(Re_eval, ff_upper, 'k--', linewidth=1, zorder=3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(15, 1100)
    plt.ylim(0.1, 20)
    plt.xlabel("Modified Reynolds number [-]", fontsize=12)
    plt.ylabel("Modified friction factor [-]", fontsize=12)
    plt.tick_params(labelsize=12)
    # plt.title('Envelope Fit to Synthetic Data Cloud')
    # plt.grid(True, which='both', ls='--')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show(block=False)

    return a_fit, b_fit, a_bound, b_bound, Re_eval, ff_pred, ff_lower, ff_upper


def archimedes_uncertainty(
        m_d, m_s, rho_f, V_total,
        u_m=0.0003,       # uncertainty in masses (g)
        u_rho_f=0.001,    # uncertainty in fluid density (g/cm^3)
        u_V_total=0.01    # uncertainty in total volume (cm^3)
):
    """
    Compute the uncertainty in V_solid and void fraction phi using Archimedes' method.

    Parameters:
    m_d : float        # dry mass in grams
    m_s : float        # submerged mass in grams
    rho_f : float      # fluid density in g/cm^3
    V_total : float    # total volume of sample in cm^3
    u_m : float        # uncertainty in m_d and m_s (assumed equal), in grams
    u_rho_f : float    # uncertainty in fluid density, in g/cm^3
    u_V_total : float  # uncertainty in total volume, in cm^3

    Returns:
    V_solid : float
    u_V_solid : float
    phi : float
    u_phi : float
    """

    # Compute solid volume
    delta_m = m_d - m_s
    V_solid = delta_m / rho_f

    # Partial derivatives for uncertainty propagation
    dVdm = 1 / rho_f
    dVdrho = -delta_m / (rho_f ** 2)

    # Uncertainty in V_solid
    u_V_solid = np.sqrt(
        (dVdm * u_m) ** 2 * 2 +  # both m_d and m_s have same u_m
        (dVdrho * u_rho_f) ** 2
    )

    # Void fraction
    phi = 1 - V_solid / V_total

    # Uncertainty in phi
    dphidVsolid = -1 / V_total
    dphidVtotal = V_solid / (V_total ** 2)

    u_phi = np.sqrt(
        (dphidVsolid * u_V_solid) ** 2 +
        (dphidVtotal * u_V_total) ** 2
    )

    return V_solid, u_V_solid, phi, u_phi


def calculate_void_fractions(
        m_d_list, m_s_list,
        V_total_list, u_V_total_list,
        T_C, u_m=0.00015, u_rho=0.0012
):
    """
    Calculate V_solid, its uncertainty, void fraction and its uncertainty
    for a list of samples using Archimedes' method.
    """
    rho_air = 0.0012  # g/cm3
    # Convert temperature to Kelvin
    T_K = T_C + 273.15

    # Get ethanol density in kg/m³, then convert to g/cm³
    rho_f = PropsSI('D', 'T', T_K, 'P', 101325, 'Ethanol') / 1000  # g/cm³

    results = []

    for m_d, m_s, V_total, u_V_total in zip(m_d_list, m_s_list, V_total_list, u_V_total_list):

        delta_m = m_d - m_s

        rho_solid_air_correction = (m_d / delta_m) * (rho_f - rho_air) + rho_air
        V_solid_air_correction = m_d / rho_solid_air_correction

        V_solid = delta_m / rho_f


        # Uncertainty in V_solid
        dVdm = 1 / rho_f
        dVdrho = -delta_m / (rho_f ** 2)

        u_V_solid = np.sqrt(
            2 * (dVdm * u_m) ** 2 +  # same balance used
            (dVdrho * u_rho) ** 2
        )

        # Void fraction
        phi = 1 - V_solid / V_total
        phi_corrected = 1 - V_solid_air_correction / V_total

        # Uncertainty in void fraction
        dphidVsolid = -1 / V_total
        dphidVtotal = V_solid / (V_total ** 2)

        u_phi = np.sqrt(
            (dphidVsolid * u_V_solid) ** 2 +
            (dphidVtotal * u_V_total) ** 2
        )

        # --- Solid density and uncertainty ---
        rho_solid = (m_d * rho_f) / delta_m

        drhod_md = (rho_f * m_s) / (delta_m ** 2)
        drhod_ms = (rho_f * m_d) / (delta_m ** 2)
        drhod_rhof = m_d / delta_m

        u_rho_solid = np.sqrt(
            (drhod_md * u_m) ** 2 +
            (drhod_ms * u_m) ** 2 +
            (drhod_rhof * u_rho) ** 2
        )


        results.append({
            'm_d (g)': m_d,
            'm_s (g)': m_s,
            'V_total (cm³)': V_total,
            'V_solid (cm³)': V_solid,
            'u_V_solid (cm³)': u_V_solid,
            'Void Fraction': phi,
            'u_void_fraction': u_phi,
            'V_solid_air_correction': V_solid_air_correction,
            'Void_fraction_corrected': phi_corrected,
            'rho_solid': rho_solid,
            'u_rho_solid': u_rho_solid
        })

    return pd.DataFrame(results)


def volume_and_uncertainty(L, W, H, u_dim=0.00005):
    """
    Calculate volume and its uncertainty from block dimensions in meters.

    Returns volume in cm³ and uncertainty in cm³.
    """
    V_m3 = L * W * H
    u_V_m3 = u_dim * ((W * H)**2 + (L * H)**2 + (L * W)**2)**0.5

    # Convert to cm³
    V_cm3 = V_m3 * 1e6
    u_V_cm3 = u_V_m3 * 1e6

    return V_cm3, u_V_cm3


if __name__ == '__main__':
    m_d_list = [6.2162, 5.7889, 5.5416, 5.9733, 5.8371, 5.5311]  # dry masses
    m_s_list = [5.2764, 4.9843, 4.7909, 5.1197, 5.0084, 4.7469]  # submerged masses

    L, W, H = 0.0161, 0.0161, 0.0062  # meters

    V_cm3, u_V_cm3 = volume_and_uncertainty(L, W, H)
    print(f"Volume = {V_cm3:.4f} ± {u_V_cm3:.4f} cm³")


    V_total_list = [V_cm3] * 6          # external volumes (cm³)
    u_V_total_list = [u_V_cm3] * 6                                  # uncertainty in volumes (cm³)
    T_C = 25  # Temperature in Celsius

    df = calculate_void_fractions(m_d_list, m_s_list, V_total_list, u_V_total_list, T_C)

    # Set max columns to None (no limit)
    pd.set_option('display.max_columns', None)
    print(df)
