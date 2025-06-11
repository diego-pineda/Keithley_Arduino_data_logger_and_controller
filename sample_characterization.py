import numpy as np
import warnings


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


def equivalent_particle_diameter(vol_block, block_area):
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


def dimension_from_SEM_error(dimension, sd_dimension, calibration_error=0.005):
    """
    Compute the total uncertainty in a dimension measured from SEM images,
    combining random measurement variation and systematic calibration error.

    Parameters:
    - dimension (float): The mean measured dimension (e.g., wall thickness) in meters.
    - sd_dimension (float): The standard deviation from repeated measurements, in meters.
    - calibration_error (float): The relative calibration uncertainty of the SEM (e.g., 0.005 for ±0.5%).

    Returns:
    - float: Total combined standard uncertainty of the dimension (in meters).
    """
    return np.sqrt(sd_dimension**2 + (calibration_error * dimension)**2)


def calculate_porosity_uncertainty(d, s1, s2, L, N_L, dd, ds1, ds2, dL):
    """
    Estimate the uncertainty in the calculated porosity using central finite difference propagation.

    Parameters:
    - d, s1, s2, L : float
        Nominal values of fiber diameter, fiber spacing in X and Y, and block height (m)
    - N_L : int
        Number of layers (assumed exact, not uncertain)
    - dd, ds1, ds2, dL : float
        Standard uncertainties (SDs) for d, s1, s2, and L

    Returns:
    - d_eps : float
        Estimated standard uncertainty in porosity
    """
    # Central value

    def partial_derivative(var_name, delta):
        args_plus = {'d': d, 's1': s1, 's2': s2, 'L': L}
        args_minus = {'d': d, 's1': s1, 's2': s2, 'L': L}
        args_plus[var_name] += delta
        args_minus[var_name] -= delta
        return (calculate_porosity(**args_plus, N_L=N_L) - calculate_porosity(**args_minus, N_L=N_L)) / (2 * delta)

    d_eps = np.sqrt(
        (partial_derivative('d', dd) * dd) ** 2 +
        (partial_derivative('s1', ds1) * ds1) ** 2 +
        (partial_derivative('s2', ds2) * ds2) ** 2 +
        (partial_derivative('L', dL) * dL) ** 2
    )

    return d_eps


def calculate_specific_surface_area_uncertainty(d, s1, s2, L, N_L, dd, ds1, ds2, dL):
    """
    Estimate the uncertainty in the specific surface area (beta) using central finite difference propagation.

    Parameters:
    - d, s1, s2, L : float
        Nominal values of fiber diameter, fiber spacing in X and Y, and block height (m)
    - N_L : int
        Number of layers (assumed exact, not uncertain)
    - dd, ds1, ds2, dL : float
        Standard uncertainties (SDs) for d, s1, s2, and L

    Returns:
    - d_beta : float
        Estimated standard uncertainty in specific surface area (1/m)
    """

    # Central value

    # Central difference derivative estimation
    def partial_derivative(var_name, delta):
        args_plus = {'d': d, 's1': s1, 's2': s2, 'L': L}
        args_minus = {'d': d, 's1': s1, 's2': s2, 'L': L}
        args_plus[var_name] += delta
        args_minus[var_name] -= delta
        return (calculate_specific_surface_area(**args_plus, N_L=N_L) -
                calculate_specific_surface_area(**args_minus, N_L=N_L)) / (2 * delta)

    d_beta = np.sqrt(
        (partial_derivative('d', dd) * dd) ** 2 +
        (partial_derivative('s1', ds1) * ds1) ** 2 +
        (partial_derivative('s2', ds2) * ds2) ** 2 +
        (partial_derivative('L', dL) * dL) ** 2
    )

    return d_beta


def calculate_Dp_eq_uncertainty(porosity, sd_porosity, beta, sd_beta):
    """
    Calculate the uncertainty in the equivalent particle diameter Dp_eq = 6(1 - porosity) / beta.

    Parameters:
    - porosity : float
        Porosity (void fraction), ε
    - sd_porosity : float
        Standard uncertainty in porosity, Δε
    - beta : float
        Specific surface area, β (1/m)
    - sd_beta : float
        Standard uncertainty in β

    Returns:
    - sd_Dp_eq : float
        Uncertainty in Dp_eq (in meters)
    """

    dDp_deps = -6 / beta
    dDp_dbeta = -6 * (1 - porosity) / (beta**2)

    sd_Dp_eq = np.sqrt(
        (dDp_deps * sd_porosity)**2 +
        (dDp_dbeta * sd_beta)**2
    )

    return sd_Dp_eq


class Sample:
    def __init__(
            self,
            *,
            name,
            V_solid_archimedes,
            porosity_Arch,
            N_layers,
            N_fibers_RAF_top,
            N_fibers_RAF_bottom,
            D_fibers,
            sd_d_fiber,
            L_block,
            W_block,
            H_block,
            wall_thick_1,
            wall_thick_2,
            wall_thick_3,
            wall_thick_4,
            sd_wall_thick_1,
            sd_wall_thick_2,
            sd_wall_thick_3,
            sd_wall_thick_4,
            S1_fibers_user=None,
            S2_fibers_user=None,
            sd_S1_fibers_user=None,
            sd_S2_fibers_user=None,
            block_dimen_unc=0.000050  # uncertainty of the dimensional device

    ):
        # TODO: implement uncertainties of Archimedes measurements if needed.
        # Basic inputs
        self.name = name
        self.V_solid_archimedes = V_solid_archimedes
        self.porosity_Arch = porosity_Arch

        # Geometry inputs
        self.N_layers = N_layers
        self.N_fibers_RAF_top = N_fibers_RAF_top
        self.N_fibers_RAF_bottom = N_fibers_RAF_bottom
        self.D_fibers = D_fibers
        self.sd_d_fiber = dimension_from_SEM_error(D_fibers, sd_d_fiber, calibration_error=0.005)
        self.L_block = L_block
        self.W_block = W_block
        self.H_block = H_block

        # Wall thicknesses and uncertainties
        self.wall_thick_1 = wall_thick_1
        self.wall_thick_2 = wall_thick_2
        self.wall_thick_3 = wall_thick_3
        self.wall_thick_4 = wall_thick_4

        self.sd_wall_thick_1 = dimension_from_SEM_error(wall_thick_1, sd_wall_thick_1, calibration_error=0.005)
        self.sd_wall_thick_2 = dimension_from_SEM_error(wall_thick_2, sd_wall_thick_2, calibration_error=0.005)
        self.sd_wall_thick_3 = dimension_from_SEM_error(wall_thick_3, sd_wall_thick_3, calibration_error=0.005)
        self.sd_wall_thick_4 = dimension_from_SEM_error(wall_thick_4, sd_wall_thick_4, calibration_error=0.005)

        # Calculated flow dimensions

        self.W_flow = W_block - wall_thick_1 - wall_thick_2
        self.sd_W_flow = np.sqrt(self.sd_wall_thick_1**2 + self.sd_wall_thick_2**2 + block_dimen_unc**2)
        self.H_flow = H_block - wall_thick_3 - wall_thick_4
        self.sd_H_flow = np.sqrt(self.sd_wall_thick_3**2 + self.sd_wall_thick_4**2 + block_dimen_unc**2)

        # Fiber spacing in each direction (corrected orientation)
        S1_calc = self.W_flow / N_fibers_RAF_top
        sd_S1_calc = self.sd_W_flow / N_fibers_RAF_top
        S2_calc = self.H_flow / N_fibers_RAF_bottom
        sd_S2_calc = self.sd_H_flow / N_fibers_RAF_bottom

        if S1_fibers_user is not None:
            self.S1_fibers = S1_fibers_user
            self.sd_S1_fibers = dimension_from_SEM_error(S1_fibers_user, sd_S1_fibers_user, calibration_error=0.005)
            if abs(S1_fibers_user - S1_calc) / S1_calc > 0.05:
                warnings.warn(f"[{name}] S1_fibers differs from calculated value by more than 5%.")
        else:
            self.S1_fibers = S1_calc
            self.sd_S1_fibers = sd_S1_calc

        if S2_fibers_user is not None:
            self.S2_fibers = S2_fibers_user
            self.sd_S2_fibers = dimension_from_SEM_error(S2_fibers_user, sd_S2_fibers_user, calibration_error=0.005)
            if abs(S2_fibers_user - S2_calc) / S2_calc > 0.05:
                warnings.warn(f"[{name}] S2_fibers differs from calculated value by more than 5%.")
        else:
            self.S2_fibers = S2_calc
            self.sd_S2_fibers = sd_S2_calc

        # Cross-sectional area for flow
        self.A_cs = self.W_flow * self.H_flow
        self.sd_A_cs = np.sqrt((self.H_flow * self.sd_W_flow)**2 + (self.W_flow * self.sd_H_flow)**2)

        self.calc_porosity = calculate_porosity(D_fibers, self.S1_fibers, self.S2_fibers, L_block, N_layers)
        self.sd_calc_porosity = calculate_porosity_uncertainty(D_fibers, self.S1_fibers, self.S2_fibers, L_block, N_layers, self.sd_d_fiber, self.sd_S1_fibers, self.sd_S2_fibers, block_dimen_unc)

        self.beta = calculate_specific_surface_area(D_fibers, self.S1_fibers, self.S2_fibers, L_block, N_layers)
        self.sd_beta = calculate_specific_surface_area_uncertainty(D_fibers, self.S1_fibers, self.S2_fibers, L_block, N_layers, self.sd_d_fiber, self.sd_S1_fibers, self.sd_S2_fibers, block_dimen_unc)

        self.Dp_eq = 6 * (1 - self.calc_porosity) / self.beta
        self.sd_Dp_eq = calculate_Dp_eq_uncertainty(self.calc_porosity, self.sd_calc_porosity, self.beta, self.sd_beta)

        # Parameters not further used but calculated
        self.A_solid = self.beta * (self.W_flow * self.H_flow * L_block)
        self.sd_A_solid = self.A_solid * np.sqrt(
            (self.sd_beta / self.beta) ** 2 +
            (self.sd_W_flow / self.W_flow) ** 2 +
            (self.sd_H_flow / self.H_flow) ** 2 +
            (0.00005 / self.L_block) ** 2
        )

        self.calc_solid_volume = (1 - self.calc_porosity) * (self.W_flow * self.H_flow * L_block)

        # Note: Dp_eq could alternatively be calculated using the function equivalent_particle_diameter()
        self.Dh = 4 * self.calc_porosity / self.beta  # Hydraulic diameter
        self.porosity = self.calc_porosity  # The Archimedes porosity could also go here

