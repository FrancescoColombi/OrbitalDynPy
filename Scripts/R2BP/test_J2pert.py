import numpy as np
from matplotlib import pyplot as plt

from Functions.OrbitPropagator.R2BP import OrbitPropagatorR2BP as OP
from Functions.OrbitPropagator.R2BP import null_perturbation
import Functions.Utilities.SolarSystemBodies as CelBody

if __name__ == '__main__':
    plt.style.use('dark_background')
    R_earth = CelBody.Earth["Radius"]

    # Orbit parameters
    altitude = 550.
    a = R_earth + altitude
    # a = 26600
    eccentricity = 0.
    incl = 98.0
    Omega = 10.0
    omega = 20.0
    theta = 0.0

    kp_leo = [a, eccentricity, incl, Omega, omega, theta]
    kp_iss = [a, eccentricity, 45, Omega, omega, theta]
    kp_geo = [42168, 0, 0, 0, 0, 0]
    kp_molnya = [26600, 0.74, 63.4, 0, 270, 0]

    t_span = np.linspace(0, 10 * 86400, 10000)

    perturbations = null_perturbation()
    perturbations["J2"] = True
    op_leo = OP(kp_molnya, t_span, coes=True, deg=True, perts=perturbations)

    op_leo.plot_3D(show_plot=True)
    op_leo.kp_evolution()
    op_leo.plot_kp(days=True, show_plot=True)
