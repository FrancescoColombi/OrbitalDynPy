import numpy as np
from matplotlib import pyplot as plt

from poliastro.bodies import Earth

from Functions.OrbitPropagator.R2BP import OrbitPropagatorR2BP as OP
from Functions.Utilities.PlotOrbit import plot_n_orbits


if __name__ == '__main__':
    plt.style.use('dark_background')
    R_earth = Earth.R.to('km').value

    # Orbit parameters
    altitude = 550.
    a = R_earth + altitude
    # a = 26600
    eccentricity = 0.
    incl = 98.0
    Omega = 10.0
    omega = 90.0
    theta = 0.0

    kp_leo = [a, eccentricity, incl, Omega, omega, theta]
    kp_iss = [a, eccentricity, 45, Omega, omega, theta]
    kp_geo = [42168, 0, 0, 0, 0, 0]
    kp_molnya = [26600, 0.74, 63.4, 0, 270, 0]

    t_span = np.linspace(0, 1 * 86400, 5000)

    op_leo = OP(kp_leo, t_span, coes=True, deg=True)

    op_iss = OP(kp_iss, t_span, coes=True, deg=True)

    op_geo = OP(kp_geo, t_span, coes=True, deg=True)

    op_molnya = OP(kp_molnya, t_span, coes=True, deg=True)

    fig, ax = plot_n_orbits([op_leo.rr_out, op_geo.rr_out, op_molnya.rr_out], labels=['LEO', 'GEO', 'Molnya'],
                            show_plot=False)
    # plot_n_orbits([op_leo.rr_out, op_iss.rr_out], labels=['LEO', 'ISS'])

    # plot central body
    _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, 15), np.linspace(0, np.pi, 15))
    _x = R_earth * np.cos(_u) * np.sin(_v)
    _y = R_earth * np.sin(_u) * np.sin(_v)
    _z = R_earth * np.cos(_v)
    ax.plot_surface(_x, _y, _z, cmap='Blues')
    plt.show()
