# Python libs
import numpy as np
from matplotlib import pyplot as plt
from numpy import sin, cos, tan, sinh, cosh, tanh, sqrt, pi
from numpy import arccos as acos

from src.OrbitPropagator.R2BP import OrbitPropagatorR2BP, null_perturbation
from src.Utilities.KeplerianParameters import kp2rv
# Project libs
from src.Utilities.spice_tools import *
from src.Utilities.SolarSystemBodies import earth


def shadow(rr_sc_vect, et, primary_body):

    # this function requires rr_sc to be an array with shape [3, n]
    try:
        if np.shape(rr_sc_vect)[0] != 3 and np.shape(rr_sc_vect)[1] != 3:
            raise Exception('Position array shall have shape equal to [3, n] or [n, 3]')
        elif np.shape(rr_sc_vect)[0] != 3 and np.shape(rr_sc_vect)[1] == 3:  # if input rr has shape [n, 3] --> transpose it
            rr_sc_vect = np.transpose(rr_sc_vect)
    except Exception as err:
        print(err)

    if np.shape(rr_sc_vect)[1] != len(et):
        raise Exception('Position array and epoch array dimension mismatch')

    shadow_vect = np.ones([len(et)])

    rr_sun_list = get_ephem_position("SUN", et, primary_body['spice_name'], ref_frame=primary_body['body_fixed_frame'], correction='NONE')
    rr_sun_list = rr_sun_list[0]

    for id in range(len(et)):
        rr_sun = rr_sun_list[id]
        rr_sc = rr_sc_vect[:, id]
        r_sun = np.linalg.norm(rr_sun)
        r_sc = np.linalg.norm(rr_sc)
        beta = acos(np.dot(rr_sun, rr_sc) / r_sun / r_sc)

        if (beta > pi / 2) and (r_sc * sin(beta) > primary_body['radius']):
            shadow_vect[id] = 0.

    return shadow_vect


if __name__ == '__main__':
    R_earth = earth['radius']
    mu_earth = earth['mu']

    load_spice_kernel()
    date0 = "2025 jan 14 15:22:40"
    jd0 = spice.utc2et(date0) / 86400


    # Orbit parameters
    altitude = 600.0
    a = R_earth + altitude
    # a = 26600
    eccentricity = 0.0
    # incl = inclination_sunsynch(a, eccentricity)
    incl = 5
    Omega = 10.0
    omega = 200.0
    theta = 150.0

    """
    # Molniya orbital elements
    Omega = 30.0
    incl = 63.4
    omega = 270.0
    a = 26600.0
    eccentricity = 0.7
    """

    kp0 = [a, eccentricity, incl, Omega, omega, theta]
    rr0, vv0 = kp2rv(kp0, mu_earth)
    print('Keplerian parameter:     {0}'.format(kp0))
    print('Initial position:        {0} km'.format(rr0))
    print('Initial velocity:        {0} km/s'.format(vv0))
    T_orb = 2 * np.pi * np.sqrt(a ** 3 / mu_earth)
    print('Orbital period:          {0} min'.format(T_orb / 60))
    print('Pericenter:              {0} km'.format(a * (1 - eccentricity)))
    print('Apocenter:               {0} km'.format(a * (1 + eccentricity)))

    # Orbit propagation
    X0 = np.hstack((rr0, vv0))
    t0 = 0
    # tf = 20 * T_orb
    tf = 86400
    t_out = np.arange(t0, tf, 30)
    perturbations = null_perturbation()
    perturbations["J2"] = True
    orbit = OrbitPropagatorR2BP(X0, t_out, earth, perts=perturbations)
    rr_orb = orbit.rr_out
    vv_orb = orbit.vv_out
    orbit.plot_3D(show_plot=False)
    orbit.plot_kp(show_plot=True)

    et = jd0 + t_out/86400.

    shadow_vect_prova = shadow(rr_orb, et, earth)

    fig0 = plt.figure()
    ax0 = fig0.add_subplot()
    ax0.plot(et, shadow_vect_prova, markersize=1.5)
    ax0.set_title('In light time vs in shadow time')
    ax0.set_ylim(0, 1)
    ax0.grid(True)
    plt.show()

    close_spice()
