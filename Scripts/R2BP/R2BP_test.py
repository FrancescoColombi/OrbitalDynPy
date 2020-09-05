import numpy as np
import pandas as pd
from scipy.integrate import odeint

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from PIL import Image

from spiceypy import spiceypy as spice


from Functions.GroundTrack import GroundTrack
from Functions.ODE.R2BP import R2BP_dyn
from Functions.Utilities.KeplerianParameters import kp2rv
from Functions.Utilities.TimeConversion import jd2GMST

def test_GroundTrack():
    # Constants
    R_earth = 0.63781600000000E+04
    mu_earth = 0.59736990612667E+25 * 6.67259e-20

    # Sidereal time of Earth Rotation [sec]
    ST_earth_rot = (23 + (56 + 4.09 / 60) / 60) / 24 * 86400
    # Earth rotation rate [rad/sec]
    omega_earth = 2 * np.pi / ST_earth_rot

    # Orbit parameters
    altitude = 550.
    a = R_earth + altitude
    # a = 26600
    eccentricity = 0.
    incl = 98.0
    Omega = 0.0
    omega = 0.0
    theta = 0.0
    kp0 = [a, eccentricity, incl, Omega, omega, theta]
    rr0, vv0 = kp2rv(kp0, mu_earth)
    print('Keplerian parameter:     {0}'.format(kp0))
    print('Initial position:        {0} km'.format(rr0))
    print('Initial velocity:        {0} km/s'.format(vv0))
    T_orb = 2 * np.pi * np.sqrt(a**3 / mu_earth)
    print('Orbital period:          {0} h'.format(T_orb/3600))

    # Reference time
    kernel_dir = 'D:/Francesco/Spice_kernels/'
    meta_kernel = kernel_dir + 'meta_kernel.tm'
    spice.furnsh(meta_kernel)
    date0 = "2020 jan 01 12:00:00"
    jd0 = spice.utc2et(date0) / 86400
    spice.kclear()

    # Reference Prime Meridian Sidereal Time
    GMST_0 = jd2GMST(jd0)

    # Orbit propagation
    X0 = np.hstack((rr0, vv0))
    t0 = 0.
    tf = 2 * T_orb
    t_out = np.linspace(t0, tf, 5000)
    y_out = odeint(R2BP_dyn, X0, t_out, args=(mu_earth,), rtol=1e-10, atol=1e-10)

    rr_orb = y_out[:, 0:3].T
    vv_orb = y_out[:, 3:6].T

    deltat = 1*3600
    t_marker = np.arange(t0, tf, deltat)
    y_marker = odeint(R2BP_dyn, X0, t_marker, args=(mu_earth,), rtol=1e-10, atol=1e-10)
    rr_marker = y_marker[:, 0:3].T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(rr_orb[0, :], rr_orb[1, :], rr_orb[2, :])
    ax.scatter(0, 0, 0, color='blue')
    ax.scatter(rr_orb[0, 0], rr_orb[1, 0], rr_orb[2, 0], color='red')
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.set_title('Orbital propagation in Equatorial Frame')
    plt.show()

    # Ground Track
    t_0 = jd0 * 86400  # [sec]
    tt = t_0 + t_out
    alpha, delta, lat, long, _ = GroundTrack(tt, rr_orb, t_0, GMST_0, omega_earth)

    tt_mark = t_0 + t_marker
    alpha, delta, lat_mark, long_mark, _ = GroundTrack(tt_mark, rr_marker, t_0, GMST_0, omega_earth)

    # print(alpha[range(5)])
    # print(delta[range(5)])
    # print(lat[range(10)])
    # print(long[range(10)])

    fig2 = plt.figure()
    ax = fig2.subplots(1, 1)
    ax.scatter(long, lat, c=tt)
    ax.scatter(long_mark, lat_mark, color='blue')
    ax.scatter(long[0], lat[0], color='red', label='Initial condition')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    plt.grid()
    plt.legend()
    plt.show()
    return


# test_GroundTrack()

test_GroundTrack()
