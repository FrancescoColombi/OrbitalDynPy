import sys
# sys.path.append('/OrbitalDynPy/Utilities')

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from PIL import Image

from spiceypy import spiceypy as spice

from ODE.R2BP import R2BP_dyn
from Functions.Utilities.KeplerianParameters import kp2rv
from Functions.Utilities.TimeConversion import jd2GMST

'''
# This function returns the ground track over a planet surface given input time-position coordinates
#
# INPUTS            Unit        Description
# --------------    ---------   ------------------------------------------------------------
# tt                [sec]       Time past from reference time t_0 for PMST_0 estimation
#                               Array size = [n]
# rr                [km]        Position vectors in Equatorial Frame
#                               Array size = [3, n]
# t_0               [sec]       Reference time
# PMST_0            [h]         Prime Meridion Sidereal Time is the time angle of the Prime
#                               Meridian of the planet wrt the Vernal Equinox at time t_0
# omega_planet      [rad/sec]   Rotation rate of the planet (sidereal)
#
# OUTPUTS           Unit        Description
# --------------    ---------   ------------------------------------------------------------
# alpha             [deg]       Right ascension in Equatorial Frame. Array size = [n]
# delta             [deg]       Declination in Equatorial Frame. Array size = [n]
# latitude          [deg]       Latitude. Array size = [n]
# longitude         [deg]       Longitude. Array size = [n]
'''


def GroundTrack(tt, rr, t_0, PMST_0, omega_planet, rr_axis=1):
    # rr shall be used in the following as an array with shape [3, n]
    try:
        if np.shape(rr)[0] != 3 and np.shape(rr)[1] != 3:
            raise Exception('Position vector array shall have shape as [3, n] or [n, 3]')
        elif np.shape(rr)[0] != 3 and np.shape(rr)[1] == 3:  # if input rr has shape [n, 3] --> transpose it
            rr = rr.T
    except Exception as err:
        print(err)

    # Init output vectors
    alpha = np.zeros(np.size(tt))
    delta = np.zeros(np.size(tt))
    latitude = np.zeros(np.size(tt))
    longitude = np.zeros(np.size(tt))

    # Initial Prime Meridian angle of the planet [rad]
    theta_0 = PMST_0 * np.pi / 12

    for n in range(np.size(tt)):
        r = np.linalg.norm(rr[:, n])

        # Compute declination [rad]
        delta[n] = np.arcsin(rr[2, n] / r)
        # Compute Right Ascension [rad]
        if rr[1, n] >= 0:
            alpha[n] = np.arccos(rr[0, n] / r / np.cos(delta[n]))
        else:
            alpha[n] = 2 * np.pi - np.arccos(rr[0, n] / r / np.cos(delta[n]))

        # Compute angle of the Geodetic Frame wrt Equatorial Frame
        theta = theta_0 + omega_planet * (tt[n] - t_0)
        # Transform position vector from Equatorial to Geodetic Frame (pcpf: planet centered - planet fixed)
        A_equatorial2pcpf = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        rr_pcpf = np.dot(A_equatorial2pcpf, rr[:, n])

        # Compute latitude [rad]
        latitude[n] = np.arcsin(rr_pcpf[2] / r)
        # NOTE: latitude = delta

        # Compute longitude [rad]
        if rr_pcpf[1] >= 0:  # East
            longitude[n] = np.arccos(rr_pcpf[0] / r / np.cos(latitude[n]))
        else:  # West
            longitude[n] = - np.arccos(rr_pcpf[0] / r / np.cos(latitude[n]))

    alpha = alpha * 180 / np.pi
    delta = delta * 180 / np.pi
    latitude = latitude * 180 / np.pi
    longitude = longitude * 180 / np.pi
    return alpha, delta, latitude, longitude


def test_GroundTrack():
    # Constants
    R_earth = 0.63781600000000E+04
    mu_earth = 0.59736990612667E+25 * 6.67259e-20

    # Sidereal time of Earth Rotation [sec]
    ST_earth_rot = (23 + (56 + 4.09 / 60) / 60) / 24 * 86400
    # Earth rotation rate [rad/sec]
    omega_earth = 2 * np.pi / ST_earth_rot

    # Orbit parameters
    altitude = 400.
    eccentricity = 0.0
    incl = 98.0
    Omega = 10.0
    omega = 30.0
    theta = 0.0
    kp0 = [R_earth + altitude, eccentricity, incl, Omega, omega, theta]
    rr0, vv0 = kp2rv(kp0, mu_earth)
    print('Keplerian parameter:         ', kp0)
    print('Initial position [km]:       ', rr0)
    print('Initial velocity [km/sec]:   ', vv0)

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
    tspan = [0, 5 * 3600]
    t_out = np.linspace(tspan[0], tspan[1], 5000)
    y_out = odeint(R2BP_dyn, X0, t_out, args=(mu_earth,), rtol=1e-10, atol=1e-10)

    rr_orb = y_out[:, 0:3].T
    vv_orb = y_out[:, 3:6].T

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
    alpha, delta, lat, long = GroundTrack(tt, rr_orb, t_0, GMST_0, omega_earth)

    # print(alpha[range(5)])
    # print(delta[range(5)])
    print(lat[range(10)])
    print(long[range(10)])

    fig2 = plt.figure()
    ax = fig2.subplots(1, 1)
    ax.scatter(long, lat, c=tt)
    ax.scatter(long[0], lat[0], color='red', label='Initial condition')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    plt.grid()
    plt.legend()
    plt.show()
    return


test_GroundTrack()
