import sys
# sys.path.append('/OrbitalDynPy/Utilities')

import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
#                               Array size = [n, 3]
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
def GroundTrack(tt, rr, t_0, PMST_0, omega_planet):
     # Init output vectors
    alpha = np.zeros(np.size(tt))
    delta = alpha
    latitude = alpha
    longitude = alpha

    # Initial Prime Meridian angle of the planet [rad]
    theta_0 = PMST_0 * np.pi / 12

    for n in range(np.size(tt)):
        r = np.linalg.norm(rr)

        # Compute declination [rad]
        delta[n] = np.arcsin(rr[n, 2] / r)
        # Compute Right Ascension [rad]
        if rr[n, 1] > 0:
            alpha[n] = np.arccos(rr[n, 0] / r / np.cos(delta[n]))
        else:
            alpha[n] = 2 * np.pi - np.arccos(rr[n, 0] / r / np.cos(delta[n]))

        # Compute angle of the Geodetic Frame wrt Equatorial Frame
        theta = theta_0 + omega_planet * (tt[n] - t_0)
        # Transform position vector from Equatorial to Geodetic Frame
        A_e2g = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        rr_geodetic = np.dot(A_e2g, rr[n, :])

        # Compute latitude [rad]
        latitude[n] = np.arcsin(rr_geodetic[2] / r)
        # Compute longitude [rad]
        if rr_geodetic[1] >= 0:
            longitude[n] = np.arccos(rr_geodetic[2] / r)
        else:
            longitude[n] = 2 * np.pi - np.arccos(rr_geodetic[2] / r)

    return alpha * 180 / np.pi, delta * 180 / np.pi, latitude * 180 / np.pi, longitude * 180 / np.pi


def test_GroundTrack():
    # Constants
    R_earth = 0.63781600000000E+04
    mu_earth = 0.59736990612667E+25*6.67259e-20

    # Sidereal time of Earth Rotation
    ST_earth_rot = (23 + (56 + 4.09/60)/60)/24 * 86400
    # Earth rotation rate [rad/sec]
    omega_earth = 2 * np.pi / ST_earth_rot

    # Orbit parameters
    altitude = 550.
    eccentricity = 0.3
    inclination = 0.0
    kp0 = [R_earth+altitude, eccentricity, inclination, 0.1, 0, 0]
    rr0, vv0 = kp2rv(kp0, mu_earth)

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
    tspan = [0, 2 * 3600]
    t_out = np.linspace(tspan[0], tspan[1], 1000)
    y_out = odeint(R2BP_dyn, X0, t_out, args=(mu_earth, ), rtol=1e-10, atol=1e-10)

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
    plt.show()

    # Ground Track
    t_0 = jd0 * 86400
    tt = t_0 + t_out
    [alpha, delta, lat, long] = GroundTrack(tt, rr_orb, t_0, GMST_0, omega_earth)

    fig = plt.figure()
    ax = fig.subplots(1, 1)
    ax.plot(lat, long)
    plt.show()
    return


test_GroundTrack()
