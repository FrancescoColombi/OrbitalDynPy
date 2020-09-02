
# For testing
from scipy.integrate import odeint
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from Functions.ODE.R2BP import R2BP_dyn

import numpy as np


'''
# From state vector (position and velocity) to Keplerian Parameters
#
# Input:
# rr - position vector in the geocentric equatorial frame [km]
# vv - velocity vector in the geocentric equatorial frame [km]
# mu - standard gravitational parameter [km^3/s^2]
#
# Output:
# kp - Keplerian parameters [a, e, incl, RA, omega, TA]
#       a - semi-major axis [km]
#       e - eccentricity [-]
#       incl - inclination of the orbit [deg]
#       RA - right ascension of the ascending node [deg]
#       omega - argument of periapsis [deg]
#       TA - true anomaly [deg]
#
# Author: Francesco Colombi, 2016
'''
def rv2kp(rr, vv, mu):
    # Define reference frame versors
    II = np.array([1, 0, 0])
    JJ = np.array([0, 1, 0])
    KK = np.array([0, 0, 1])

    # 1) r and v magnitude
    r = np.linalg.norm(rr)
    v = np.linalg.norm(vv)

    # 2) compute a [km] and e [-]
    E = 0.5 * v**2 - mu/r
    a = - mu/2/E

    hh = np.cross(rr, vv)
    h = np.linalg.norm(hh)

    ee = 1 / mu * (np.cross(vv, hh) - mu * rr / r)
    e = np.linalg.norm(ee)

    # 3) Inclination [rad]
    aaa = np.arccos(np.dot(hh/h, KK))
    incl = np.arccos(np.dot(hh/h, KK))

    # 4) Right Ascension (RA)
    # Nodal axis
    nn = np.cross(KK, hh/h)
    if np.linalg.norm(nn) == 0:     # hh parallel to KK
        nn = II
        RA = 0
    else:
        nn = nn / np.linalg.norm(nn)
        if np.dot(nn, JJ) >= 0:
            RA = np.arccos(np.dot(nn, II))
        elif np.dot(nn, JJ) < 0:
            RA = 2*np.pi - np.arccos(np.dot(nn, II))

    # 5) Argument of Periapsis, omega
    circular = False
    if np.round(e, 10) == 0:    # if circular orbit
        circular = True
        ee = nn         # assume eccentricity vector = nodal axis
        omega = 0
    elif incl == 0 and RA == 0:
        if np.dot(ee, JJ) >= 0:
            omega = np.arccos(np.dot(ee / e, II))
        else:
            omega = 2 * np.pi - np.arccos(np.dot(ee / e, II))
    elif np.dot(ee, KK) >= 0:
        omega = np.arccos(np.dot(nn, ee/e))
    else:
        omega = 2*np.pi - np.arccos(np.dot(nn, ee / e))

    # 6) True Anomaly, theta
    if circular:
        ee_versor = ee
    else:
        ee_versor = ee/e

    if np.dot(vv, rr) > 0:
        theta = np.arccos(np.dot(rr/r, ee_versor))
    elif np.dot(vv, rr) < 0:
        theta = 2*np.pi - np.arccos(np.dot(rr/r, ee_versor))
    else:
        if r < a:
            theta = 0
        else:
            theta = np.pi

    return [a, e, incl*180/np.pi, RA*180/np.pi, omega*180/np.pi, theta*180/np.pi]


'''
# From Keplerian Parameters to state vector (position and velocity)
#
Input:
% kp - Keplerian parameters [a, e, incl, RA, omega, TA]
%       a - semi-major axis [km]
%       e - eccentricity [-]
%       incl - inclination of the orbit [deg]
%       RA - right ascension of the ascending node [deg]
%       omega - argument of periapsis [deg]
%       TA - true anomaly [deg]
% mu - standard gravitational parameter [km^3/s^2]
%
% Output:
% rr - position col vector in the geocentric equatorial frame [km]
% vv - velocity col vector in the geocentric equatorial frame [km/s]
#
# Author: Francesco Colombi, 2016
'''
def kp2rv(kp, mu):
    a = kp[0]
    e = kp[1]
    incl = kp[2] * np.pi / 180
    RA = kp[3] * np.pi / 180
    omega = kp[4] * np.pi / 180
    theta = kp[5] * np.pi / 180

    # 1) orbital parameters expressed in the orbital reference frame
    p = a * (1 - e**2)

    # position expressed in perifocal frame
    rr_pf = np.array([
        p * np.cos(theta) / (1 + e * np.cos(theta)),
        p * np.sin(theta) / (1 + e * np.cos(theta)),
        0
    ])

    # radial and tangential velocity
    v_r = np.sqrt(mu / p) * e * np.sin(theta)
    v_t = np.sqrt(mu / p) * (e * np.cos(theta) + 1)

    # velocity expressed in perifocal frame
    vv_pf = np.array([
        v_r*np.cos(theta) - v_t*np.sin(theta),
        v_r*np.sin(theta) + v_t*np.cos(theta),
        0
    ])

    # 2) Transformation from perifocal frame to inertial reference frame
    # Frame transformation for a rotation RA around KK
    R_Omega = np.array([
        [np.cos(RA), np.sin(RA), 0],
        [-np.sin(RA), np.cos(RA), 0],
        [0, 0, 1]
    ])
    # Frame transformation for a rotation incl around nn
    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(incl), np.sin(incl)],
        [0, -np.sin(incl), np.cos(incl)]
    ])
    # Frame transformation for a rotation omega around hh
    R_omega = np.array([
        [np.cos(omega), np.sin(omega), 0],
        [-np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])

    # Direction Cosine Matrix for frame transformation
    T_i2pf = R_omega @ R_i @ R_Omega
    rr = np.dot(T_i2pf.T, rr_pf)
    vv = np.dot(T_i2pf.T, vv_pf)
    return rr, vv


#### TEST ####
def KepPar_Test():
    # Constants
    R_earth = 0.63781600000000E+04                  # [km]
    mu_earth = 0.59736990612667E+25*6.67259e-20     # [km3/sec2]

    # Orbit parameters
    altitude = 550.
    eccentricity = 0.0
    incl = 20.0
    Omega = 30.0
    omega = 15.0
    theta = 200.0
    kp0 = [R_earth+altitude, eccentricity, incl, Omega, omega, theta]
    print('Keplerian parameters: ', kp0)
    rr0, vv0 = kp2rv(kp0, mu_earth)
    print('Position [km]:       ', rr0)
    print('Velocity [km/sec]:   ', vv0)
    kp0_v2 = rv2kp(rr0, vv0, mu_earth)
    print('Keplerian parameters regression: ', kp0_v2)


def test_orbit():
    # Constants
    R_earth = 0.63781600000000E+04
    mu_earth = 0.59736990612667E+25*6.67259e-20

    # Sidereal time of Earth Rotation
    ST_earth_rot = (23 + (56 + 4.09/60)/60)/24 * 86400
    # Earth rotation rate [rad/sec]
    omega_earth = 2 * np.pi / ST_earth_rot

    # Orbit parameters
    altitude = 550.
    eccentricity = 0.0
    inclination = 90.0
    Omega = 0.0
    omega = 0.0
    theta = 0.0
    kp0 = [R_earth+altitude, eccentricity, inclination, Omega, omega, theta]
    print(kp0)
    rr0, vv0 = kp2rv(kp0, mu_earth)
    print(rr0)
    print(vv0)
    kp0_v2 = rv2kp(rr0, vv0, mu_earth)
    print(kp0_v2)

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
    return


# test_orbit()
