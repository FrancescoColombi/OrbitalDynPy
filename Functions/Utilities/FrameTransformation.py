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
    E = .5 * v**2 - mu/r
    a = -mu/2/E

    hh = np.cross(rr, vv)
    h = np.linalg.norm(hh)

    ee = 1 / mu * (np.cross(vv, hh) - mu * rr / r)
    e = np.linalg.norm(ee)

    # 3) Inclination [rad]
    incl = np.arccos(np.dot(hh/h, KK) / np.linalg.norm(np.dot(hh/h, KK)))

    # 4) Right Ascension (RA)
    # Nodal axis
    nn = np.cross(KK, hh/h)
    nn = nn / np.linalg.norm(nn)
    if np.dot(nn, JJ) >= 0:
        RA = np.arccos(np.dot(nn, JJ))
    elif np.dot(nn, JJ) < 0:
        RA = 2*np.pi - np.arccos(np.dot(nn, JJ))
    else:
        nn = II
        RA = 0

    # 5) Argument of Periapsis, omega
    # if circular orbit --> assume eccentricity vector = nodal axis
    if np.round(e, 15) == 0:
        ee = nn
        omega = 0
    elif incl == 0 and RA == 0:
        if np.dot(ee, JJ) >= 0:
            omega = np.arccos(ee / e, II)
        else:
            omega = 2 * np.pi - np.arccos(ee / e, II)
    elif np.dot(ee, KK) >= 0:
        omega = np.arccos(np.dot(nn, ee/e))
    else:
        omega = 2*np.pi - np.arccos(np.dot(nn, ee / e))

    # 6) True Anomaly, theta
    if np.dot(vv, rr) >= 0:
        theta = np.arccos(np.dot(rr/r, ee/e))
    else:
        theta = 2*np.pi - np.arccos(np.dot(rr / r, ee / e))

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
    T_pf2I = np.dot(R_omega, np.dot(R_i, R_Omega)).T
    rr = np.dot(T_pf2I, rr_pf)
    vv = np.dot(T_pf2I, vv_pf)
    return [rr, vv]
