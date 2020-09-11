import numpy as np
from numpy import sin, cos
from numpy import arccos as acos


def rv2kp(rr, vv, mu, deg=True):
    """
    From state vector (position and velocity) to Keplerian Parameters

    Input:
    rr  - position vector in the geocentric equatorial frame [km]
    vv  - velocity vector in the geocentric equatorial frame [km]
    mu  - standard gravitational parameter [km^3/s^2]
    deg - bool variable. Return angles in degrees if true, in radians if false

    Output:
    kp  - Keplerian parameters [a, e, incl, RA, omega, theta]
        a - semi-major axis [km]
        e - eccentricity [-]
        incl - inclination of the orbit [deg]
        RA - right ascension of the ascending node [deg]
        omega - argument of periapsis [deg]
        TA - true anomaly [deg]

    Francesco Colombi, 2016
    """
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
    aaa = acos(np.dot(hh/h, KK))
    incl = acos(np.dot(hh/h, KK))

    # 4) Right Ascension (RA)
    # Nodal axis
    nn = np.cross(KK, hh/h)
    if np.linalg.norm(nn) == 0:     # hh parallel to KK
        nn = II
        RA = 0
    else:
        nn = nn / np.linalg.norm(nn)
        if np.dot(nn, JJ) >= 0:
            RA = acos(np.dot(nn, II))
        elif np.dot(nn, JJ) < 0:
            RA = 2*np.pi - acos(np.dot(nn, II))

    # 5) Argument of Periapsis, omega
    circular = False
    if np.round(e, 10) == 0:    # if circular orbit
        circular = True
        ee = nn         # assume eccentricity vector = nodal axis
        omega = 0
    elif incl == 0 and RA == 0:
        if np.dot(ee, JJ) >= 0:
            omega = acos(np.dot(ee / e, II))
        else:
            omega = 2 * np.pi - acos(np.dot(ee / e, II))
    elif np.dot(ee, KK) >= 0:
        omega = acos(np.dot(nn, ee/e))
    else:
        omega = 2*np.pi - acos(np.dot(nn, ee / e))

    # 6) True Anomaly, theta
    if circular:
        ee_versor = ee
    else:
        ee_versor = ee/e

    if np.dot(vv, rr) > 0:
        theta = acos(np.dot(rr/r, ee_versor))
    elif np.dot(vv, rr) < 0:
        theta = 2*np.pi - acos(np.dot(rr/r, ee_versor))
    else:
        if r < a:
            theta = 0
        else:
            theta = np.pi

    if deg:
        kp = [a, e, incl * 180 / np.pi, RA * 180 / np.pi, omega * 180 / np.pi, theta * 180 / np.pi]
    else:
        kp = [a, e, incl, RA, omega, theta]

    return kp


def kp2rv(kp, mu, deg=True):
    """
    From Keplerian Parameters to state vector (position and velocity)

    Input:
    kp  - Keplerian parameters [a, e, incl, RA, omega, TA]
          a - semi-major axis [km]
          e - eccentricity [-]
          incl - inclination of the orbit [deg]
          RA - right ascension of the ascending node [deg]
          omega - argument of periapsis [deg]
          theta - true anomaly [deg]
    mu  - standard gravitational parameter [km^3/s^2]
    deg - bool variable. Input angles in degrees if true, in radians if false

    Output:
    rr - position col vector in the geocentric equatorial frame [km]
    vv - velocity col vector in the geocentric equatorial frame [km/s]

    Author: Francesco Colombi, 2016
    """
    a = kp[0]
    e = kp[1]
    incl = kp[2]
    RA = kp[3]
    omega = kp[4]
    theta = kp[5]

    if deg:
        incl = incl * np.pi / 180
        RA = RA * np.pi / 180
        omega = omega * np.pi / 180
        theta = theta * np.pi / 180

    # 1) orbital parameters expressed in the orbital reference frame
    p = a * (1 - e**2)

    # position expressed in perifocal frame
    rr_pf = np.array([
        p * cos(theta) / (1 + e * cos(theta)),
        p * sin(theta) / (1 + e * cos(theta)),
        0
    ])

    # radial and tangential velocity
    v_r = np.sqrt(mu / p) * e * sin(theta)
    v_t = np.sqrt(mu / p) * (e * cos(theta) + 1)

    # velocity expressed in perifocal frame
    vv_pf = np.array([
        v_r * cos(theta) - v_t * sin(theta),
        v_r * sin(theta) + v_t * cos(theta),
        0
    ])

    # 2) Transformation from perifocal frame to inertial reference frame
    # Frame transformation for a rotation RA around KK
    R_Omega = np.array([
        [cos(RA), sin(RA), 0],
        [-sin(RA), cos(RA), 0],
        [0, 0, 1]
    ])
    # Frame transformation for a rotation incl around nn
    R_i = np.array([
        [1, 0, 0],
        [0, cos(incl), sin(incl)],
        [0, -sin(incl), cos(incl)]
    ])
    # Frame transformation for a rotation omega around hh
    R_omega = np.array([
        [cos(omega), sin(omega), 0],
        [-sin(omega), cos(omega), 0],
        [0, 0, 1]
    ])

    # Direction Cosine Matrix for frame transformation
    T_pf2i = np.transpose(R_omega @ R_i @ R_Omega)
    rr = np.dot(T_pf2i, rr_pf)
    vv = np.dot(T_pf2i, vv_pf)
    return rr, vv


#### TEST ####
if __name__ == '__main__':
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

