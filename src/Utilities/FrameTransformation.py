import numpy as np
from numpy import sin, cos
from numpy import arcsin as asin
from numpy import arccos as acos
from numpy.linalg import norm


def spherical2cartesian(alpha, delta, r_norm):
    """
    Transform a set of position from the spherical representation (alpha, delta, r)
    to the cartesian one (xyz, 3D vectors)

    :param alpha:   Array (n) of the right ascension of the positions
    :param delta:   Array (n) of the declination of the positions
    :param r_norm:  Array (n) of the radial distance of the positions
    :return:        xyz
    """
    xyz = np.zeros([len(r_norm), 3])

    for n in range(len(r_norm)):
        xyz[n, 0] = r_norm[n] * cos(alpha[n]) * sin(delta[n])
        xyz[n, 1] = r_norm[n] * sin(alpha[n]) * sin(delta[n])
        xyz[n, 2] = r_norm[n] * cos(delta[n])

    return xyz


def cartesian2spherical(xyz):
    """
    Transform a set of position from the cartesian representation (xyz, 3D vectors)
    to the spherical one (alpha, delta, r)

    :param xyz:     Array (n, 3) of the position to be transformed
    :return:        alpha, delta, r_norm
    """
    r_norm = np.zeros(np.shape(xyz)[0])
    alpha = np.zeros(np.shape(xyz)[0])
    delta = np.zeros(np.shape(xyz)[0])

    for n in range(len(r_norm)):
        r_norm[n] = norm(xyz[n, :])

        # Compute latitude [rad]
        delta[n] = asin(xyz[n, 2] / r_norm[n])
        # NOTE: latitude = delta

        # Compute longitude [rad]
        if xyz[n, 1] >= 0:  # East
            alpha[n] = acos(xyz[n, 0] / r_norm[n] / cos(delta[n]))
        else:  # West
            alpha[n] = - acos(xyz[n, 0] / r_norm[n] / cos(delta[n]))

        return alpha, delta, r_norm


def T_Rz(theta):
    """
    Frame transformation matrix for a rotation around the z-axis

    :param theta: scalar, rotation angle [0, 2*pi]
    :return: TRz, frame transformation matrix
    """
    TRz = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float64)
    return TRz


def T_Rx(theta):
    """
    Frame transformation matrix for a rotation around the x-axis

    :param theta: scalar, rotation angle [0, 2*pi]
    :return: TRx, frame transformation matrix
    """
    TRx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), np.sin(theta)],
        [0, -np.sin(theta), np.cos(theta)]
    ], dtype=np.float64)
    return TRx


def T_Ry(theta):
    """
    Frame transformation matrix for a rotation around the y-axis

    :param theta: scalar, rotation angle [0, 2*pi]
    :return: TRx, frame transformation matrix
    """
    TRy = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)]
    ], dtype=np.float64)
    return TRy


def eq2ecef(rr, tspan, t_0, PMST_0, omega_planet):
    """
    Transformation from Equatorial Inertial reference frame (eq) to Planet-Centered Planet-Fixed frame (ecef)
    given as input the initial orientation of the generic planet (PMST_0) at a given epoch (t_0)
    and assuming a constant rotational rate of the planet (omega_planet).

    Return the transformed vectors (array rr_ecef(n, 3))
    and the DCM (list of n C_eq2ecef(3, 3)) at each time of tspan (array (n))

    :param rr:              Set of position vectors in the Equatorial frame
    :param tspan:           Array of the time corresponding to each position of the array rr
    :param t_0:             Reference epoch
    :param PMST_0:          Sidereal Time (ST) hour angle of the Prime Meridian (PM) at reference epoch
    :param omega_planet:    Rotational rate of the planet

    :return: rr_ecef, C_eq2ecef
    """
    # init output
    rr_ecef = np.zeros([len(tspan), 3])
    dcm_eq2ecef = []

    # Initial Prime Meridian angle of the planet [rad]
    theta_0 = PMST_0 * np.pi / 12

    for n in range(len(tspan)):
        # Compute angle of the Geodetic Frame wrt Equatorial Frame
        theta = theta_0 + omega_planet * (tspan[n] - t_0)

        # Transform position vector from Equatorial to Geodetic Frame (pcpf: planet centered - planet fixed)
        A_equatorial2pcpf = np.array([
            [cos(theta), sin(theta), 0],
            [-sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
        dcm_eq2ecef[n] = A_equatorial2pcpf
        rr_ecef[n, :] = np.dot(A_equatorial2pcpf, rr[n, :])

    return rr_ecef, dcm_eq2ecef


def eq2latlong(rr, tspan, t_0, PMST_0, omega_planet):
    """

    :param rr:              Set of "n" position vectors in the Equatorial frame (array rr(n, 3))
    :param tspan:           Array of the time corresponding to each position of the array rr
    :param t_0:             Reference epoch
    :param PMST_0:          Sidereal Time (ST) hour angle of the Prime Meridian (PM) at reference epoch
    :param omega_planet:    Rotational rate of the planet

    :return: lat_long, r_norm
    """
    # init output
    lat_long = np.zeros([len(tspan), 2])
    r_norm = np.zeros(len(tspan))

    # frame transformation to ecef
    rr_ecef, _ = eq2ecef(rr, tspan, t_0, PMST_0, omega_planet)

    for n in range(len(tspan)):
        r_norm[n] = norm(rr[n, :])

        # Compute latitude [rad]
        latitude = asin(rr_ecef[n, 2] / r_norm[n])
        # NOTE: latitude = delta

        # Compute longitude [rad]
        if rr_ecef[n, 1] >= 0:  # East
            longitude = acos(rr_ecef[n, 0] / r_norm[n] / cos(latitude))
        else:  # West
            longitude = - acos(rr_ecef[n, 0] / r_norm[n] / cos(latitude))

        lat_long[n, :] = [latitude, longitude]
    return lat_long, r_norm


def lvlh_framebuilder(xx):
    """
    Build the Direction Cosine Matrix of the LVLH-frame with respect to the Reference-frame.
    The R-bar points at the origin of the reference frame in which the sspacecraft state is expressed.

    :param xx: state vectors [position and velocity] [6]
    :return: LVLH-frame DCM transformation 3-by-3 matrix
    """

    rr = xx[:3]  # position vector
    vv = xx[3:]  # velocity vector
    hh = np.cross(rr, vv)  # angular momentum vector

    z_lvlh = -rr / norm(rr)  # R-bar
    y_lvlh = -hh / norm(hh)  # H-bar
    x_lvlh = np.cross(y_lvlh, z_lvlh)  # V-bar

    dcm_lvlh = np.array([x_lvlh, y_lvlh, z_lvlh])
    return dcm_lvlh
