from numpy import *

from poliastro.bodies import Body
import poliastro.core.perturbations as pert_fun


# Restricted 2-Body Problem
def R2BP_dyn(t, X, mu):
    """
    Restricted Two Body Problem Dynamics.

    :param t: scalar
        Current time (s)
    :param X: numpy.ndarray 6x1
        State vector [x, y, z, vx, vy, vz] (km, km/s)
    :param mu: scalar
        Gravitational parameter of the primary (km^3/s^2)

    :return: X_dot numpy.ndarray 6x1 - Differential
    """
    x, y, z, vx, vy, vz = X

    r3 = linalg.norm(X[:3]) ** 3

    X_dot = [
        vx,
        vy,
        vz,
        - mu / r3 * x,
        - mu / r3 * y,
        - mu / r3 * z
    ]
    return X_dot


# Restricted 2-Body Problem plus perturbations
# IMPORT FROM POLIASTRO.CORE

# Third body perturbation acceleration
def thirdBody_perturbation(t, rr, mu, rr_3rd, mu_3rd):
    """
    Computation of the third body effect

    mu_3rdB   gravitational constant third body [km3/s2]
    rr_sc     position vector of the spacescraft wrt the moon [km]
    rr_moon   position vector of the moon wrt Jupiter [km]

    all the vector must be in the Equatorial plane of Jupiter

    f_3rdB    perturbation of the 3rd body [kN/kg]

    This function computes the acceleration perturbation due to the gravitational presence of a third body in the
    classical restricted 2-body problem.
    The position vectors shall be expressed in the same reference frame.

    :param t: Time
    :param rr: Position vector (of the spacecraft)
    :param mu: Gravitational parameter of the primary [km3/s2]
    :param rr_3rd: Position vector of the 3rd body respect the main attractor
    :param mu_3rd: Gravitational parameter of the 3rd body [km3/s2]

    :return: a_3rd - Acceleration perturbation due to 3rd body
    """

    rhorho = -rr_3rd[:3]
    dd = rr + rr_3rd[:3]

    rho = linalg.norm(rhorho)
    d = linalg.norm(dd)

    a_3rdB = - mu_3rd * (dd / d ** 3 + rhorho / rho ** 3)
    return a_3rdB


# J2 perturbation
def J2_perturbation(t, rr, mu, J2, R_p):
    """
    This function computes the perturbation acceleration due to the J2 term of the primary attractor
    Parameters
    ----------
    :param t: Time
    :param rr: Body position
    :param mu: Gravitational parameter of the primary
    :param J2: J2 value of the primary
    :param R_p: Radius of the primary

    :returns a_j2: J2 perturbation acceleration
    """
    x, y, z = rr
    r = linalg.norm(rr)

    a_j2 = [
        3 / 2 * (J2 * mu * R_p ** 2) / r ** 4 * ((5 * ((z ** 2) / (r ** 2)) - 1) * x / r),
        3 / 2 * (J2 * mu * R_p ** 2) / r ** 4 * ((5 * ((z ** 2) / (r ** 2)) - 1) * y / r),
        3 / 2 * (J2 * mu * R_p ** 2) / r ** 4 * ((5 * ((z ** 2) / (r ** 2)) - 3) * z / r),
    ]
    return a_j2

# Rotational Dynamics driven by gravity gradient and considering rigid body


# Relative motion
