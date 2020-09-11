import numpy as np

from poliastro.bodies import Body
import poliastro.core.perturbations as pert_fun

# Restricted 2-Body Problem
def R2BP_dyn(t, X, mu):
    x, y, z, vx, vy, vz = X

    r3 = np.linalg.norm(X[:3]) ** 3

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
    This function computes the acceleration perturbation due to the gravitational presence of a third body in the
    classical restricted 2-body problem.
    The position vectors shall be expressed in the same reference frame.

    :param t: Time
    :param rr: Position vector (of the spacecraft)
    :param mu: Gravitational parameter of the primary
    :param rr_3rd: Position vector of the 3rd body respect the main attractor
    :param mu_3rd: Gravitational parameter of the 3rd body

    :return: a_3rd - Acceleration perturbation due to 3rd body
    """

    rhorho = -rr_3rd
    dd = rr + rr_3rd

    rho = np.linalg.norm(rhorho)
    d = np.linalg.norm(dd)

    a_3rdB = - mu_3rd * (dd / d ** 3 + rhorho / rho ** 3)
    return a_3rdB


# J2 perturbation
def J2_perturbation(t, rr, mu, J2, R_p):
    """
    This function computes the perturbation acceleration due to the J2 term of the primary attractor

    :param t: Time
    :param rr: Body position
    :param mu: Gravitational parameter of the primary
    :param J2: J2 value of the primary
    :param R_p: Radius of the primary

    :returns: a_j2 - J2 perturbation acceleration
    """
    x, y, z = rr
    r = np.linalg.norm(rr)

    a_j2 = [
        3 / 2 * (J2 * mu * R_p ** 2) / r ** 4 * ((5 * ((z ** 2) / (r ** 2)) - 1) * x / r),
        3 / 2 * (J2 * mu * R_p ** 2) / r ** 4 * ((5 * ((z ** 2) / (r ** 2)) - 1) * y / r),
        3 / 2 * (J2 * mu * R_p ** 2) / r ** 4 * ((5 * ((z ** 2) / (r ** 2)) - 3) * z / r),
    ]
    return a_j2




# Rotational Dynamics driven by gravity gradient and considering rigid body


# Relative motion

