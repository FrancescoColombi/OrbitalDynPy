import numpy as np
from scipy.integrate import solve_ivp, solve_bvp


def qck(angle):
    """
    function [angle] = qck(angle)

    qck.m - Reduce an angle between 0 and 2*pi

    PROTOTYPE:
      [angle]=qck(angle)

    DESCRIPTION:
    This function takes any angle and reduces it, if necessary,
    so that it lies in the range from 0 to 2 PI radians.

    INPUTS:
      ANGLE[1]    Angle to be reduced (in radians)

    OUTPUTS:
      QCK[1]      The angle reduced, if necessary, to the range
                  from 0 to 2 PI radians (in radians)

    CALLED FUNCTIONS:
      pi (from MATLAB)

    AUTHOR:
      W.T. Fowler, July, 1978

    CHANGELOG:
      8/20/90, REVISION: Darrel Monroe

    """
    twopi = 2 * np.pi

    diff = twopi * (np.fix(angle / twopi) + min([0, np.sign(angle)]))

    angle = angle - diff

    return angle


def lamberts_universal_variables(rr0, rr1, tof, mu, args={}):
    """
    Solve Lambert's problem using universal variable method

    from AWP | Astrodynamics with Python by Alfonso Gonzalez
    https://github.com/alfonsogonzalez/AWP

    Reference: Lambert Universal Variable Algorithm
    https://www.researchgate.net/publication/236012521_Lambert_Universal_Variable_Algorithm
    Authors: M. A. Sharaf
             A. S. Saad, Qassim University
             Mohamed Ibrahim Nouh, National Research Institute of Astronomy and Geophysics

    AUTHOR:
      Alfonso Gonzalez, July, 2021

    CHANGELOG:
      2023/02/18, REVISION: Francesco Colombi
    """
    _args = {
        'tm': 1,
        'tol': 1e-6,
        'max_steps': 1000,
        'psi': 0.0,
        'psi_u': 4.0 * np.pi ** 2,
        'psi_l': -4.0 * np.pi ** 2,
    }
    for key in args.keys():
        _args[key] = args[key]
    psi = _args['psi']
    psi_l = _args['psi_l']
    psi_u = _args['psi_u']

    sqrt_mu = np.sqrt(mu)
    r0_norm = np.linalg.norm(rr0)
    r1_norm = np.linalg.norm(rr1)
    gamma = np.dot(rr0, rr1) / r0_norm / r1_norm
    c2 = 0.5
    c3 = 1 / 6.0
    solved = False
    A = _args['tm'] * np.sqrt(r0_norm * r1_norm * (1 + gamma))

    if A == 0.0:
        raise RuntimeWarning('Universal variables solution was passed in Hohmann transfer')
        return np.array([0, 0, 0]), np.array([0, 0, 0])

    for n in range(_args['max_steps']):
        B = r0_norm + r1_norm + A * (psi * c3 - 1) / np.sqrt(c2)

        if A > 0.0 and B < 0.0:
            psi_l += np.pi
            B *= -1.0

        chi3 = np.sqrt(B / c2) ** 3
        deltat_ = (chi3 * c3 + A * np.sqrt(B)) / sqrt_mu

        if abs(tof - deltat_) < _args['tol']:
            solved = True
            break

        if deltat_ <= tof:
            psi_l = psi

        else:
            psi_u = psi

        psi = (psi_u + psi_l) / 2.0
        c2, c3 = findc2c3(psi)

    if not solved:
        raise RuntimeWarning(
            'Universal variables solver did not converge.')
        return np.array([0, 0, 0]), np.array([0, 0, 0])

    f = 1 - B / r0_norm
    g = A * np.sqrt(B / mu)
    gdot = 1 - B / r1_norm
    vv0 = (rr1 - f * rr0) / g
    vv1 = (gdot * rr1 - rr0) / g

    return vv0, vv1


def findc2c3(znew):
    """
    this function calculates the c2 and c3 functions for use in the universal variable calculation of z.

    author        : david vallado                  719-573-2600   27 may 2002

    revisions
                  -

    inputs          description                    range / units
      znew        - z variable                     rad2

    outputs       :
      c2new       - c2 function value
      c3new       - c3 function value

    locals        :
      sqrtz       - square root of znew

    coupling      :
      sinh        - hyperbolic sine
      cosh        - hyperbolic cosine

    references    :
      vallado       2001, 70-71, alg 1

    [c2new,c3new] = findc2c3 ( znew );
    ------------------------------------------------------------------------------
    """
    small = 0.00000001
    if znew > small:
        sqrtz = np.sqrt(znew)
        c2new = (1.0 - np.cos(sqrtz)) / znew
        c3new = (sqrtz - np.sin(sqrtz)) / (sqrtz ** 3)
    else:
        if znew < -small:
            sqrtz = np.sqrt(-znew)
            c2new = (1.0 - np.cosh(sqrtz)) / znew
            c3new = (np.sinh(sqrtz) - sqrtz) / (sqrtz ** 3)
        else:
            c2new = 0.5
            c3new = 1.0 / 6.0
    return c2new, c3new
