import sys

from math import *
from numpy import *
from scipy import linalg
from scipy.integrate import solve_ivp
from pycse import *

from copy import *

from CR3BP_integration import *


def FirstGuessFixedAz3D(Az, mu, PointFlag):

    """
    First guess state for Halo orbits (fixed z0).

    :param Az:
    :param mu:
    :param PointFlag:
    :return:
    """

    # Identification of the equilibrium point, c2, c3, c4
    L1, L2, L3, L4x, L4y = LibrationPoints(mu)
    if PointFlag == 1:
        L = L1
        c3 = mu / (abs(L - 1 + mu)) ** 3 - (1 - mu) * abs(L - 1 + mu) / (abs(L + mu)) ** 4
    elif PointFlag == 2:
        L = L2
        c3 = - mu / (abs(L - 1 + mu)) ** 3 - (1 - mu) * abs(L - 1 + mu) / (abs(L + mu)) ** 4
    else:
        sys.exit('Incorrect PointFlag')

    c2 = mu / (abs(L - 1 + mu)) ** 3 + (1 - mu) / (abs(L + mu)) ** 3
    c4 = mu / (abs(L - 1 + mu)) ** 3 + (1 - mu) * abs(L - 1 + mu) ** 2 / (abs(L + mu)) ** 5

    # lam, k, Delta
    lam = sqrt((2 - c2 + sqrt(9 * c2 ** 2 - 8 * c2)) / 2)
    k = 2 * lam / (lam ** 2 + 1 - c2)
    Delta = lam ** 2 - c2

    # d1, d2, d3
    d1 = 16 * lam ** 4 + 4 * lam ** 2 * (c2 - 2) - 2 * c2 ** 2 + c2 + 1
    d2 = 81 * lam ** 4 + 9 * lam ** 2 * (c2 - 2) - 2 * c2 ** 2 + c2 + 1
    d3 = 2 * lam * (lam * (1 + k ** 2) - 2 * k)

    # aij, bij, dij
    a21 = 3 * c3 * (k ** 2 - 2) / (4 * (1 + 2 * c2))
    a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = - 3 * lam * c3 / (4 * k * d1) * (3 * k ** 3 * lam - 6 * k * (k - lam) + 4)
    a24 = - 3 * lam * c3 / (4 * k * d1) * (2 + 3 * lam * k)
    b21 = - 3 * c3 * lam / (2 * d1) * (3 * lam * k - 4)
    b22 = 3 * lam * c3 / d1
    d21 = - c3 / (2 * lam ** 2)
    a31 = - 9 * lam / d2 * (c3 * (k * a23 - b21) + k * c4 * (1 + 1 / 4 * k ** 2)) + (9 * lam ** 2 + 1 - c2) / (2 * d2) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k ** 2))
    a32 = - 9 * lam / (4 * d2) * (4 * c3 * (k * a24 - b22) + k * c4) - 3 * (9 * lam ** 2 + 1 - c2) / (2 * d2) * (c3 * (k * b22 + d21 - 2 * a24) - c4)
    b31 = 1 / d2 * (3 * lam * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k ** 2)) + (9 * lam ** 2 + 1 + 2 * c2) * (12 * c3 * (k * a23 - b21) + 3 * k * c4 * (4 + k ** 2)) / 8)
    b32 = 1 / d2 * (3 * lam * (3 * c3 * (k * b22 + d21 - 2 * a24) - 3 * c4) + (9 * lam ** 2 + 1 + 2 * c2) * (12 * c3 * (k * a24 - b22) + 3 * c4 * k) / 8)
    d31 = 3 / (64 * lam ** 2) * (4 * c3 * a24 + c4)
    d32 = 3 / (64 * lam ** 2) * (4 * c3 * (a23 - d21) + c4 * (4 + k ** 2))

    # s1, s2
    s1 = 1 / d3 * (3 / 2 * c3 * (2 * a21 * (k ** 2 - 2) - a23 * (k ** 2 + 2) - 2 * k * b21) - 3 / 8 * c4 * (3 * k ** 4 - 8 * k ** 2 + 8))
    s2 = 1 / d3 * (3 / 2 * c3 * (2 * a22 * (k ** 2 - 2) + a24 * (k ** 2 + 2) + 2 * k * b22 + d21 * (2 + 3)) + 3 / 8 * c4 * ((8 + 4) - k ** 2 * (2 - 1)))

    # Parameters for the Thurman and Worfolk correction
    b33 = - k / (16 * lam) * (12 * c3 * (b21 - 2 * k * a21 + k * a23) + 3 * c4 * k * (3 * k ** 2 - 4) + 16 * s1 * lam * (lam * k - 1))
    b34 = - k / (8 * lam) * (-12 * c3 * k * a22 + 3 * c4 * k + 8 * s2 * lam * (lam * k - 1))
    b35 = - k / (16 * lam) * (12 * c3 * (b22 + k * a24) + 3 * c4 * k)

    # a1, a2, l1, l2
    a1 = - 3 / 2 * c3 * (2 * a21 + a23 + 5 * d21) - 3 / 8 * c4 * (12 - k ** 2)
    a2 = 3 / 2 * c3 * (a24 - 2 * a22) + 9 / 8 * c4
    l1 = a1 + 2 * lam ** 2 * s1
    l2 = a2 + 2 * lam ** 2 * s2

    # Scale Az(reference frame centered at L with length unit = L-P2 distance)
    Az = Az / abs(L - 1 + mu)

    # Az, Ax, Ay(adimensional)
    Ax = sqrt(-(l2 * Az ** 2 + Delta) / l1)  # Ax = Ax(Az)subject to a constraint !!!
    # Ay = k * Ax

    # First guess initial condition (third order Lindstedt-Poincaré method)
    # x0
    x0 = -Ax + a21 * Ax ** 2 + a22 * Az ** 2 + a23 * Ax ** 2 - a24 * Az ** 2 + a31 * Ax ** 3 - a32 * Ax * Az ** 2

    # z0
    z0 = Az - 2 * d21 * Ax * Az + d32 * Az * Ax ** 2 - d31 * Az ** 3

    # v0
    om2 = s1 * Ax ** 2 + s2 * Az ** 2
    dtao1 = lam * (1 + om2)
    v0 = k * Ax * dtao1 + 2 * dtao1 * (b21 * Ax ** 2 - b22 * Az ** 2) + 3 * dtao1 * (b31 * Ax ** 3 - b32 * Ax * Az ** 2)

    # Thurman and Worfolk correction
    v0_mod = v0 + dtao1 * (b33 * Ax ** 3 + b34 * Ax * Az ** 2 - b35 * Ax * Az ** 2)

    # Re-transform x0 and vy0 in the R3BP synodic frame
    x0 = L + x0 * abs(L - 1 + mu)
    z0 = z0 * abs(L - 1 + mu)
    v0_mod = v0_mod * abs(L - 1 + mu)

    # First guess state
    X0 = array([x0, 0, z0, 0, v0_mod, 0])

    return X0


def SSDC_2D_x0fix(T0in, X0in, mu):

    """
    Single Shooting Differential Corrector for Periodic LPO (fixed x0).
    Suitable for IC = [x0 0 0 0 v0 0] (Lyapunov, DRO).

    :param T0in:
    :param X0in:
    :param mu:
    :return:
    """

    # Initialization
    T0 = deepcopy(T0in)
    X0 = deepcopy(X0in)
    STM0 = reshape(eye(6), (1, 36))
    tol = 1e-012
    Iter = int(0)
    MaxIter = int(350)

    # Integrate the initial condition
    # Up to the next intersection with the xz-plane
    _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                 TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
    if len(ie) == 0:
        sys.exit('Unable to compute first guess solution')

    # Pick the state at the half period
    tf = te[-1]
    Xf = xxMMe[-1, 0:6]
    STMf = reshape(xxMMe[-1, 6:42], (6, 6))
    err = abs(Xf[3])

    # Differential correction scheme
    while err > tol and Iter < MaxIter:

        # Compute correction (1x1 system reduction) (fast convergence rate, small error)
        ff = CR3BP(Xf, tf, mu)
        delta = Xf[3] * ff[1] / (STMf[3, 4] * ff[1] - STMf[1, 4] * ff[3])  # dv0

        # Correct the initial state for the new iteration
        X0[4] -= delta

        # Integrate the(k-1)-th corrected initial condition
        # Up to the next intersection with the xz-plane
        _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                     TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
        if len(ie) == 0:
            sys.exit('Unable to compute first guess solution')

        # Update iteration parameters
        tf = te[-1]
        Xf = xxMMe[-1, 0:6]
        STMf = reshape(xxMMe[-1, 6:42], (6, 6))
        err = abs(Xf[3])
        Iter += 1

    # Convergence check
    if Iter == MaxIter:
        sys.exit('No convergence (maximum number of iterations reached)')

    # Corrected initial condition
    T0 = 2 * te[-1]

    return T0, X0, Iter, err


def SSDC_3D_v0fix(T0in, X0in, mu):

    """
    Single Shooting Differential Corrector for Periodic LPO (fixed v0).
    Suitable for IC = [x0 0 0 0 v0 w0] (Axial,Vertical).

    :param T0in:
    :param X0in:
    :param mu:
    :return:
    """

    # Initialization
    T0 = deepcopy(T0in)
    X0 = deepcopy(X0in)
    STM0 = reshape(eye(6), (1, 36))
    tol = 1e-012
    Iter = int(0)
    MaxIter = int(350)

    # Integrate the initial condition
    # Up to the next intersection with the xz-plane
    _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                 TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
    if len(ie) == 0:
        sys.exit('Unable to compute first guess solution')

    # Pick the state at the half period
    tf = te[-1]
    Xf = xxMMe[-1, 0:6]
    STMf = reshape(xxMMe[-1, 6:42], (6, 6))
    err = max(abs(Xf[2]), abs(Xf[3]))

    # Differential correction scheme
    while err > tol and Iter < MaxIter:

        # Compute correction (3x3 system)
        ff = CR3BP(Xf, tf, mu)
        A = array([[STMf[1, 0], STMf[1, 5], ff[1]],
                   [STMf[2, 0], STMf[2, 5], ff[2]],
                   [STMf[3, 0], STMf[3, 5], ff[3]]])
        b = array([[Xf[1]], [Xf[2]], [Xf[3]]])
        delta = linalg.solve(A, b)  # [dx0, dw0, dt0]

        # Correct the initial state for the new iteration
        X0[0] -= delta[0, 0]
        X0[5] -= delta[1, 0]

        # Integrate the(k-1)-th corrected initial condition
        # Up to the next intersection with the xz-plane
        _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                     TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
        if len(ie) == 0:
            sys.exit('Unable to compute first guess solution')

        # Update iteration parameters
        tf = te[-1]
        Xf = xxMMe[-1, 0:6]
        STMf = reshape(xxMMe[-1, 6:42], (6, 6))
        err = max(abs(Xf[2]), abs(Xf[3]))
        Iter += 1

    # Convergence check
    if Iter == MaxIter:
        sys.exit('No convergence (maximum number of iterations reached)')

    # Corrected initial condition
    T0 = 2 * te[-1]

    return T0, X0, Iter, err


def SSDC_3D_w0fix(T0in, X0in, mu):

    """
    Single Shooting Differential Corrector for Periodic LPO (fixed w0).
    Suitable for IC = [x0 0 0 0 v0 w0] (Axial, Vertical).

    :param T0in:
    :param X0in:
    :param mu:
    :return:
    """

    # Initialization
    T0 = deepcopy(T0in)
    X0 = deepcopy(X0in)
    STM0 = reshape(eye(6), (1, 36))
    tol = 1e-012
    Iter = int(0)
    MaxIter = int(350)

    # Integrate the initial condition
    # Up to the next intersection with the xy-plane
    _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                 TOLERANCE=1e-014, events=[xyPlaneCrossing], args=(mu, ))
    if len(ie) == 0:
        sys.exit('Unable to compute first guess solution')

    # Pick the state at the half period
    tf = te[-1]
    Xf = xxMMe[-1, 0:6]
    STMf = reshape(xxMMe[-1, 6:42], (6, 6))
    err = max(abs(Xf[2]), abs(Xf[3]))

    # Differential correction scheme
    while err > tol and Iter < MaxIter:

        # Compute correction (3x3 system)
        ff = CR3BP(Xf, tf, mu)
        A = array([[STMf[1, 0], STMf[1, 4], ff[1]],
                   [STMf[2, 0], STMf[2, 4], ff[2]],
                   [STMf[3, 0], STMf[3, 4], ff[3]]])
        b = array([[Xf[1]], [Xf[2]], [Xf[3]]])
        delta = linalg.solve(A, b)  # [dx0, dv0, dt0]

        # Correct the initial state for the new iteration
        X0[0] -= delta[0, 0]
        X0[4] -= delta[1, 0]

        # Integrate the(k-1)-th corrected initial condition
        # Up to the next intersection with the xy-plane
        _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                     TOLERANCE=1e-014, events=[xyPlaneCrossing], args=(mu, ))
        if len(ie) == 0:
            sys.exit('Unable to compute first guess solution')

        # Update iteration parameters
        tf = te[-1]
        Xf = xxMMe[-1, 0:6]
        STMf = reshape(xxMMe[-1, 6:42], (6, 6))
        err = max(abs(Xf[2]), abs(Xf[3]))
        Iter += 1

    # Convergence check
    if Iter == MaxIter:
        sys.exit('No convergence (maximum number of iterations reached)')

    # Corrected initial condition
    T0 = 2 * te[-1]

    return T0, X0, Iter, err


def SSDC_3D_x0fix(T0in, X0in, mu):

    """
    Single Shooting Differential Corrector for Periodic LPO (fixed x0).
    Suitable for IC = [x0 0 z0 0 v0 0] (Halo).

    :param T0in:
    :param X0in:
    :param mu:
    :return:
    """

    # Initialization
    T0 = deepcopy(T0in)
    X0 = deepcopy(X0in)
    STM0 = reshape(eye(6), (1, 36))
    tol = 1e-012
    Iter = int(0)
    MaxIter = int(350)

    # Integrate the initial condition
    # Up to the next intersection with the xz-plane
    _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                 TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
    if len(ie) == 0:
        sys.exit('Unable to compute first guess solution')

    # Pick the state at the half period
    tf = te[-1]
    Xf = xxMMe[-1, 0:6]
    STMf = reshape(xxMMe[-1, 6:42], (6, 6))
    err = max(abs(Xf[3]), abs(Xf[5]))

    # Differential correction scheme
    while err > tol and Iter < MaxIter:

        # Compute correction (3x3 system)
        ff = CR3BP(Xf, tf, mu)
        A = array([[STMf[1, 2], STMf[1, 4], ff[1]],
                   [STMf[3, 2], STMf[3, 4], ff[3]],
                   [STMf[5, 2], STMf[5, 4], ff[5]]])
        b = array([[0], [Xf[3]], [Xf[5]]])
        delta = linalg.solve(A, b)  # [dz0, dv0, dt0]

        # Correct the initial state for the new iteration
        X0[2] -= delta[0, 0]
        X0[4] -= delta[1, 0]

        # Integrate the(k-1)-th corrected initial condition
        # Up to the next intersection with the xz-plane
        _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                     TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
        if len(ie) == 0:
            sys.exit('Unable to compute first guess solution')

        # Update iteration parameters
        tf = te[-1]
        Xf = xxMMe[-1, 0:6]
        STMf = reshape(xxMMe[-1, 6:42], (6, 6))
        err = max(abs(Xf[3]), abs(Xf[5]))
        Iter += 1

    # Convergence check
    if Iter == MaxIter:
        sys.exit('No convergence (maximum number of iterations reached)')

    # Corrected initial condition
    T0 = 2 * te[-1]

    return T0, X0, Iter, err


def SSDC_3D_z0fix(T0in, X0in, mu):

    """
    Single Shooting Differential Corrector for Periodic LPO (fixed z0).
    Suitable for IC = [x0 0 z0 0 v0 0] (Halo).

    :param T0in:
    :param X0in:
    :param mu:
    :return:
    """

    # Initialization
    T0 = deepcopy(T0in)
    X0 = deepcopy(X0in)
    STM0 = reshape(eye(6), (1, 36))
    tol = 1e-012
    Iter = int(0)
    MaxIter = int(350)

    # Integrate the initial condition
    # Up to the next intersection with the xz-plane
    _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                 TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
    if len(ie) == 0:
        sys.exit('Unable to compute first guess solution')

    # Pick the state at the half period
    tf = te[-1]
    Xf = xxMMe[-1, 0:6]
    STMf = reshape(xxMMe[-1, 6:42], (6, 6))
    err = max(abs(Xf[3]), abs(Xf[5]))

    # Differential correction scheme
    while err > tol and Iter < MaxIter:

        # Compute correction (3x3 system)
        ff = CR3BP(Xf, tf, mu)
        A = array([[STMf[1, 0], STMf[1, 4], ff[1]],
                   [STMf[3, 0], STMf[3, 4], ff[3]],
                   [STMf[5, 0], STMf[5, 4], ff[5]]])
        b = array([[0], [Xf[3]], [Xf[5]]])
        delta = linalg.solve(A, b)  # [dx0, dv0, dt0]

        # Correct the initial state for the new iteration
        X0[0] -= delta[0, 0]
        X0[4] -= delta[1, 0]

        # Integrate the(k-1)-th corrected initial condition
        # Up to the next intersection with the xz-plane
        _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                     TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
        if len(ie) == 0:
            sys.exit('Unable to compute first guess solution')

        # Update iteration parameters
        tf = te[-1]
        Xf = xxMMe[-1, 0:6]
        STMf = reshape(xxMMe[-1, 6:42], (6, 6))
        err = max(abs(Xf[3]), abs(Xf[5]))
        Iter += 1

    # Convergence check
    if Iter == MaxIter:
        sys.exit('No convergence (maximum number of iterations reached)')

    # Corrected initial condition
    T0 = 2 * te[-1]

    return T0, X0, Iter, err


def HaloFixedAz(Az, mu, PointFlag, Npts):

    """
    Halo orbit (3D) for fixed Az amplitude.

    :param Az:
    :param mu:
    :param PointFlag:
    :param Npts:
    :return:
    """

    # Compute first guess state
    X0 = FirstGuessFixedAz3D(Az, mu, PointFlag)

    # Initialize the corrector
    STM0 = reshape(eye(6), (1, 36))
    tol1 = 1e-012
    tol2 = 1e-012
    Iter1 = int(0)
    Iter2 = int(0)
    MaxIter1 = int(350)
    MaxIter2 = int(350)

    # Loop #1
    # Integrate the initial condition
    # Up to the next intersection with the xz-plane
    _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, 2 * pi),
                                 TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))

    if len(ie) == 0:
        sys.exit('Unable to compute first guess solution')

    # Pick the state at the half period
    tf = te[-1]
    Xf = xxMMe[-1, 0:6]
    STMf = reshape(xxMMe[-1, 6:42], (6, 6))
    err = max(abs(Xf[3]), abs(Xf[5]))

    # Differential correction scheme (Loop #1)
    while err > tol1 and Iter1 < MaxIter1:

        # Compute correction (3x3 system)
        ff = CR3BP(Xf, tf, mu)
        A = array([[STMf[1, 0], STMf[1, 4], ff[1]],
                   [STMf[3, 0], STMf[3, 4], ff[3]],
                   [STMf[5, 0], STMf[5, 4], ff[5]]])
        b = array([[0], [Xf[3]], [Xf[5]]])
        delta = linalg.solve(A, b)  # [dx0, dv0, dt0]

        # Correct the initial state for the new iteration
        X0[0] -= delta[0, 0]
        X0[4] -= delta[1, 0]

        # Integrate the(k-1)-th corrected initial condition
        # Up to the next intersection with the xz-plane
        _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, 2 * pi),
                                     TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu,))
        if len(ie) == 0:
            sys.exit('Unable to compute first guess solution')

        # Update iteration parameters
        tf = te[-1]
        Xf = xxMMe[-1, 0:6]
        STMf = reshape(xxMMe[-1, 6:42], (6, 6))
        err = max(abs(Xf[3]), abs(Xf[5]))
        Iter1 += 1

    # Convergence check (Loop #1)
    if Iter1 == MaxIter1:
        sys.exit('Loop #1, no convergence (maximum number of iterations reached)')

    # Corrected initial condition
    T0 = 2 * te[-1]

    # Loop #2
    # Integrate the initial condition
    xxMM = odeint(STM, concatenate((X0, STM0), axis=None), linspace(0, T0, int(1e+003+1)),
                  args=(mu,), atol=1e-014, rtol=1e-012)
    # tt, xxMM, *_ = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0, Npts),
    #                                  TOLERANCE=1e-014, events=[], args=(mu,))

    # Pick the state at the end of the period
    tf = deepcopy(T0)
    # tf = tt[-1]
    Xf = xxMM[-1, 0:6]
    STMf = reshape(xxMM[-1, 6:42], (6, 6))
    err = linalg.norm(X0 - Xf)

    # Differential correction scheme (Loop #2)
    while err > tol2 and Iter2 < MaxIter2:

        # Compute correction (3x3 system)
        ff = CR3BP(Xf, tf, mu)
        A = array([[STMf[1, 0], STMf[1, 4], ff[1]],
                   [STMf[3, 0], STMf[3, 4], ff[3]],
                   [STMf[5, 0], STMf[5, 4], ff[5]]])
        b = array([[0], [Xf[3]], [Xf[5]]])
        delta = linalg.solve(A, b)  # [dx0, dv0, dt0]

        # Correct the initial state for the new iteration
        T0 -= delta[2, 0]
        X0[0] -= delta[0, 0]
        X0[4] -= delta[1, 0]

        # Integrate the(k-1)-th corrected initial condition
        xxMM = odeint(STM, concatenate((X0, STM0), axis=None), linspace(0, T0, int(1e+003+1)),
                      args=(mu,), atol=1e-014, rtol=1e-012)
        # tt, xxMM, *_ = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0, Npts),
        #                                 TOLERANCE=1e-014, events=[], args=(mu,))

        # Update iteration parameters
        tf = deepcopy(T0)
        # tf = tt[-1]
        Xf = xxMM[-1, 0:6]
        STMf = reshape(xxMM[-1, 6:42], (6, 6))
        err = linalg.norm(X0 - Xf)
        Iter2 += 1

        print(Iter2)
        print(err)

    # Convergence check (Loop #2)
    if Iter2 == MaxIter2:
        sys.exit('Loop #2, no convergence (maximum number of iterations reached)')

    #
    Iter = Iter1 + Iter2

    return linspace(0, T0, int(1e+002+1)), xxMM, Iter, err


def PseudoArchlength_2D_AxiSym(T0in, X0in, mu, ds):

    """
    Pseudo-Arclength Continuation algorithm for 2D Axi Symmetric Periodic LPO.
    Suitable for IC = [x0 0 0 0 v0 0] (Lyapunov, DRO).

    :param T0in:
    :param X0in:
    :param mu:
    :param ds:
    :return:
    """

    # Initialization
    T0 = deepcopy(T0in)
    X0 = deepcopy(X0in)
    STM0 = reshape(eye(6), (1, 36))
    tol = 1e-012
    Iter = int(0)
    MaxIter = int(350)

    # Integrate the initial condition
    # Up to the next intersection with the xz-plane
    _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                 TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
    if len(ie) == 0:
        sys.exit('Unable to compute first guess solution')

    # Pick the state at the half period
    tf = te[-1]
    Xf = xxMMe[-1, 0:6]
    STMf = reshape(xxMMe[-1, 6:42], (6, 6))

    # Find Null Space direction along the family
    ff = CR3BP(Xf, tf, mu)
    DFk = array([[STMf[1, 0], STMf[1, 4], ff[1]],
                 [STMf[3, 0], STMf[3, 4], ff[3]]])
    Fk = array([[Xf[1]], [Xf[3]]])
    k = linalg.null_space(DFk)

    # Augmented constraint
    x = array([[X0[0]], [X0[4]], [tf]])
    xc = array([[X0in[0]], [X0in[4]], [T0in / 2]])
    Gk = concatenate((Fk, transpose(x - xc) @ k - ds))
    DGk = concatenate((DFk, transpose(k)))
    delta = linalg.solve(DGk, Gk)  # [dx0, dv0, dt]

    # Update iteration parameters
    err = max(abs(Gk))

    # Pseudo-Arclength Continuation algorithm
    while err > tol and Iter < MaxIter:

        # Correct the initial state for the new iteration
        T0 = 2 * (tf + delta[2, 0])
        X0[0] -= delta[0, 0]
        X0[4] -= delta[1, 0]

        # Integrate the(k-1)-th corrected initial condition
        # Up to the next intersection with the xz-plane
        _, _, te, xxMMe, ie = odelay(STM, concatenate((X0, STM0), axis=None), linspace(0, T0),
                                     TOLERANCE=1e-014, events=[xzPlaneCrossing], args=(mu, ))
        if len(ie) == 0:
            sys.exit('Unable to compute first guess solution')

        # Newton-Raphson Method
        ff = CR3BP(Xf, tf, mu)
        DFk = array([[STMf[1, 0], STMf[1, 4], ff[1]],
                     [STMf[3, 0], STMf[3, 4], ff[3]]])
        Fk = array([[Xf[1]], [Xf[3]]])

        # Augmented constraint
        x = array([[X0[0]], [X0[4]], [tf]])
        xc = array([[X0in[0]], [X0in[4]], [T0in / 2]])
        Gk = concatenate((Fk, transpose(x - xc) @ k - ds))
        DGk = concatenate((DFk, transpose(k)))
        delta = linalg.solve(DGk, Gk)  # [dx0, dv0, dt]

        # Update iteration parameters
        err = max(abs(Gk))
        Iter += 1
        print(err)
        print(Iter)

    # Convergence check
    if Iter == MaxIter:
        sys.exit('No convergence (maximum number of iterations reached)')

    return T0, X0, Iter, err


def xzPlaneCrossing(y, t):

    value = y[1]
    isterminal = True
    direction = -1

    return value, isterminal, direction


def xyPlaneCrossing(y, t):

    value = y[2]
    isterminal = True
    direction = -1

    return value, isterminal, direction
