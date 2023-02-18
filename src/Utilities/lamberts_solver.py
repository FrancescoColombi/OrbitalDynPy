from numpy import sin, cos, tan, sinh, cosh, tanh, sqrt, pi
from numpy import arcsin as asin
from numpy import arccos as acos
from numpy import arctan as atan
from math import atan2
from numpy import dot, cross
from numpy.linalg import norm
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import warnings


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
      - 1990/08/20, REVISION: Darrel Monroe
      - 2023/02/18, Francesco Colombi: import in Python
    """
    twopi = 2 * pi

    diff = twopi * (np.fix(angle / twopi) + min([0, np.sign(angle)]))

    angle = angle - diff

    return angle


def h_E(E, y, m, Nrev):
    """
    function [h, dh] = h_E(E, y, m, Nrev)

    # h_E.m - Equation of multirevolution Lambert's problem h = h(E).
    #
    # PROTOTYPE:
    #   [h, dh] = h_E(E, y, m, Nrev)
    #
    # DESCRIPTION:
    #   Equation of multirevolution Lambert's problem:
    #   h(E) = (Nrev*pi + E - sin(E)) / tan(E/2)^3 - 4/m * (y^3 - y^2)
    #   See: "USING BATTIN METHOD TO OBTAIN MULTIPLE-REVOLUTION LAMBERT'S
    #      SOLUTIONS", Shen, Tsiotras, pag. 12
    #
    # INPUT
    #   E, y, m, Nrev   See paper for detailed description.
    #
    # OUTPUT
    #   h               Value of h(E).
    #   dh              Value of dh(E)/dE.
    #
    # ORIGINAL VERSION:
    #   Camilla Colombo, 20/02/2006, MATLAB, cubicN.m
    #
    # AUTHOR:
    #   Matteo Ceriotti, 27/01/2009
    #   - changed name of cubicN.m and added at the bottom of lambertMR.m file
    #   - 2023/02/18, Francesco Colombi: import in Python
    # -------------------------------------------------------------------------

    tanE2 = tan(E/2);
    h = (Nrev*pi + E - sin(E)) / tanE2^3 - 4/m * (y^3 - y^2);

    if nargout > 1  # two output arguments
        # h'(E)
        dh = (1-cos(E))/tanE2^3 - 3/2*(Nrev*pi+E-sin(E))*sec(E/2)^2 / tanE2^4;
    end

    return
    """
    tanE2 = tan(E / 2)
    h = (Nrev * pi + E - sin(E)) / tanE2 ** 3 - 4 / m * (y ** 3 - y ** 2)
    dh = (1 - cos(E)) / tanE2 ** 3 - 3 / 2 * (Nrev * pi + E - sin(E)) / cos(E / 2) ** 2 / tanE2 ** 4
    return h, dh


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
        'psi_u': 4.0 * pi ** 2,
        'psi_l': -4.0 * pi ** 2,
    }
    for key in args.keys():
        _args[key] = args[key]
    psi = _args['psi']
    psi_l = _args['psi_l']
    psi_u = _args['psi_u']

    sqrt_mu = sqrt(mu)
    r0_norm = norm(rr0)
    r1_norm = norm(rr1)
    gamma = np.dot(rr0, rr1) / r0_norm / r1_norm
    c2 = 0.5
    c3 = 1 / 6.0
    solved = False
    A = _args['tm'] * sqrt(r0_norm * r1_norm * (1 + gamma))

    if A == 0.0:
        raise RuntimeWarning('Universal variables solution was passed in Hohmann transfer')
        return np.array([0, 0, 0]), np.array([0, 0, 0])

    for n in range(_args['max_steps']):
        B = r0_norm + r1_norm + A * (psi * c3 - 1) / sqrt(c2)

        if A > 0.0 and B < 0.0:
            psi_l += pi
            B *= -1.0

        chi3 = sqrt(B / c2) ** 3
        deltat_ = (chi3 * c3 + A * sqrt(B)) / sqrt_mu

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
    g = A * sqrt(B / mu)
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
        sqrtz = sqrt(znew)
        c2new = (1.0 - cos(sqrtz)) / znew
        c3new = (sqrtz - sin(sqrtz)) / (sqrtz ** 3)
    else:
        if znew < -small:
            sqrtz = sqrt(-znew)
            c2new = (1.0 - cosh(sqrtz)) / znew
            c3new = (sinh(sqrtz) - sqrtz) / (sqrtz ** 3)
        else:
            c2new = 0.5
            c3new = 1.0 / 6.0
    return c2new, c3new


def lambertMR(RI, RF, TOF, MU, orbitType=0, Nrev=0, Ncase=0, optionsLMR=0):
    """
    function [A,P,E,ERROR,VI,VF,TPAR,THETA] = lambertMR(RI,RF,TOF,MU,orbitType,Nrev,Ncase,optionsLMR)

    # lambertMR.m - Lambert's problem solver for all possible transfers
    #   (multi-revolution transfer included).
    #
    # PROTOTYPE:
    #   [A,P,E,ERROR,VI,VF,TPAR,THETA] = lambertMR(RI,RF,TOF,MU,orbitType,Nrev,Ncase,optionsLMR)
    #
    # DESCRIPTION:
    #   Lambert's problem solver for all possible transfers:
    #       1- zero-revolution (for all possible types of orbits: circles, ellipses,
    #       	parabolas and hyperbolas)
    #       2- multirevolution case
    #       3- inversion of the motion
    #
    #   1- ZERO-REVOLUTION LAMBERT'S PROBLEM
    #
    #   For the solution of Lambert's problem with number of revolution = 0 the
    #   subroutine by Chris D'Souza is included here.
    #   This subroutine is a Lambert algorithm which given two radius vectors
    #   and the time to get from one to the other, it finds the orbit
    #   connecting the two. It solves the problem using a new algorithm
    #   developed by R. Battin. It solves the Lambert problem for all possible
    #   types of orbits (circles, ellipses, parabolas and hyperbolas).
    #   The only singularity is for the case of a transfer angle of 360 degrees,
    #   which is a rather obscure case.
    #   It computes the velocity vectors corresponding to the given radius
    #   vectors except for the case when the transfer angle is 180 degrees
    #   in which case the orbit plane is ambiguous (an infinite number of
    #   transfer orbits exist).
    #
    #   2- MULTIREVOLUTION LAMBERT'S PROBLEM
    #
    #   For the solution of Lambert's problem with Nrev>0 number of revolution,
    #   Battin's formulation has been extended to accomodate N-revolution
    #   transfer orbits, by following the paper: "Using Battin Mathod to obtain
    #   Multiple-revolution Lambert's Solutions" by Shen and Tsiotras.
    #
    #   When Nrev>0 the possible orbits are just ellipses.
    #   If 0<=Nrev<=Nmax, there are two Nrev-revolution transfer orbits.
    #   These two transfer orbits have different semi-major axis and they may
    #   be all combinations of large-e and small-e transfer orbits.
    #   The Original Successive Substitution Method by Battin converges to one
    #   of the two possible solution with a viable initial guest, however it
    #   diverges from the other one. Then a Reversed Successive Substitution is
    #   used to converge to the second solution.
    #   A procedure is implemented in order to guarantee to provide initial
    #   guesses in the convergence region. If Nrew exceeds the maximum number
    #   of revolution an ERROR is given:
    #   warning('off','lambertMR:SuccessiveSubstitutionDiverged') to take out
    #   the warnings or use optionsLMR(1) = 0.
    #
    #   3- INVERSION OF THE MOTION
    #
    #   Direct or retrograde option can be selected for the transfer.
    #
    #   The algorithm computes the semi-major axis, the parameter (semi-latus
    #   rectum), the eccentricity and the velocity vectors.
    #
    #   NOTE: If ERROR occurs or the 360 or 180 degree transfer case is
    #   encountered.
    #
    # INPUT:
    #	RI[3]           Vector containing the initial position in Cartesian
    #                   coordinates [L].
    #	RF[3]           Vector containing the final position vector in
    #                   Cartesian coordinates [L].
    #	TOF[1]          Transfer time, time of flight [T].
    #  	MU[1]           Planetary constant of the planet (mu = mass * G) [L^3/T^2]
    #	orbitType[1]    Logical variable defining whether transfer is
    #                       0: direct transfer from R1 to R2 (counterclockwise)
    #                       1: retrograde transfer from R1 to R2 (clockwise)
    #	Nrev[1]         Number of revolutions.
    #                   if Nrev = 0 ZERO-REVOLUTION transfer is calculated
    #                   if Nrev > 0 two transfers are possible. Ncase should be
    #                          defined to select one of the two.
    #	Ncase[1]        Logical variable defining the small-a or large-a option
    #                   in case of Nrev>0:
    #                       0: small-a option
    #                       1: large-a option
    #	optionsLMR[1]	lambertMR options:
    #                    optionsLMR(1) = display options:
    #                                    0: no display
    #                                    1: warnings are displayed only when
    #                                       the algorithm does not converge
    #                                    2: full warnings displayed
    #
    # OUTPUT:
    #	A[1]        Semi-major axis of the transfer orbit [L].
    # 	P[1]        Semi-latus rectum of the transfer orbit [L].
    #  	E[1]        Eccentricity of the transfer orbit.
    #	ERROR[1]	Error flag
    #                   0:	No error
    #                   1:	Error, routine failed to converge
    #                   -1:	180 degrees transfer
    #                   2:  360 degrees transfer
    #                   3:  the algorithm doesn't converge because the number
    #                       of revolutions is bigger than Nrevmax for that TOF
    #                   4:  Routine failed to converge, maximum number of
    #                       iterations exceeded.
    #	VI[3]       Vector containing the initial velocity vector in Cartesian
    #               coordinates [L/T].
    #	VT[1]		Vector containing the final velocity vector in Cartesian
    #               coordinates [L/T].
    #	TPAR[1] 	Parabolic flight time between RI and RF [T].
    #	THETA[1]	Transfer angle [radians].
    #
    # NOTE: The semi-major axis, positions, times, and gravitational parameter
    #       must be in compatible units.
    #
    # CALLED FUNCTIONS:
    #   qck, h_E (added at the bottom of this file)
    #
    # REFERENCES:
    #   - Shen and Tsiotras, "Using Battin method to obtain Multiple-Revolution
    #       Lambert's solutions".
    #   - Battin R., "An Introduction to the Mathematics and Methods of
    #       Astrodynamics, Revised Edition", 1999.
    #
    # FUTURE DEVELOPMENT:
    #   - 180 degrees transfer indetermination
    #   - 360 degrees transfer singularity
    #   - Nmax number of max revolution for a given TOF:
    #     work in progress - Camilla Colombo
    #
    # ORIGINAL VERSION:
    #   Chris D'Souza, 20/01/1989, MATLAB, lambert.m
    #       verified by Darrel Monroe, 10/25/90
    #       - Labert.m solved only direct transfer, without multi-revolution
    #         option
    #
    # AUTHOR:
    #   Camilla Colombo, 10/11/2006, MATLAB, lambertMR.m
    #
    # CHANGELOG:
    #   13/11/2006, Camilla Colombo: added ERROR = 3 if Nrev > NrevMAX
    #	21/11/2006, Camilla Colombo: added another case of ERROR = 3 (index
    #   	N3) corresponding to the limit case when small-a solution = large-a
    #       solution. No solution is given in this case.
    #	06/08/2007, Camilla Colombo: optionsLMR added as an input
    #	28/11/2007, Camilla Colombo: minor changes
    #   29/01/2009, Matteo Ceriotti:
    #       - Introduced variable for maximum number of iterations nitermax.
    #       - Corrected final check on maximum number of iterations exceeded, from
    #           "==" to ">=" (if N1 >= nitermax || N >= nitermax).
    #       - Increased maxumum number of iterations to 2000, not to lose some
    #           solutions.
    #       - In OSS loop, added check for maximum number of iterations exceeded,
    #           which then sets checkNconvOSS = 0.
    #       - Changed the way of coumputing X given Y1 in RSS. Now the
    #           Newton-Raphson method with initial guess suggested by Shen,
    #           Tsiotras is used. This should guarantee convergence without the
    #           need of an external zero finder (fsolve).
    #       - Changed absolute tolerance into relative tolerance in all loops X0-X.
    #           Now the condition is: while "abs(X0-X) >= abs(X)*TOL+TOL".
    #       - Added return immediately when any error is detected.
    #       - Moved check on 4*TOF*LAMBDA==0 after computing LAMBDA.
    #       - Moved check on THETA==0 || THETA==2*PI after computing THETA.
    #       - Added error code 4 (number of iterations exceeded).
    #       - Removed variable Nwhile, as not strictly needed.
    #       - Removed variable PIE=pi.
    #   29/01/2009, REVISION: Matteo Ceriotti
    #   21/07/2009, Matteo Ceriotti, Camilla Colombo:
    #       added condition to detect case 180 degrees transfer indetermination
    #   30/01/2010, Camilla Colombo: Header and function name in accordance
    #       with guidlines.
    #
    # Note: Please if you have got any changes that you would like to be done,
    #   do not change the function, please contact the author.
    #
    # -------------------------------------------------------------------------
    """

    nitermax = 2000  # Maximum number of iterations for loops
    TOL = 1e-14

    TWOPI = 2 * pi

    # Reset
    A = 0.
    P = 0.
    E = 0.
    VI = np.array([0., 0., 0.])
    VF = np.array([0., 0., 0.])

    # ----------------------------------
    # Compute the vector magnitudes and various cross and dot products

    RIM2 = dot(RI, RI)
    RIM = sqrt(RIM2)
    RFM2 = dot(RF, RF)
    RFM = sqrt(RFM2)
    CTH = dot(RI, RF) / (RIM * RFM)
    CR = cross(RI, RF)
    STH = norm(CR) / (RIM * RFM)

    # Choose angle for up angular momentum
    match orbitType:
        case 0:  # direct transfer
            if CR[2] < 0:
                STH = -STH
        case 1:  # retrograde transfer
            if CR[2] > 0:
                STH = -STH
        case _:
            raise Exception('{0} is not an allowed orbitType'.format(orbitType))

    THETA = qck(atan2(STH, CTH))
    # if abs(THETA - pi) >= 0.01
    if THETA == TWOPI or THETA == 0:
        ERROR = 2
        A = 0
        P = 0
        E = 0
        VI = np.array([0, 0, 0])
        VF = np.array([0, 0, 0])
        TPAR = 0
        THETA = 0
        return A, P, E, ERROR, VI, VF, TPAR, THETA

    B1 = np.sign(STH)
    if STH == 0:
        B1 = 1

    # ----------------------------------
    # Compute the chord and the semi-perimeter

    C = sqrt(RIM2 + RFM2 - 2. * RIM * RFM * CTH)
    S = (RIM + RFM + C) / 2.
    BETA = 2. * asin(sqrt((S - C) / S))
    PMIN = TWOPI * sqrt(S ** 3 / (8. * MU))
    TMIN = PMIN * (pi - BETA + sin(BETA)) / (TWOPI)
    LAMBDA = B1 * sqrt((S - C) / S)

    if 4 * TOF * LAMBDA == 0 or abs((S - C) / S) < TOL:
        ERROR = -1
        A = 0
        P = 0
        E = 0
        VI = np.array([0, 0, 0])
        VF = np.array([0, 0, 0])
        TPAR = 0
        THETA = 0
        return A, P, E, ERROR, VI, VF, TPAR, THETA

    # ----------------------------------
    # Compute L carefully for transfer angles less than 5 degrees

    if THETA * 180 / pi <= 5:
        W = atan((RFM / RIM) ** .25) - pi / 4.
        R1 = (sin(THETA / 4.)) ** 2
        S1 = (tan(2. * W)) ** 2
        L = (R1 + S1) / (R1 + S1 + cos(THETA / 2.))
    else:
        L = ((1. - LAMBDA) / (1. + LAMBDA)) ** 2

    M = 8. * MU * TOF ** 2 / (S ** 3 * (1. + LAMBDA) ** 6)
    TPAR = (sqrt(2. / MU) / 3.) * (S ** 1.5 - B1 * (S - C) ** 1.5)
    L1 = (1. - L) / 2.

    CHECKFEAS = 0
    N1 = 0
    N = 0

    if Nrev == 0:
        # ----------------------------------
        # Initialize values of y, n, and x

        Y = 1
        N = 0
        N1 = 0
        ERROR = 0
        # CHECKFEAS=0

        if (TOF - TPAR) <= 1e-3:
            X0 = 0
        else:
            X0 = L

        X = -1.e8

        # ----------------------------------
        # Begin iteration

        # ---> CL: 26/01/2009, Matteo Ceriotti:
        #       Changed absolute tolerance into relative tolerance here below.
        while (abs(X0 - X) >= abs(X) * TOL + TOL) and (N <= nitermax):
            N = N + 1
            X = X0
            ETA = X / (sqrt(1. + X) + 1.) ** 2
            CHECKFEAS = 1

            # ----------------------------------
            # Compute x by means of an algorithm devised by
            # Gauticci for evaluating continued fractions by the
            # 'Top Down' method

            DELTA = 1
            U = 1
            SIGMA = 1
            M1 = 0

            while abs(U) > TOL and M1 <= nitermax:
                M1 = M1 + 1
                GAMMA = (M1 + 3.) ** 2 / (4. * (M1 + 3.) ** 2 - 1.)
                DELTA = 1. / (1. + GAMMA * ETA * DELTA)
                U = U * (DELTA - 1.)
                SIGMA = SIGMA + U

            C1 = 8. * (sqrt(1. + X) + 1.) / (3. + 1. / (5. + ETA + (9. * ETA / 7.) * SIGMA))

            # ----------------------------------
            # Compute H1 and H2

            if N == 1:
                DENOM = (1. + 2. * X + L) * (3. * C1 + X * C1 + 4. * X)
                H1 = (L + X) ** 2 * (C1 + 1. + 3. * X) / DENOM
                H2 = M * (C1 + X - L) / DENOM
            else:
                QR = sqrt(L1 ** 2 + M / Y ** 2)
                XPLL = QR - L1
                LP2XP1 = 2. * QR
                DENOM = LP2XP1 * (3. * C1 + X * C1 + 4. * X)
                H1 = ((XPLL ** 2) * (C1 + 1. + 3. * X)) / DENOM
                H2 = M * (C1 + X - L) / DENOM

            B = 27. * H2 / (4. * (1. + H1) ** 3)
            U = -B / (2. * (sqrt(B + 1.) + 1.))

            # ----------------------------------
            # Compute the continued fraction expansion K(u)
            # by means of the 'Top Down' method

            # Y can be computed finding the roots of the formula and selecting
            # the real one:
            # y^3 - (1+h1)*y^2 - h2 = 0     (7.113) Battin
            #
            # Ycami_ = roots([1 -1-H1 0 -H2])
            # kcami = find( abs(imag(Ycami_)) < eps );
            # Ycami = Ycami_(kcami)

            DELTA = 1
            U0 = 1
            SIGMA = 1
            N1 = 0

            while N1 < nitermax and abs(U0) >= TOL:
                if N1 == 0:
                    GAMMA = 4 / 27
                    DELTA = 1 / (1 - GAMMA * U * DELTA)
                    U0 = U0 * (DELTA - 1)
                    SIGMA = SIGMA + U0
                else:
                    for I8 in [1, 2]:
                        if I8 == 1:
                            GAMMA = 2 * (3 * N1 + 1) * (6 * N1 - 1) / (9 * (4 * N1 - 1) * (4 * N1 + 1))
                        else:
                            GAMMA = 2 * (3 * N1 + 2) * (6 * N1 + 1) / (9 * (4 * N1 + 1) * (4 * N1 + 3))
                        DELTA = 1 / (1 - GAMMA * U * DELTA)
                        U0 = U0 * (DELTA - 1)
                        SIGMA = SIGMA + U0

                N1 = N1 + 1

            KU = (SIGMA / 3) ** 2
            Y = ((1 + H1) / 3) * (2 + sqrt(B + 1) / (1 - 2 * U * KU))  # Y = Ycami

            X0 = sqrt(((1 - L) / 2) ** 2 + M / Y ** 2) - (1 + L) / 2
            # fprintf('n= %d, x0=%.14f\n',N,X0);

    # MULTIREVOLUTION
    elif (Nrev > 0) and (4 * TOF * LAMBDA != 0):  # (abs(THETA)-pi > 0.5*pi/180)

        checkNconvRSS = 1
        checkNconvOSS = 1
        N3 = 1

        while N3 < 3:

            if Ncase == 0 or checkNconvRSS == 0:

                # - Original Successive Substitution -
                # always converges to xL - small a

                # ----------------------------------
                # Initialize values of y, n, and x

                Y = 1
                N = 0
                N1 = 0
                ERROR = 0
                # CHECKFEAS = 0;
                #             if (TOF-TPAR) <= 1e-3
                #                 X0 = 0;
                #             else
                if checkNconvOSS == 0:
                    X0 = 2 * X0
                    checkNconvOSS = 1
                    # see p. 11 USING BATTIN METHOD TO OBTAIN
                    # MULTIPLE-REVOLUTION LAMBERT'S SOLUTIONS - Shen, Tsiotras
                elif checkNconvRSS == 0:
                    X0 = X0  # X0 is taken from the RSS
                else:
                    X0 = L

                X = -1.e8

                # ----------------------------------
                # Begin iteration

                # ---> CL: 26/01/2009,Matteo Ceriotti
                #   Changed absolute tolerance into relative tolerance here
                #   below.
                while (abs(X0 - X) >= abs(X) * TOL + TOL) and (N <= nitermax):
                    N = N + 1
                    X = X0
                    ETA = X / (sqrt(1 + X) + 1) ** 2
                    CHECKFEAS = 1

                    # ----------------------------------
                    # Compute x by means of an algorithm devised by
                    # Gauticci for evaluating continued fractions by the
                    # 'Top Down' method

                    DELTA = 1
                    U = 1
                    SIGMA = 1
                    M1 = 0

                    while abs(U) > TOL and M1 <= nitermax:
                        M1 = M1 + 1
                        GAMMA = (M1 + 3) ** 2 / (4 * (M1 + 3) ** 2 - 1)
                        DELTA = 1 / (1 + GAMMA * ETA * DELTA)
                        U = U * (DELTA - 1)
                        SIGMA = SIGMA + U

                    C1 = 8 * (sqrt(1 + X) + 1) / (3 + 1 / (5 + ETA + (9 * ETA / 7) * SIGMA))

                    # ----------------------------------
                    # Compute H1 and H2

                    if N == 1:
                        DENOM = (1 + 2 * X + L) * (3 * C1 + X * C1 + 4 * X)
                        H1 = (L + X) ** 2 * (C1 + 1 + 3 * X) / DENOM
                        H2 = M * (C1 + X - L) / DENOM
                    else:
                        QR = sqrt(L1 ** 2 + M / Y ** 2)
                        XPLL = QR - L1
                        LP2XP1 = 2 * QR
                        DENOM = LP2XP1 * (3 * C1 + X * C1 + 4 * X)
                        H1 = ((XPLL ** 2) * (C1 + 1 + 3 * X)) / DENOM
                        H2 = M * (C1 + X - L) / DENOM

                    H3 = M * Nrev * pi / (4 * X * sqrt(X))
                    H2 = H3 + H2

                    B = 27 * H2 / (4 * (1 + H1) ** 3)
                    U = -B / (2 * (sqrt(B + 1) + 1))

                    # ----------------------------------
                    # Compute the continued fraction expansion K(u)
                    # by means of the 'Top Down' method

                    # Y can be computed finding the roots of the formula and selecting
                    # the real one:
                    # y^3 - (1+h1)*y^2 - h2 = 0     (7.113) Battin
                    #
                    # Ycami_ = roots([1 -1-H1 0 -H2])
                    # kcami = find( abs(imag(Ycami_)) < eps );
                    # Ycami = Ycami_(kcami)

                    DELTA = 1
                    U0 = 1
                    SIGMA = 1
                    N1 = 0

                    while N1 < nitermax and abs(U0) >= TOL:
                        if N1 == 0:
                            GAMMA = 4 / 27
                            DELTA = 1 / (1 - GAMMA * U * DELTA)
                            U0 = U0 * (DELTA - 1)
                            SIGMA = SIGMA + U0
                        else:
                            for I8 in [1, 2]:
                                if I8 == 1:
                                    GAMMA = 2 * (3 * N1 + 1) * (6 * N1 - 1) / (9 * (4 * N1 - 1) * (4 * N1 + 1))
                                else:
                                    GAMMA = 2 * (3 * N1 + 2) * (6 * N1 + 1) / (9 * (4 * N1 + 1) * (4 * N1 + 3))
                                DELTA = 1 / (1 - GAMMA * U * DELTA)
                                U0 = U0 * (DELTA - 1)
                                SIGMA = SIGMA + U0

                        N1 = N1 + 1

                    KU = (SIGMA / 3) ** 2
                    Y = ((1 + H1) / 3) * (2 + sqrt(B + 1) / (1 - 2 * U * KU))  # Y = Ycami
                    if Y > sqrt(M / L):
                        if optionsLMR == 2:
                            raise Warning("lambertMR:SuccessiveSubstitutionDiverged",
                                          "Original Successive Substitution is diverging\n"
                                          "-> Reverse Successive Substitution used to find the proper XO.")

                        checkNconvOSS = 0
                        break

                    X0 = sqrt(((1 - L) / 2) ** 2 + M / Y ** 2) - (1 + L) / 2
                    # fprintf('N: %d X0: %.14f\n',N,X0);

                # When 2 solutions exist (small and big a), the previous loop
                # must either converge or diverge because Y > sqrt(M/L) at some
                # point. Thus, the upper bound on the number of iterations
                # should not be necessary. Though, nothing can be said in the
                # case tof<tofmin and so no solution exist. In this case, an
                # upper bound on number of iterations could be needed.

                if N >= nitermax:  # Checks if previous loop ended due to maximum number of iterations
                    if optionsLMR == 2:
                        raise Warning("lambertMR:SuccessiveSubstitutionExceedMaxIter",
                                      "Original Successive Substitution exceeded max number of iteration\n"
                                      "-> Reverse Successive Substitution used to find the proper XO.")
                    checkNconvOSS = 0

            if (Ncase == 1 or checkNconvOSS == 0) and not (checkNconvRSS == 0 and checkNconvOSS == 0):

                # - Reverse Successive Substitution -
                # always converges to xR - large a

                # ----------------------------------
                # Initialize values of y, n, and x

                N = 0
                N1 = 0
                ERROR = 0
                # CHECKFEAS=0
                if checkNconvRSS == 0:
                    X0 = X0 / 2  # XL/2
                    checkNconvRSS = 1
                    # see p. 11 USING BATTIN METHOD TO OBTAIN
                    # MULTIPLE-REVOLUTION LAMBERT'S SOLUTIONS - Shen, Tsiotras
                elif checkNconvOSS == 0:
                    X0 = X0  # X0 is taken from the OSS
                else:
                    X0 = L

                X = -1.e8

                # ----------------------------------
                # Begin iteration

                # ---> CL: 26/01/2009, Matteo Ceriotti
                #   Changed absolute tolerance into relative tolerance here
                #   below.
                while (abs(X0 - X) >= abs(X) * TOL + TOL) and (N <= nitermax):
                    N = N + 1
                    X = X0
                    CHECKFEAS = 1

                    Y = sqrt(M / ((L + X) * (1 + X)))  # y1 in eq. (8a) in Shen, Tsiotras

                    if Y < 1:
                        if optionsLMR == 2:
                            raise Warning("lambertMR:SuccessiveSubstitutionDiverged",
                                          "Reverse Successive Substitution is diverging\n"
                                          "-> Original Successive Substitution used to find the proper XO.")

                        checkNconvRSS = 0
                        break

                    # ---> CL: 27/01/2009, Matteo Ceriotti
                    #   This is the Newton-Raphson method suggested by USING
                    #   BATTIN METHOD TO OBTAIN MULTIPLE-REVOLUTION LAMBERT'S
                    #   SOLUTIONS - Shen, Tsiotras

                    # To assure the Newton-Raphson method to be convergent
                    Erss = 2 * atan(sqrt(X))
                    temp_h_E, temp_dh_E = h_E(Erss, Y, M, Nrev)
                    while temp_h_E < 0:
                        Erss = Erss / 2

                    Nnew = 1
                    Erss_old = -1.e8

                    # The following Newton-Raphson method should always
                    # converge, given the previous first guess choice,
                    # according to the paper. Therefore, the condition on
                    # number of iterations should not be neccesary. It could be
                    # necessary for the case tof < tofmin.
                    while (abs(Erss - Erss_old) >= abs(Erss) * TOL + TOL) and Nnew < nitermax:
                        Nnew = Nnew + 1
                        h, dh = h_E(Erss, Y, M, Nrev)
                        Erss_old = Erss
                        Erss = Erss - h / dh
                        # fprintf('Nnew: %d Erss: %.16f h_E: %.16f\n',Nnew,Erss,h);

                    if Nnew >= nitermax:
                        if optionsLMR != 0:
                            raise Warning("lambertMR:NewtonRaphsonIterExceeded",
                                          "Newton-Raphson exceeded max iterations.")

                    X0 = tan(Erss / 2) ** 2

            if checkNconvOSS == 1 and checkNconvRSS == 1:
                break

            if checkNconvRSS == 0 and checkNconvOSS == 0:
                if optionsLMR != 0:
                    raise Warning('lambertMR:SuccessiveSubstitutionDiverged',
                                  'Both Original Successive Substitution and Reverse '
                                  'Successive Substitution diverge because Nrev > NrevMAX.\n'
                                  'Work in progress to calculate NrevMAX.')

                ERROR = 3
                A = 0
                P = 0
                E = 0
                VI = np.array([0, 0, 0])
                VF = np.array([0, 0, 0])
                TPAR = 0
                THETA = 0
                return A, P, E, ERROR, VI, VF, TPAR, THETA

            N3 = N3 + 1

        if N3 == 3:
            if optionsLMR != 0:
                raise Warning('lambertMR:SuccessiveSubstitutionDiverged',
                              'Either Original Successive Substitution or Reverse '
                              'Successive Substitution is always diverging\n'
                              'because Nrev > NrevMAX or because large-a solution = small-a solution (limit case).\n'
                              'Work in progress to calculate NrevMAX.\n')

            ERROR = 3
            A = 0
            P = 0
            E = 0
            VI = np.array([0, 0, 0])
            VF = np.array([0, 0, 0])
            TPAR = 0
            THETA = 0
            return A, P, E, ERROR, VI, VF, TPAR, THETA

    # ----------------------------------
    # Compute the velocity vectors

    if CHECKFEAS == 0:
        ERROR = 1
        A = 0;
        P = 0;
        E = 0;
        VI = [0, 0, 0];
        VF = [0, 0, 0];
        TPAR = 0;
        THETA = 0;
        A = 0
        P = 0
        E = 0
        VI = np.array([0, 0, 0])
        VF = np.array([0, 0, 0])
        TPAR = 0
        THETA = 0
        return A, P, E, ERROR, VI, VF, TPAR, THETA

    if N1 >= nitermax or N >= nitermax:
        ERROR = 4
        if optionsLMR != 0:
            raise Warning('Lambert algorithm has not converged, maximum number of iterations exceeded.')

        A = 0
        P = 0
        E = 0
        VI = np.array([0, 0, 0])
        VF = np.array([0, 0, 0])
        TPAR = 0
        THETA = 0
        return A, P, E, ERROR, VI, VF, TPAR, THETA

    CONST = M * S * (1 + LAMBDA) ** 2
    A = CONST / (8 * X0 * Y ** 2)

    R11 = (1 + LAMBDA) ** 2 / (4 * TOF * LAMBDA)
    S11 = Y * (1 + X0)
    T11 = (M * S * (1 + LAMBDA) ** 2) / S11

    VI[:] = -R11 * (S11 * (RI[:] - RF[:]) - T11 * RI[:] / RIM)
    VF[:] = -R11 * (S11 * (RI[:] - RF[:]) + T11 * RF[:] / RFM)

    P = (2 * RIM * RFM * Y ** 2 * (1 + X0) ** 2 * sin(THETA / 2) ** 2) / CONST
    E = sqrt(1 - P / A)

    return A, P, E, ERROR, VI, VF, TPAR, THETA
