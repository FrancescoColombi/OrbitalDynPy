from math import *
from numpy import *
from scipy import optimize


def JacobianMatrix(X, mu):

    """
    JACOBIANMATRIX Compute the CR3BP Jacobian matrix.

    :param X:
    :param mu:
    :return:
    """

    I = eye(3)
    O = zeros((3, 3))
    G = GradientMatrix(X, mu)
    K = array([[0, 2, 0],
               [-2, 0, 0],
               [0, 0, 0]])
    A = concatenate((concatenate((O, I), axis=1), concatenate((G, K), axis=1)))

    return A


def GradientMatrix(X, mu):

    """
    GRADIENTMATRIX Compute the CR3BP Gradient matrix.

    :param X:
    :param mu:
    :return:
    """

    r1 = sqrt((X[0] + mu) ** 2 + X[1] ** 2 + X[2] ** 2)
    r2 = sqrt((X[0] - 1 + mu) ** 2 + X[1] ** 2 + X[2] ** 2)

    omegaxx = 1 - (1 - mu) / r1 ** 3 + 3 * (1 - mu) * (X[0] + mu) ** 2 / r1 ** 5 - mu / r2 ** 3 + 3 * mu * (
                X[0] - 1 + mu) ** 2 / r2 ** 5
    omegaxy = 3 * (1 - mu) * (X[0] + mu) * X[1] / r1 ** 5 + 3 * mu * (X[0] + mu - 1) * X[1] / r2 ** 5
    omegaxz = 3 * X[2] * (1 - mu) * (X[0] + mu) / r1 ** 5 + 3 * X[2] * mu * (X[0] - 1 + mu) / r2 ** 5
    omegayy = 1 - (1 - mu) / r1 ** 3 + 3 * (1 - mu) * X[1] ** 2 / r1 ** 5 - mu / r2 ** 3 + 3 * mu * X[1] ** 2 / r2 ** 5
    omegayz = 3 * X[1] * X[2] * (1 - mu) / r1 ** 5 + 3 * X[1] * X[2] * mu / r2 * 5
    omegazz = - (1 - mu) / r1 ** 3 + 3 * X[2] ** 2 * (1 - mu) / r1 ** 5 - mu / r2 ** 3 + 3 * X[2] ** 2 * mu / r2 ** 5

    G = array([[omegaxx, omegaxy, omegaxz],
               [omegaxy, omegayy, omegayz],
               [omegaxz, omegayz, omegazz]])

    return G


def L123(x, mu):

    """
    Collinear points quintic polynomial equation.

    :param x:
    :param mu:
    :return:
    """

    fun = x * (x + mu) * (x - 1 + mu) * abs(x + mu) * abs(x - 1 + mu) - \
          (1 - mu) * (x - 1 + mu) * abs(x - 1 + mu) - \
          mu * (x + mu) * abs(x + mu)

    return fun


def LibrationPoints(mu):

    """
    LIBRATIONPOINTS Give the location of the CR3BP Lagrangian points.

    :param mu:
    :return:
    """

    if mu > 0.5:
        mu = 1 - mu
        print('Reverse Problem for mu')

    L1 = optimize.brenth(L123, -mu, 1 - mu, args=(mu, ))
    L2 = optimize.brenth(L123, 1 - mu, 2, args=(mu, ))
    L3 = optimize.brenth(L123, -2, 0 - finfo(float).eps, args=(mu, ))

    L4x = (1 / 2) - mu
    L4y = sqrt(3) / 2

    return L1, L2, L3, L4x, L4y
