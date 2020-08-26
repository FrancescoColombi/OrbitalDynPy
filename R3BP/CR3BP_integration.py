from math import *
from numpy import *

from CR3BP_basic import *


def CR3BP(X, t, mu):

    """
    CR3BP Compute the differential of the CR3BP state vector.

    :param X:
    :param t:
    :param mu:
    :return:
    """

    # dX vector preallocation
    dX = zeros(6)

    # dposition
    dX[0] = X[3]
    dX[1] = X[4]
    dX[2] = X[5]

    # dvelocity
    r1 = sqrt((X[0] + mu) ** 2 + X[1] ** 2 + X[2] ** 2)
    r2 = sqrt((X[0] - 1 + mu) ** 2 + X[1] ** 2 + X[2] ** 2)

    dX[3] = X[0] + 2 * X[4] - (1 - mu) * (X[0] + mu) / r1 ** 3 - mu * (X[0] - 1 + mu) / r2 ** 3
    dX[4] = X[1] - 2 * X[3] - (1 - mu) * X[1] / r1 ** 3 - mu * X[1] / r2 ** 3
    dX[5] = - (1 - mu) * X[2] / r1 ** 3 - mu * X[2] / r2 ** 3

    return dX


def STM(Y, t, mu):

    """
    STM Compute the differential of both the CR3BP state vector and STM.

    :param Y:
    :param t:
    :param mu:
    :return:
    """

    # Initialization
    X = Y[0:6]
    PHI = reshape(Y[6:42], (6, 6))

    # dX vector preallocation
    dX = zeros(6)

    # dposition
    dX[0] = X[3]
    dX[1] = X[4]
    dX[2] = X[5]

    # dvelocity
    r1 = sqrt((X[0] + mu) ** 2 + X[1] ** 2 + X[2] ** 2)
    r2 = sqrt((X[0] - 1 + mu) ** 2 + X[1] ** 2 + X[2] ** 2)

    dX[3] = X[0] + 2 * X[4] - (1 - mu) * (X[0] + mu) / r1 ** 3 - mu * (X[0] - 1 + mu) / r2 ** 3
    dX[4] = X[1] - 2 * X[3] - (1 - mu) * X[1] / r1 ** 3 - mu * X[1] / r2 ** 3
    dX[5] = - (1 - mu) * X[2] / r1 ** 3 - mu * X[2] / r2 ** 3

    # Jacobian matrix of the state derivative vector dX
    A = JacobianMatrix(X[0:3], mu)

    # State transition matrix differential equation
    dPHI = A @ PHI

    # Build the output derivative vector
    dPHI = reshape(dPHI, (1, 36))
    dY = concatenate((dX, dPHI), axis=None)

    return dY
