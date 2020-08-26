import numpy as np


# Restricted 2-Body Problem
def R2BP_ode(X, t, mu):
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]

    r = np.linalg.norm(X[0:2])

    X_dot = [
        vx,
        vy,
        vz,
        -mu / (r ^ 3) * x,
        -mu / (r ^ 3) * y,
        -mu / (r ^ 3) * z
    ]
    return X_dot

# Restricted 2-Body Problem plus perturbations

# Rotational Dynamics driven by gravity gradient and considering rigid body

# Relative motion

