import numpy as np


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


# Rotational Dynamics driven by gravity gradient and considering rigid body


# Relative motion

