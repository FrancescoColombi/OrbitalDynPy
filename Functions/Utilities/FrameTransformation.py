import numpy as np
from numpy import sin, cos

def eq2ecef(rr, tspan, t_0, PMST_0, omega_planet):
    # init output
    rr_ecef = np.zeros([len(tspan), 3])
    C_eq2ecef = np.zeros([len(tspan), 3, 3])

    # Initial Prime Meridian angle of the planet [rad]
    theta_0 = PMST_0 * np.pi / 12

    for n in range(len(tspan)):
        # Compute angle of the Geodetic Frame wrt Equatorial Frame
        theta = theta_0 + omega_planet * (tspan[n] - t_0)

        # Transform position vector from Equatorial to Geodetic Frame (pcpf: planet centered - planet fixed)
        A_equatorial2pcpf = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        C_eq2ecef[n, :, :] = A_equatorial2pcpf
        rr_ecef[n, :] = np.dot(A_equatorial2pcpf, rr[:, n])

    return rr_ecef, C_eq2ecef


def eq2latlong(rr, tspan, t_0, PMST_0, omega_planet):
    # init output
    lat_long = np.zeros([len(tspan), 2])
    r_norm = np.zeros(len(tspan))

    # frame transformation to ecef
    rr_ecef, _ = eq2ecef(rr, tspan, t_0, PMST_0, omega_planet)

    for n in range(len(tspan)):
        r_norm[n] = np.linalg.norm(rr[n, :])

        # Compute latitude [rad]
        latitude = np.arcsin(rr_ecef[2] / r_norm[n])
        # NOTE: latitude = delta

        # Compute longitude [rad]
        if rr_ecef[1] >= 0:  # East
            longitude = np.arccos(rr_ecef[0] / r_norm[n] / np.cos(latitude))
        else:  # West
            longitude = - np.arccos(rr_ecef[0] / r_norm[n] / np.cos(latitude))

        lat_long[n, :] = [latitude, longitude]
    return lat_long, r_norm
