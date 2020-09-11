import numpy as np


def GroundTrack(tt, rr, t_0, PMST_0, omega_planet):
    """
    This function returns the ground track over a planet surface given input time-position coordinates

    :param tt:              [sec]       Time past from reference time t_0 for PMST_0 estimation
                                        Array size = [n]
    :param rr:              [length]    Position vectors in Equatorial Frame
                                        Array size = [3, n]
    :param t_0:             [sec]       Reference time
    :param PMST_0:          [h]         Prime Meridion Sidereal Time is the time angle of the Prime
                                        Meridian of the planet wrt the Vernal Equinox at time t_0
    :param omega_planet:    [rad/sec]   Rotation rate of the planet (sidereal)

    :return alpha:          [deg]       Right ascension in Equatorial Frame. Array size = [n]
    :return delta:          [deg]       Declination in Equatorial Frame. Array size = [n]
    :return latitude:       [deg]       Latitude. Array size = [n]
    :return longitude:      [deg]       Longitude. Array size = [n]
    :return radius:         [length]    Radius. Array size = [n]
    """

    # this function requires rr to be an array with shape [3, n]
    try:
        if np.shape(rr)[0] != 3 and np.shape(rr)[1] != 3:
            raise Exception('Position array shall have shape equal to [3, n] or [n, 3]')
        elif np.shape(rr)[0] != 3 and np.shape(rr)[1] == 3:  # if input rr has shape [n, 3] --> transpose it
            rr = np.transpose(rr)
    except Exception as err:
        print(err)

    # Init output vectors
    alpha = np.zeros(np.size(tt))
    delta = np.zeros(np.size(tt))
    latitude = np.zeros(np.size(tt))
    longitude = np.zeros(np.size(tt))
    radius = np.zeros(np.size(tt))

    # Initial Prime Meridian angle of the planet [rad]
    theta_0 = PMST_0 * np.pi / 12

    for n in range(np.size(tt)):
        r = np.linalg.norm(rr[:, n])
        radius[n] = r

        # Compute declination [rad]
        delta[n] = np.arcsin(rr[2, n] / r)
        # Compute Right Ascension [rad]
        if rr[1, n] >= 0:
            alpha[n] = np.arccos(rr[0, n] / r / np.cos(delta[n]))
        else:
            alpha[n] = 2 * np.pi - np.arccos(rr[0, n] / r / np.cos(delta[n]))

        # Compute angle of the Geodetic Frame wrt Equatorial Frame
        theta = theta_0 + omega_planet * (tt[n] - t_0)
        # Transform position vector from Equatorial to Geodetic Frame (pcpf: planet centered - planet fixed)
        A_equatorial2pcpf = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        rr_pcpf = np.dot(A_equatorial2pcpf, rr[:, n])

        # Compute latitude [rad]
        latitude[n] = np.arcsin(rr_pcpf[2] / r)
        # NOTE: latitude = delta

        # Compute longitude [rad]
        if rr_pcpf[1] >= 0:  # East
            longitude[n] = np.arccos(rr_pcpf[0] / r / np.cos(latitude[n]))
        else:  # West
            longitude[n] = - np.arccos(rr_pcpf[0] / r / np.cos(latitude[n]))

    alpha = alpha * 180 / np.pi
    delta = delta * 180 / np.pi
    latitude = latitude * 180 / np.pi
    longitude = longitude * 180 / np.pi
    return alpha, delta, latitude, longitude, radius


