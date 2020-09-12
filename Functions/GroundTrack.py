import numpy as np
import matplotlib.pyplot as plt


def ground_track(tt, rr, t_0, PMST_0, omega_planet, deg=True):
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
    :param deg                          Bool. Return angles in degrees

    :return alpha:          [deg]       Right ascension in Equatorial Frame. Array size = [n]
    :return delta:          [deg]       Declination in Equatorial Frame. Array size = [n]
    :return latitude:       [deg]       Latitude. Array size = [n]
    :return longitude:      [deg]       Longitude. Array size = [n]
    :return radius:         [length]    Range distance. Array size = [n]
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
    alpha = np.zeros(len(tt))
    delta = np.zeros(len(tt))
    latitude = np.zeros(len(tt))
    longitude = np.zeros(len(tt))
    r_norm = np.zeros(len(tt))

    # Initial Prime Meridian angle of the planet [rad]
    theta_0 = PMST_0 * np.pi / 12

    for n in range(len(tt)):
        r = np.linalg.norm(rr[:, n])
        r_norm[n] = r

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

    if deg:
        alpha = alpha * 180 / np.pi
        delta = delta * 180 / np.pi
        latitude = latitude * 180 / np.pi
        longitude = longitude * 180 / np.pi

    return alpha, delta, latitude, longitude, r_norm


def plot_ground_track(coords, labels=None, show_plot=True, colors=['b', 'r', 'g', 'y', 'm'],
                      cities=[], save_plot=False, filename='groundtrack.png', dpi=300):
    """
    This functions plot the ground track of each orbit given as input

    :param coords:      List of 2D arrays with [lat, long] of each orbit
    :param labels:      List of labels of each orbit
    :param show_plot:   Bool
    :param colors:      List of colors of each orbit
    :param cities:      List of cities with shape [lat, long, city_name, city_color]
    :param save_plot:   Bool
    :param filename:    String
    :param dpi:         Resolution

    :return:
    """

    # init figure
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot()

    # load coastline coords [long, lat]
    coast_coords = np.genfromtxt('/Francesco/OrbitalDynPy/Functions/Utilities/coastlines.csv', delimiter=',')
    # plot coastline
    ax.plot(coast_coords[:, 0], coast_coords[:, 1], 'ko', markersize=0.2)

    # plots orbits
    for n in range(len(coords)):
        if labels is None:
            label = 'orbit #' + str(n)
        else:
            label = labels[n]

        # plot starting point and ground track
        ax.plot(coords[n][0, 1], coords[n][0, 0], colors[n] + 'o')
        ax.plot(coords[n][:, 1], coords[n][:, 0], colors[n] + 'o', markersize=1.5, label=label)

    # plot cities
    for city in cities:
        city_coords = city[:2]
        city_name = city[2]
        city_color = city[3]
        ax.plot([city_coords[1]], [city_coords[0]], city_color + 'o', markersize=3)
        ax.annotate(city_name, [city_coords[1], city_coords[0]],
                    textcoords='offset points', xytext=(0, 2),
                    ha='center', color=city_color, fontsize='small')


    # plot settings
    ax.grid(linestyle='dotted')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xticks(np.arange(-180, 180, 20))
    ax.set_yticks(np.arange(-90, 90, 10))
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude [degrees]')
    ax.set_ylabel('Latitude [degrees]')
    ax.legend()

    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(filename, dpi=dpi)

    return fig, ax


"""
def city_dict():
    with open('/Francesco/OrbitalDynPy/Functions/Utilities/world_cities.csv', 'r') as f:
        lines = f.readlines()

    header = lines[0]

    cities = {}

    for line in lines:
        line = line.split(',')

        # try create a new dictionary for new city
        try:
            # city name and lat/long coords
            cities[line[1]] = [float(line[2]), float(line[3])]
        except:
            pass

    return cities
"""
