import numpy as np
import matplotlib.pyplot as plt
from src.Utilities.FrameTransformation import *
import os

COASTLINES_COORDINATES_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           os.path.join('Utilities', 'texture', 'coastlines.csv'))

EARTH_SURFACE_IMAGE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   os.path.join('Utilities', 'texture', 'Earth.jpg'))


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

    :return: alpha          [deg]       Right ascension in Equatorial Frame. Array size = [n]
    :return: delta          [deg]       Declination in Equatorial Frame. Array size = [n]
    :return: latitude       [deg]       Latitude. Array size = [n]
    :return: longitude      [deg]       Longitude. Array size = [n]
    :return: radius         [length]    Range distance. Array size = [n]
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
    alpha = np.empty(len(tt))
    delta = np.empty(len(tt))
    latitude = np.empty(len(tt))
    longitude = np.empty(len(tt))
    r_norm = np.empty(len(tt))

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


def plot_ground_track(coords, args={}, ground_stations=None):
    """
    This functions plot the ground track of each orbit given as input

    :param coords:              List of 2D arrays with [lat, long] of each orbit
    :param args:                List of arguments of ground track plot settings
    :param labels:              List of labels of each orbit
    :param show_plot:           Bool
    :param colors:              List of colors of each orbit
    :param ground_stations:     List of ground station with shape [lat, long, city_name, city_color]
    :param visibility_angle:    Minimum angle of visibility from the ground station
    :param save_plot:           Bool
    :param filename:            String
    :param dpi:                 Resolution

    :return:
    """
    _args = {
        'fig_size': (16, 8),
        'labels': None,
        'colors': ['b', 'r', 'g', 'y', 'm'],
        'show_coasline': COASTLINES_COORDINATES_FILE,
        'show_surface': EARTH_SURFACE_IMAGE,
        'show_plot': False,
        'save_plot': False,
        'filename': "groundtrack.png",
        'dpi': 300,
    }
    for key in args.keys():
        _args[key] = args[key]

    if ground_stations is None:
        ground_stations = []

    # init figure
    fig = plt.figure(figsize=_args['fig_size'])
    ax = fig.add_subplot()

    if _args['show_coasline'] is not None:
        # load coastline coords [long, lat]
        coast_coords = np.genfromtxt(_args['show_coasline'], delimiter=',')
        # plot coastline
        ax.plot(coast_coords[:, 0], coast_coords[:, 1], 'k.', markersize=0.5)

    if _args['show_surface'] is not None:
        ax.imshow(plt.imread(_args['show_surface']), extent=[-180, 180, -90, 90])

    # plots orbits
    for n in range(len(coords)):
        if _args['labels'] is None:
            label = 'orbit #' + str(n + 1)
        else:
            label = _args['labels'][n]

        # plot starting point and ground track
        ax.plot(coords[n][0, 1], coords[n][0, 0], _args['colors'][n] + '.', markersize=3, label=label)
        ax.plot(coords[n][:, 1], coords[n][:, 0], _args['colors'][n] + '.', markersize=1.5)

    # plot ground station and their visibility area
    for city in ground_stations:
        city_coords = city[:2]
        city_name = city[2]
        city_color = city[3]
        ax.plot(city_coords[1], city_coords[0], city_color + 'o', markersize=3)
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

    if _args['show_plot']:
        plt.show()

    if _args['save_plot']:
        plt.savefig(_args['filename'], dpi=_args['dpi'])

    return fig, ax


def ground_station_visibility(tt, rr, gs_coord, r_gs, t_0, pmst_0, omega_planet, theta_aperture):
    """
    Return if the spacecraft/target is in visibility from a given ground station
    :param tt:              [sec]       Time past from reference time t_0 for PMST_0 estimation
                                        Array size = [n]
    :param rr:              [length]    Position vectors in Equatorial Frame
                                        Array size = [3, n]
    :param gs_coord:        [deg]       Latitude and Longitude of the Ground Station [lat, long]
    :param r_gs:            [km]        Distance from CoM of the primary of the Ground Station [km]
    :param t_0:             [sec]       Reference time
    :param pmst_0:          [h]         Prime Meridion Sidereal Time is the time angle of the Prime
                                        Meridian of the planet wrt the Vernal Equinox at time t_0
    :param omega_planet:    [rad/sec]   Rotation rate of the planet (sidereal)
    :param theta_aperture   [rad]       Visibility aperture angle

    :return: visibility, rr_lh, vis_window.
             visibility     [-]         Visibility vector vs tt (visibility[i] = 1 if in visibility at time = tt[i])
             rr_topo        [km]        Position vector in the Ground Station Topocentric Frame
                                        [x: North, y: West, z: Zenith]
             vis_window     [sec]       List Array = [n][1, 2] where n = number of times spacecraft is in visibility
                                        vis_window = [initial time in visibility, final time in visibility]
    """

    # this function requires rr to be an array with shape [3, n]
    try:
        if np.shape(rr)[0] != 3 and np.shape(rr)[1] != 3:
            raise Exception('Position array shall have shape equal to [3, n] or [n, 3]')
        elif np.shape(rr)[0] != 3 and np.shape(rr)[1] == 3:  # if input rr has shape [n, 3] --> transpose it
            rr = np.transpose(rr)
    except Exception as err:
        print(err)

    visibility = np.zeros(len(tt), dtype=int)
    rr_lh = np.empty([3, len(tt)])
    rr_topo = np.empty([3, len(tt)])

    gs_lat = gs_coord[0] * np.pi / 180
    gs_long = gs_coord[1] * np.pi / 180
    # Initial Prime Meridian angle of the planet [rad]
    theta_0 = pmst_0 * np.pi / 12
    # aperture angle from deg to rad
    theta_aperture = theta_aperture * np.pi / 180

    # Position of the Ground station in the Primary centered Local horizon frame
    rr_gs_lh = np.array([r_gs, 0, 0])  # Local Horizon = x: Zenith, y: East, z: North
    zenith = np.array([0, 0, 1])  # Topocentric Frame  = x: North, y: West, z: Zenith
    in_visibility = False
    vis_window = []
    vis_entrance = 0
    vis_exit = 0

    for i_t in range(len(tt)):
        # Transform the coordinates from Primary Centric Equatorial Frame to Local Horizon Frame of the Ground Station
        # Compute angle of the Geodetic Frame wrt Equatorial Frame
        theta = gs_long + theta_0 + omega_planet * (tt[i_t] - t_0)
        T_theta = T_Rz(theta)
        T_phi = T_Ry(-gs_lat)
        # Local Horizon:
        # x: Zenith, y: East, z: North
        T_equatorial2localhorizon = T_phi @ T_theta

        # from Inertial Equatorial Primary Centric frame to Local Horizon Primary Centric frame
        rr_lh_i = np.dot(T_equatorial2localhorizon, rr[:, i_t])
        # translate to Ground Station Centric reference Frame
        rr_lh_i = rr_lh_i - rr_gs_lh
        rr_lh[:, i_t] = rr_lh_i
        rr_topo[0, i_t] = rr_lh_i[2]
        rr_topo[1, i_t] = - rr_lh_i[1]
        rr_topo[2, i_t] = rr_lh_i[0]
        u_rr_lh = rr_topo[:, i_t] / np.linalg.norm(rr_topo[:, i_t])
        if np.dot(u_rr_lh, zenith) >= np.cos(theta_aperture):
            visibility[i_t] = 1
            if not in_visibility:
                in_visibility = True
                vis_entrance = tt[i_t]
        else:
            if in_visibility:
                in_visibility = False
                vis_exit = tt[i_t]
                vis_window.append([vis_entrance, vis_exit])

    return visibility, rr_topo, vis_window


"""
def city_dict():
    with open('/Francesco/OrbitalDynPy/src/Utilities/world_cities.csv', 'r') as f:
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
