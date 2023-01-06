import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from spiceypy import spiceypy as spice

# from astropy import constants as const
# from astropy import units as u

# from poliastro.bodies import Earth, Moon, Sun
# from poliastro.twobody import Orbit


from Functions.GroundTrack import *
from Functions.OrbitPropagator.R2BP import OrbitPropagatorR2BP as op
from Functions.OrbitPropagator.R2BP import *
from Functions.Utilities.SolarSystemBodies import *
from Functions.Utilities.KeplerianParameters import *
from Functions.Utilities.TimeConversion import *
from Functions.SunSynch import *

if __name__ == '__main__':
    # Constants
    R_earth = 0.63781600000000E+04
    mu_earth = 0.59736990612667E+25 * 6.67259e-20
    # R_earth = const.R_earth.to('km').value
    # mu_earth = const.GM_earth.to('km3/s2').value

    # Sidereal time of Earth Rotation [sec]
    ST_earth_rot = (23 + (56 + 4.09 / 60) / 60) / 24 * 86400
    # Earth rotation rate [rad/sec]
    omega_earth = 2 * np.pi / ST_earth_rot

    # Orbit parameters
    altitude = 15000.
    a = R_earth + altitude
    # a = 26600
    eccentricity = 0.
    # incl = inclination_sunsynch(a, eccentricity)
    incl = 15
    Omega = 10.0
    omega = 200.0
    theta = 150.0
    kp0 = [a, eccentricity, incl, Omega, omega, theta]
    rr0, vv0 = kp2rv(kp0, mu_earth)
    print('Keplerian parameter:     {0}'.format(kp0))
    print('Initial position:        {0} km'.format(rr0))
    print('Initial velocity:        {0} km/s'.format(vv0))
    T_orb = 2 * np.pi * np.sqrt(a ** 3 / mu_earth)
    print('Orbital period:          {0} min'.format(T_orb / 60))

    # Reference time
    kernel_dir = "D:/Documents/Francesco/Space_Engineering/spice_kernels/"
    meta_kernel = kernel_dir + 'meta_kernel.tm'
    spice.furnsh(meta_kernel)
    date0 = "2020 jan 01 15:22:40"
    jd0 = spice.utc2et(date0) / 86400
    spice.kclear()

    # Reference Prime Meridian Sidereal Time
    GMST_0 = jd2GMST(jd0)

    # Orbit propagation
    X0 = np.hstack((rr0, vv0))
    t0 = 0
    tf = 50 * T_orb
    t_out = np.arange(t0, tf, 30)
    perturbations = null_perturbation()
    perturbations["J2"] = True
    orbit = op(X0, t_out, Earth, perts=perturbations)
    rr_orb = orbit.rr_out
    vv_orb = orbit.vv_out
    orbit.plot_3D(show_plot=False)
    orbit.plot_kp(show_plot=True)

    """# plot orbit
    fig = plt.figure(figsize=[6, 6], tight_layout=True)
    ax = fig.add_subplot(projection='3d')
    ax.plot(rr_orb[0, :], rr_orb[1, :], rr_orb[2, :], lw=1, label='orbit')
    ax.scatter(rr_orb[0, 0], rr_orb[1, 0], rr_orb[2, 0], color='red')
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    maxval = np.abs(rr_orb).max()
    ax.set_xlim([-maxval, maxval])
    ax.set_ylim([-maxval, maxval])
    ax.set_zlim([-maxval, maxval])
    ax.set_title('Earth Centered - Earth Equatorial Frame')
    plt.show()"""

    """# test using plotly for nice plots
    data = {'time': t_out,
            'x':    y_out[:, 0],
            'y':    y_out[:, 1],
            'z':    y_out[:, 2],
            'vx':   y_out[:, 3],
            'vy':   y_out[:, 4],
            'vz':   y_out[:, 5]}
    df = pd.DataFrame(data)
    print(df.head())
    fig_plotly = px.line_3d(df, x='x', y='y', z='z')
    maxval = np.abs(rr_orb).max()
    xratio = np.abs(rr_orb[0, :]).max() / maxval
    yratio = np.abs(rr_orb[1, :]).max() / maxval
    zratio = np.abs(rr_orb[2, :]).max() / maxval
    fig_plotly.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=xratio, y=yratio, z=zratio))
    fig_plotly.show()"""

    # Ground Track
    # t_0 = jd0 * 86400  # [sec]
    t_0 = 0
    tt = t_0 + t_out
    alpha, delta, lat, long, _ = ground_track(tt, rr_orb, t_0, GMST_0, omega_earth)

    ground_station_coord = [45.042236, 9.679320]
    visibility_aperture = 75
    visibility, rr_lh, visibility_window = ground_station_visibility(tt, rr_orb, ground_station_coord, R_earth, t_0,
                                                                     GMST_0, omega_earth, visibility_aperture)

    # print(visibility_window)
    n_visibility = len(visibility_window)
    print("Number of visibility contact = {0}".format(n_visibility))
    for i_n in range(n_visibility):
        contact_period = visibility_window[i_n][1] - visibility_window[i_n][0]
        print("Visibility Contact # {0}".format(i_n))
        print("Contact Visibility Period = {0} min".format(contact_period/60))

    fig0 = plt.figure()
    ax0 = fig0.add_subplot()
    ax0.plot(tt, visibility, markersize=1.5)
    ax0.set_title('Visibility from Ground Station vs. Time')
    ax0.set_ylim(0, 1)
    ax0.grid(True)

    visibility_num = sum(visibility)
    visibility_pos = np.zeros([2, visibility_num])
    visibility_rr_lh = np.zeros([3, visibility_num])
    id_counter = 0
    for i_n in range(len(tt)):
        rr_lh[:, i_n] = rr_lh[:, i_n] / np.linalg.norm(rr_lh[:, i_n])
        if visibility[i_n]:
            visibility_pos[0, id_counter] = lat[i_n]
            visibility_pos[1, id_counter] = long[i_n]
            visibility_rr_lh[:, id_counter] = rr_lh[:, i_n]
            id_counter = id_counter + 1

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.plot(rr_lh[0, :], rr_lh[1, :], rr_lh[2, :], lw=1, label='Trajectory in Local Horizon')
    ax1.plot(visibility_rr_lh[0, :], visibility_rr_lh[1, :], visibility_rr_lh[2, :], 'ro', markersize=1)
    max_val = np.max(np.abs(rr_lh))
    # visibility cone
    theta_circle = np.linspace(-180, 180) * np.pi / 180
    R_a = np.linspace(0, max_val)
    x = np.cos(visibility_aperture * np.pi / 180) * np.outer(np.ones(np.size(theta_circle)), R_a).T
    y = np.sin(visibility_aperture * np.pi / 180) * np.outer(np.cos(theta_circle), R_a).T
    z = np.sin(visibility_aperture * np.pi / 180) * np.outer(np.sin(theta_circle), R_a).T
    ax1.plot_surface(x, y, z, rstride=4, cstride=4)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Zenith [km]')
    ax1.set_ylabel('East [km]')
    ax1.set_zlabel('North [km]')
    ax1.set_title('Local Horizon view from Ground Station')

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot()
    ax2.plot(long, lat, 'bo', markersize=1.5)
    ax2.plot(visibility_pos[1, :], visibility_pos[0, :], 'ro', markersize=1.5)
    ax2.plot(ground_station_coord[1], ground_station_coord[0], 'mo', markersize=3, label='Ground Station')
    ax2.annotate('ground station', [ground_station_coord[1], ground_station_coord[0]], textcoords='offset points',
                 xytext=(0, 2), ha='center', color='m', fontsize='small')
    ax2.set_xlim([-180, 180])
    ax2.set_ylim([-90, 90])
    ax2.set_xticks(np.arange(-180, 180, 20))
    ax2.set_yticks(np.arange(-90, 90, 10))
    ax2.set_aspect('equal')
    ax2.set_aspect('equal')
    ax2.set_xlabel('Longitude [degrees]')
    ax2.set_ylabel('Latitude [degrees]')
    ax2.set_title('Ground Track and Visibility Area')

    """
    delta_t = 1 * 3600
    t_marker = np.arange(t0, tf, delta_t)
    y_marker = odeint(R2BP_dyn, X0, t_marker, args=(mu_earth,), rtol=1e-10, atol=1e-10)
    rr_marker = np.transpose(y_marker[:, :3])
    tt_mark = t_0 + t_marker
    alpha, delta, lat_mark, long_mark, _ = GroundTrack(tt_mark, rr_marker, t_0, GMST_0, omega_earth)
    """

    """
    fig2 = plt.figure(figsize=[10, 6], tight_layout=True)
    ax = fig2.subplots(1, 1)
    ax.scatter(long, lat, c=tt)
    # ax.scatter(long_mark, lat_mark, color='blue')
    ax.scatter(long[0], lat[0], color='red', label='Initial condition')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(-180, 180, 30))
    ax.set_yticks(np.arange(-90, 90, 15))
    plt.grid(ls='--')
    

    plt.show()
    """
    coords = [np.transpose(np.reshape([lat, long], (2, -1)))]
    piacenza = [45.042236, 9.679320, 'Piacenza', 'r']
    # figure1, ax1 = plot_ground_track(coords)
    plot_ground_track(coords, ground_stations=[piacenza])

    print("End of script")
