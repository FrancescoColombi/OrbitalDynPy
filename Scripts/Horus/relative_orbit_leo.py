# Python libs
# import numpy as np
# import pandas as pd
# import scipy as sp

# from astropy import constants as const
# from astropy import units as u
# from poliastro.bodies import Earth, Moon, Sun
# from poliastro.twobody import Orbit

# Project libs
from src.GroundTrack import *
from src.OrbitPropagator.R2BP import *
from src.SunSynch import *
from src.Utilities.KeplerianParameters import *
from src.Utilities.TimeConversion import *
from src.Utilities.spice_tools import *

R_earth = earth["radius"]
mu_earth = earth["mu"]

# Sidereal time of Earth Rotation [sec]
ST_earth_rot = (23 + (56 + 4.09 / 60) / 60) / 24 * 86400
# Earth rotation rate [rad/sec]
omega_earth = 2 * np.pi / ST_earth_rot


"""REFERENCE ORBIT OF MOTHER MISSION"""
# Orbit parameters
altitude = 600.
a = R_earth + altitude
# a = 26600
eccentricity = 0.0
# incl = inclination_sunsynch(a, eccentricity)
incl = 5.2
Omega = 10.0
omega = 0.0
theta = 0.0

kp0 = [a, eccentricity, incl, Omega, omega, theta]
rr0, vv0 = kp2rv(kp0, mu_earth)
print('Keplerian parameter:     {0}'.format(kp0))
print('Initial position:        {0} km'.format(rr0))
print('Initial velocity:        {0} km/s'.format(vv0))
T_orb = 2 * np.pi * np.sqrt(a ** 3 / mu_earth)
print('Orbital period:          {0} min'.format(T_orb / 60))
print('Pericenter:              {0} km'.format(a * (1 - eccentricity)))
print('Apocenter:               {0} km'.format(a * (1 + eccentricity)))

# Reference time
# kernel_dir = "D:/Documents/Francesco/Space_Engineering/spice_kernels/"
# meta_kernel = kernel_dir + 'meta_kernel.tm'
# spice.furnsh(meta_kernel)
load_spice_kernel()
date0 = "2025 jan 14 15:22:40"
jd0 = spice.utc2et(date0) / 86400
close_spice()

# Reference Prime Meridian Sidereal Time
GMST_0 = jd2GMST(jd0)

# Orbit propagation
X0 = np.hstack((rr0, vv0))
t0 = 0
tf = 3 * T_orb
# tf = 86400 * 5
t_out = np.arange(t0, tf, 20)
perturbations = null_perturbation()
perturbations["J2"] = False
orbit = OrbitPropagatorR2BP(X0, t_out, earth, perts=perturbations)
rr_orb = orbit.rr_out
vv_orb = orbit.vv_out
orbit.plot_3D(show_plot=False)
orbit.plot_kp(show_plot=True)

"""ORBITAL PROPAGATION CUBESAT"""
"""DELTA INITIAL CONDITION"""
dcm_lvlh0 = lvlh_framebuilder(X0)  # dcm transformation from inertial frame to lvlh frame
delta_rr_deploy = np.array([0, 0, 0.010])  # initial delta position = 10 m along +R-bar (towards Earth)
# delta_rr_deploy = np.array([-0.010, 0, 0])  # initial delta position = 10 m along -V-bar (opposite orbit direction)
# delta_rr_deploy = np.array([0, 0.0100, 0])  # initial delta position = 10 m along +H-bar (ortogonal to orbit)
delta_rr0 = np.dot(np.transpose(dcm_lvlh0), delta_rr_deploy)
delta_vv_deploy = np.array([0, 0, 0.00005])  # initial delta position = 0.05 m/s along H-bar (towards Earth)
# delta_vv_deploy = np.array([-0.0001, 0, 0])  # initial delta position = 0.1 m/s along -V-bar (ortogonal to orbit)
# delta_vv_deploy = np.array([0, 0, 0])  # initial delta position = 0.1 m/s along H-bar (ortogonal to orbit)
delta_vv0 = np.dot(np.transpose(dcm_lvlh0), delta_vv_deploy)
rr0_cubesat = rr0 + delta_rr0
vv0_cubesat = vv0 + delta_vv0

X0_cubesat = np.hstack((rr0_cubesat, vv0_cubesat))
orbit_cubesat = OrbitPropagatorR2BP(X0_cubesat, t_out, earth, perts=perturbations)
rr_cubesat = orbit_cubesat.rr_out
vv_cubesat = orbit_cubesat.vv_out
#orbit_cubesat.plot_3D(show_plot=False)
#orbit_cubesat.plot_kp(show_plot=True)

"""RELATIVE MOTION IN EQUATORIAL FRAME"""
rr_delta = rr_cubesat - rr_orb
fig_rel_motion_abs = plt.figure()
ax_rel_motion_abs = fig_rel_motion_abs.add_subplot(projection='3d')
ax_rel_motion_abs.plot(rr_delta[:, 0], rr_delta[:, 1], rr_delta[:, 2], lw=1, label='Relative Motion')
ax_rel_motion_abs.plot(rr_delta[0, 0], rr_delta[0, 1], rr_delta[0, 2], '.', label='Initial position')
ax_rel_motion_abs.plot(0, 0, 0, '.', label='Target')
# set plot equal apect ration
ax_rel_motion_abs.set_aspect('equal')
ax_rel_motion_abs.set_xlabel("x - equatorial frame [km]")
ax_rel_motion_abs.set_ylabel("y - equatorial frame [km]")
ax_rel_motion_abs.set_zlabel("z - equatorial frame [km]")
ax_rel_motion_abs.set_title("Relative motion in Equatorial Frame")
plt.legend()


"""RELATIVE MOTION IN LVLH FRAME"""
#dcm_lvlh_list = []
rr_delta_lvlh = np.empty([orbit_cubesat.n_step, 3])
rr_delta_dist = np.empty([orbit_cubesat.n_step, 1])
for n in range(orbit.n_step):
    xx_temp = np.hstack((rr_orb[n, :], vv_orb[n, :]))
    #dcm_lvlh_list[n] = lvlh_framebuilder(xx_temp)
    rr_delta_lvlh[n, :] = np.dot(lvlh_framebuilder(xx_temp), rr_delta[n, :])
    rr_delta_dist[n] = np.linalg.norm(rr_delta[n, :])

fig_rel_motion_lvlh = plt.figure()
ax_rel_motion_lvlh = fig_rel_motion_lvlh.add_subplot(projection='3d')
ax_rel_motion_lvlh.plot(rr_delta_lvlh[:, 0], rr_delta_lvlh[:, 1], rr_delta_lvlh[:, 2], lw=1, label='Relative Motion')
ax_rel_motion_lvlh.plot(rr_delta_lvlh[0, 0], rr_delta_lvlh[0, 1], rr_delta_lvlh[0, 2], '.', label='Initial position')
ax_rel_motion_lvlh.plot(0, 0, 0, '.', label='Target')
# set plot equal aspect ratio
ax_rel_motion_lvlh.set_aspect('equal')
ax_rel_motion_lvlh.set_xlabel("V - bar [km]")
ax_rel_motion_lvlh.set_ylabel("H - bar [km]")
ax_rel_motion_lvlh.set_zlabel("R - bar [km]")
ax_rel_motion_lvlh.set_title("Relative motion in LVLH Frame")
plt.legend()

fig_rel_motion_vrbar = plt.figure()
ax_rel_motion_vrbar = fig_rel_motion_vrbar.add_subplot()
ax_rel_motion_vrbar.plot(rr_delta_lvlh[:, 0], rr_delta_lvlh[:, 2], lw=1, label='Relative Motion')
ax_rel_motion_vrbar.plot(rr_delta_lvlh[0, 0], rr_delta_lvlh[0, 2], '.', label='Initial position')
ax_rel_motion_vrbar.plot(0, 0, '.', label='Target')
# set plot equal aspect ratio
ax_rel_motion_vrbar.set_aspect('equal')
ax_rel_motion_vrbar.set_xlabel("V - bar [km]")
ax_rel_motion_vrbar.set_ylabel("R - bar [km]")
ax_rel_motion_vrbar.set_title("Relative motion in LVLH Frame")
plt.legend()

fig_rel_dist = plt.figure()
ax_rel_dist = fig_rel_dist.add_subplot()
ax_rel_dist.plot(t_out, rr_delta_dist, lw=1, label='Relative Motion')
ax_rel_dist.set_xlabel("time [sec]")
ax_rel_dist.set_ylabel("Distance [km]")
ax_rel_dist.set_title("Relative distance from targer")

plt.show()


"""DELTA INITIAL ORBIT"""
e_cubesat = 0.000005
kp0_cubesat = [a, e_cubesat, incl, Omega, omega, theta]
rr0_cubesat, vv0_cubesat = kp2rv(kp0_cubesat, mu_earth)
print('Keplerian parameter:     {0}'.format(kp0_cubesat))
print('Initial position:        {0} km'.format(rr0_cubesat))
print('Initial velocity:        {0} km/s'.format(vv0_cubesat))
T_orb = 2 * np.pi * np.sqrt(a ** 3 / mu_earth)
print('Orbital period:          {0} min'.format(T_orb / 60))
print('Pericenter:              {0} km'.format(a * (1 - e_cubesat)))
print('Apocenter:               {0} km'.format(a * (1 + e_cubesat)))


X0_cubesat = np.hstack((rr0_cubesat, vv0_cubesat))
orbit_cubesat = OrbitPropagatorR2BP(X0_cubesat, t_out, earth, perts=perturbations)
rr_cubesat = orbit_cubesat.rr_out
vv_cubesat = orbit_cubesat.vv_out
#orbit_cubesat.plot_3D(show_plot=False)
#orbit_cubesat.plot_kp(show_plot=True)

"""RELATIVE MOTION IN EQUATORIAL FRAME"""
rr_delta = rr_cubesat - rr_orb
fig_rel_motion_abs = plt.figure()
ax_rel_motion_abs = fig_rel_motion_abs.add_subplot(projection='3d')
ax_rel_motion_abs.plot(rr_delta[:, 0], rr_delta[:, 1], rr_delta[:, 2], lw=1, label='Relative Motion')
ax_rel_motion_abs.plot(rr_delta[0, 0], rr_delta[0, 1], rr_delta[0, 2], '.', label='Initial position')
ax_rel_motion_abs.plot(0, 0, 0, '.', label='Target')
# set plot equal apect ration
ax_rel_motion_abs.set_aspect('equal')
ax_rel_motion_abs.set_xlabel("x - equatorial frame [km]")
ax_rel_motion_abs.set_ylabel("y - equatorial frame [km]")
ax_rel_motion_abs.set_zlabel("z - equatorial frame [km]")
ax_rel_motion_abs.set_title("Relative motion in Equatorial Frame")
plt.legend()


"""RELATIVE MOTION IN LVLH FRAME"""
# dcm_lvlh_list = []
rr_delta_lvlh = np.empty([orbit_cubesat.n_step, 3])
rr_delta_dist = np.empty([orbit_cubesat.n_step, 1])
for n in range(orbit.n_step):
    xx_temp = np.hstack((rr_orb[n, :], vv_orb[n, :]))
    lvlh_temp = lvlh_framebuilder(xx_temp)
    # dcm_lvlh_list[n] = lvlh_temp
    rr_delta_lvlh[n, :] = np.dot(lvlh_framebuilder(xx_temp), rr_delta[n, :])
    rr_delta_dist[n] = np.linalg.norm(rr_delta[n, :])

fig_rel_motion_lvlh = plt.figure()
ax_rel_motion_lvlh = fig_rel_motion_lvlh.add_subplot(projection='3d')
ax_rel_motion_lvlh.plot(rr_delta_lvlh[:, 0], rr_delta_lvlh[:, 1], rr_delta_lvlh[:, 2], lw=1, label='Relative Motion')
ax_rel_motion_lvlh.plot(rr_delta_lvlh[0, 0], rr_delta_lvlh[0, 1], rr_delta_lvlh[0, 2], '.', label='Initial position')
ax_rel_motion_lvlh.plot(0, 0, 0, '.', label='Target')
# set plot equal aspect ratio
ax_rel_motion_lvlh.set_aspect('equal')
ax_rel_motion_lvlh.set_xlabel("V - bar [km]")
ax_rel_motion_lvlh.set_ylabel("H - bar [km]")
ax_rel_motion_lvlh.set_zlabel("R - bar [km]")
ax_rel_motion_lvlh.set_title("Relative motion in LVLH Frame")
plt.legend()

fig_rel_motion_vrbar = plt.figure()
ax_rel_motion_vrbar = fig_rel_motion_vrbar.add_subplot()
ax_rel_motion_vrbar.plot(rr_delta_lvlh[:, 0], rr_delta_lvlh[:, 2], lw=1, label='Relative Motion')
ax_rel_motion_vrbar.plot(rr_delta_lvlh[0, 0], rr_delta_lvlh[0, 2], '.', label='Initial position')
ax_rel_motion_vrbar.plot(0, 0, '.', label='Target')
# set plot equal aspect ratio
ax_rel_motion_vrbar.set_aspect('equal')
ax_rel_motion_vrbar.set_xlabel("V - bar [km]")
ax_rel_motion_vrbar.set_ylabel("R - bar [km]")
ax_rel_motion_vrbar.set_title("Relative motion in LVLH Frame")
plt.legend()

fig_rel_dist = plt.figure()
ax_rel_dist = fig_rel_dist.add_subplot()
ax_rel_dist.plot(t_out, rr_delta_dist, lw=1, label='Relative Motion')
ax_rel_dist.set_xlabel("time [sec]")
ax_rel_dist.set_ylabel("Distance [km]")
ax_rel_dist.set_title("Relative distance from targer")

plt.show()
