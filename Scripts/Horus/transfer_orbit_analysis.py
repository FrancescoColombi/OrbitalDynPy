# Python libraries
# import numpy as np
# import pandas as pd
# import scipy as sp

# from astropy import constants as const
# from astropy import units as u
# from poliastro.bodies import Earth, Moon, Sun
# from poliastro.twobody import Orbit

# Project libraries
from src.GroundTrack import *
from src.SunSynch import *
from src.OrbitPropagator.R2BP import *
from src.Utilities.KeplerianParameters import *
from src.Utilities.FrameTransformation import *
from src.Utilities.SolarSystemBodies import *
from src.Utilities.TimeConversion import *
from src.Utilities.spice_tools import *
from src.Utilities.lamberts_solver import *


R_earth = earth["radius"]
mu_earth = earth["mu"]

"""REFERENCE ORBIT OF MOTHER MISSION"""
# Orbit parameters
altitude = 600.0
a = R_earth + altitude
eccentricity = 0.0
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

# TARGET ORBIT - Orbit propagation
X0 = np.hstack((rr0, vv0))
t0 = 0
tf = T_orb
# tf = 86400 * 5
t_out = np.arange(t0, tf, 30)
perturbations = null_perturbation()
perturbations["J2"] = False
orbit = OrbitPropagatorR2BP(X0, t_out, earth, perts=perturbations)
rr_orb = orbit.rr_out
vv_orb = orbit.vv_out
orbit.plot_3D()
orbit.plot_kp()
plt.show()

"""DELTA INITIAL ORBIT"""
e_cubesat = 0.000005
kp0_cubesat = [a, e_cubesat, incl, Omega, omega, theta+179.999]
rr0_cubesat, vv0_cubesat = kp2rv(kp0_cubesat, mu_earth)
print('Keplerian parameter:     {0}'.format(kp0_cubesat))
print('Initial position:        {0} km'.format(rr0_cubesat))
print('Initial velocity:        {0} km/s'.format(vv0_cubesat))
T_orb = 2 * np.pi * np.sqrt(a ** 3 / mu_earth)
print('Orbital period:          {0} min'.format(T_orb / 60))
print('Pericenter:              {0} km'.format(a * (1 - e_cubesat)))
print('Apocenter:               {0} km'.format(a * (1 + e_cubesat)))


# X0_cubesat = np.hstack((rr0_cubesat, vv0_cubesat))
# orbit_cubesat = OrbitPropagatorR2BP(X0_cubesat, t_out, earth, perts=perturbations)
# rr_cubesat = orbit_cubesat.rr_out
# vv_cubesat = orbit_cubesat.vv_out

ToF = T_orb/2
a_to, p_to, e_to, error_lambert, vv_to_dep, vv_to_arr, tpar, theta = lambertMR(rr0, rr0_cubesat, ToF, mu_earth, Ncase=1, Nrev=0, optionsLMR=0)
print("-----------------------------------")
print("Lambert Solver outputs")
print("Result {0}".format(error_lambert))
print("a_to  = {0} km".format(a_to))
print("p_to  = {0} km".format(p_to))
print("e_to  = {0}".format(e_to))
print("tpar  = {0}".format(tpar))
print("theta = {0}".format(theta))
print("-----------------------------------")
print("")


print("Departure Point")
print('Initial velocity:        {0} km/s'.format(vv0))
print('     TO velocity:        {0} km/s'.format(vv_to_dep))
print('   DeltaV vector:        {0} km/s'.format(vv_to_dep-vv0))
print('          DeltaV:        {0} m/s'.format(np.linalg.norm(vv_to_dep-vv0)*1000))

print("Arrival Point")
print('     TO velocity:        {0} km/s'.format(vv_to_arr))
print('  Final velocity:        {0} km/s'.format(vv0_cubesat))
print('   DeltaV vector:        {0} km/s'.format(vv0_cubesat-vv_to_arr))
print('          DeltaV:        {0} m/s'.format(np.linalg.norm(vv0_cubesat-vv_to_arr)*1000))

kp_to = rv2kp(rr0, vv_to_dep, mu_earth)
print('Keplerian parameter:     {0}'.format(kp_to))
X0_to = np.hstack((rr0, vv_to_dep))
t_t0_to = np.arange(t0, ToF, 30)
orbit_to = OrbitPropagatorR2BP(X0_to, t_t0_to, earth, perts=perturbations)
rr_to = orbit_to.rr_out
vv_to = orbit_to.vv_out
orbit_to.plot_3D()
plt.show()

### RELATIVE MOTION IN TARGET CENTRIC FRAME ###
rr_delta = np.empty([orbit_to.n_step, 3])
for n in range(orbit_to.n_step):
    rr_delta[n, :] = rr_to[n, :] - rr_orb[n, :]
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
ax_rel_motion_abs.set_title("Relative motion in Target Centric Frame")
plt.legend()


### RELATIVE MOTION IN LVLH FRAME ###
# dcm_lvlh_list = []
rr_delta_lvlh = np.empty([orbit_to.n_step, 3])
rr_delta_dist = np.empty([orbit_to.n_step, 1])
for n in range(orbit_to.n_step):
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
ax_rel_dist.plot(t_t0_to, rr_delta_dist, lw=1, label='Relative Motion')
ax_rel_dist.set_xlabel("time [sec]")
ax_rel_dist.set_ylabel("Distance [km]")
ax_rel_dist.set_title("Relative distance from targer")

plt.show()

