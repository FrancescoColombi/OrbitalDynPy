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

show_initial_things = False

########################################################################################################################
# REFERENCE ORBIT OF TARGET MISSION #
########################################################################################################################

# Orbit parameters
altitude = 550.0
a = R_earth + altitude
eccentricity = 0.0
incl = 5.2
Omega = 10.0
omega = 0.0
theta = 0.0

kp0 = [a, eccentricity, incl, Omega, omega, theta]
rr0, vv0 = kp2rv(kp0, mu_earth)
print("-----------------------------------")
print("## TARGET ORBIT")
print('Keplerian parameter:     {0}'.format(kp0))
print('Initial position:        {0} km'.format(rr0))
print('Initial velocity:        {0} km/s'.format(vv0))
T_orb = 2 * np.pi * np.sqrt(a ** 3 / mu_earth)
print('Orbital period:          {0} min'.format(T_orb / 60))
print('Pericenter:              {0} km'.format(a * (1 - eccentricity)))
print('Apocenter:               {0} km'.format(a * (1 + eccentricity)))
print("-----------------------------------")
print("")

# TARGET ORBIT - Orbit propagation
X0 = np.hstack((rr0, vv0))
t0 = 0
tf = 3 * T_orb
# tf = 86400 * 5
dt = 60
t_out = np.arange(t0, tf, dt)
perturbations = null_perturbation()
perturbations["J2"] = False
orbit_target = OrbitPropagatorR2BP(X0, t_out, earth, perts=perturbations)
rr_orb = orbit_target.rr_out
vv_orb = orbit_target.vv_out
if show_initial_things:
    orbit_target.plot_3D()
    orbit_target.plot_kp()
    plt.show()

########################################################################################################################
# DEPLOYMENT PHASE AND SAFETY DISTANCING #
########################################################################################################################

# ASSUMPTION
deploy_dist = 0.002  # km
deploy_vel = 0.00005  # km/s
dcm_lvlh0 = lvlh_framebuilder(X0)  # dcm transformation from inertial frame to lvlh frame
# delta_rr_deploy = np.array([0, 0, 1]) * deploy_dist  # initial delta position along +R-bar (towards Earth)
# delta_vv_deploy = np.array([0, 0, 1]) * deploy_vel  # initial delta velocity along R-bar (towards Earth)

delta_rr_deploy = np.array([-1, 0, 0]) * deploy_dist  # initial delta position along -V-bar (opposite orbit direction)
delta_vv_deploy = np.array([-1, 0, 0]) * deploy_vel  # initial delta velocity along -V-bar (opposite orbit direction)

# delta_rr_deploy = np.array([0, 1, 0]) * deploy_dist  # initial delta position along +H-bar (orthogonal to orbit)
# delta_vv_deploy = np.array([0, 1, 0]) * deploy_vel  # initial delta velocity along H-bar (orthogonal to orbit)

delta_rr0 = np.dot(np.transpose(dcm_lvlh0), delta_rr_deploy)
delta_vv0 = np.dot(np.transpose(dcm_lvlh0), delta_vv_deploy)
rr0_cubesat = rr0 + delta_rr0
vv0_cubesat = vv0 + delta_vv0

X0_cubesat = np.hstack((rr0_cubesat, vv0_cubesat))
orbit_deploy = OrbitPropagatorR2BP(X0_cubesat, t_out, earth, perts=perturbations)
rr_cubesat = orbit_deploy.rr_out
vv_cubesat = orbit_deploy.vv_out
if show_initial_things:
    orbit_deploy.plot_3D(show_plot=False)
    orbit_deploy.plot_kp(show_plot=True)
    plt.show()

"""RELATIVE MOTION IN EQUATORIAL FRAME"""
rr_delta = rr_cubesat - rr_orb
fig_rel_motion_abs = plt.figure()
ax_rel_motion_abs = fig_rel_motion_abs.add_subplot(projection='3d')
ax_rel_motion_abs.plot(rr_delta[:, 0], rr_delta[:, 1], rr_delta[:, 2], lw=1, label='Relative Motion')
ax_rel_motion_abs.plot(rr_delta[0, 0], rr_delta[0, 1], rr_delta[0, 2], '.', label='Initial position')
ax_rel_motion_abs.plot(0, 0, 0, '.', label='Target')
# set plot equal aspect ratio
ax_rel_motion_abs.set_aspect('equal')
ax_rel_motion_abs.set_xlabel("x - equatorial frame [km]")
ax_rel_motion_abs.set_ylabel("y - equatorial frame [km]")
ax_rel_motion_abs.set_zlabel("z - equatorial frame [km]")
ax_rel_motion_abs.set_title("Relative motion in Equatorial Frame")
plt.legend()


"""RELATIVE MOTION IN LVLH FRAME"""
#dcm_lvlh_list = []
rr_delta_lvlh = np.empty([orbit_deploy.n_step, 3])
rr_delta_dist = np.empty([orbit_deploy.n_step, 1])
for n in range(orbit_target.n_step):
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

########################################################################################################################
# TRANSFER ORBIT TO OPERATIONAL ORBIT
########################################################################################################################

"""FINAL ORBIT KEPLERIAN PARAMETERS - EXAMPLE"""
delta_peri = 100/1000  # km
delta_apo = 100/1000  # km
a_cubesat = a
e_cubesat = 1 - (a-delta_peri)/a
kp0_cubesat = [a_cubesat, e_cubesat, incl, Omega, omega, theta]
rr0_cubesat, vv0_cubesat = kp2rv(kp0_cubesat, mu_earth)
print("-----------------------------------")
print("## Operative Orbit Keplerian Parameters")
print('Semi major axis:         {0} km'.format(a_cubesat))
print('Eccentricity:            {0}'.format(e_cubesat))
print('Pericenter:              {0} km'.format(a * (1 - e_cubesat)))
print('Apocenter:               {0} km'.format(a * (1 + e_cubesat)))
print("-----------------------------------")
print("")


""" VARIABLE OF TRANSFER ORBIT EVALUATION"""
# ASSUMPTION
# Target = Circular orbit : therefore consider only a reference starting condition
# START CONDITION = Final states of the SAFETY DISTANCING PHASE
rr_parking = rr_cubesat[orbit_deploy.n_step-1, :]
vv_parking = vv_cubesat[orbit_deploy.n_step-1, :]

# ToF time of flight
N_ToF = 25
tof_vect = np.arange(0, T_orb*9/10, T_orb/N_ToF)
N_ToF = np.size(tof_vect)

rr0 = rr_orb[-1, :]
vv0 = vv_orb[-1, :]
X0 = np.hstack((rr0, vv0))
orbit_target_2 = OrbitPropagatorR2BP(X0, tof_vect, earth, perts=perturbations)
rr_orb_2 = orbit_target_2.rr_out
vv_orb_2 = orbit_target_2.vv_out
kp_orb_2 = orbit_target_2.kp_out

# delta_u arrival orbit orientation wrt the target orbit
delta_u_vect = np.arange(0, 360, 15)
N_u = np.size(delta_u_vect)

# init empty list [N_ToF, N_u]
kp_arrival_out = [[0 for j in range(N_u)] for i in range(N_ToF)]
rr_arrival_out = [[0 for j in range(N_u)] for i in range(N_ToF)]
vv_arrival_out = [[0 for j in range(N_u)] for i in range(N_ToF)]
vv_to_dep_out = [[0 for j in range(N_u)] for i in range(N_ToF)]
vv_to_arr_out = [[0 for j in range(N_u)] for i in range(N_ToF)]
delta_v_out = [[0 for j in range(N_u)] for i in range(N_ToF)]
current_min_out = None
min_id_out = [0, 0]
for n_tof in range(1, N_ToF):
    tof_to = tof_vect[n_tof]
    # evaluate pericenter anomaly for the arrival point of the transfer orbit (starting from 0)
    theta_target = tof_to/T_orb * 360  # deg

    kp_target = kp_orb_2[n_tof, :]

    for n_u in range(N_u):
        delta_u = delta_u_vect[n_u]
        # evaluate arrival states at operative orbit
        kp_arrival = kp_target + [0, e_cubesat, 0, 0, delta_u, -delta_u]
        rr_arrival, vv_arrival = kp2rv(kp_arrival, mu_earth)
        # store data
        kp_arrival_out[n_tof][n_u] = kp_arrival
        rr_arrival_out[n_tof][n_u] = rr_arrival
        vv_arrival_out[n_tof][n_u] = vv_arrival
        # tmp = np.linalg.norm(rr_arrival - rr_orb_2[n_tof, :])
        # print(tmp)

        # Lambert's Problem
        a_to, p_to, e_to, error_lambert, vv_to_dep, vv_to_arr, tpar, theta = lambertMR(
            rr_parking, rr_arrival, tof_to, mu_earth, Ncase=0, Nrev=0, optionsLMR=0
        )
        vv_to_dep_out[n_tof][n_u] = vv_to_dep
        vv_to_arr_out[n_tof][n_u] = vv_to_arr
        delta_v_dep = np.linalg.norm(vv_to_dep - vv_parking)
        delta_v_arr = np.linalg.norm(vv_arrival - vv_to_arr)
        delta_v_out[n_tof][n_u] = (delta_v_dep + delta_v_arr)*1000
        if current_min_out is None:
            current_min_out = delta_v_out[n_tof][n_u]
            min_id_out = [n_tof, n_u]
        elif (delta_v_out[n_tof][n_u] < current_min_out) and (delta_v_out[n_tof][n_u] > 0):
            current_min_out = delta_v_out[n_tof][n_u]
            min_id_out = [n_tof, n_u]


X, Y = np.meshgrid(delta_u_vect, tof_vect)
fig_to_cost = plt.figure(figsize=(6, 5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig_to_cost.add_axes([left, bottom, width, height])
# cp = ax.contourf(X, Y, delta_v_out, levels=np.arange(0,100,5))
cp = ax.contour(X, Y, delta_v_out, levels=np.arange(0, 10, 0.25))
# ax.clabel(cp, inline=True, fontsize=10)
ax.set_title('Delta V Transfer Orbit')
ax.set_ylabel('ToF [sec]')
ax.set_xlabel('Orientation [deg]')
cbar = plt.colorbar(cp)
cbar.ax.set_ylabel("$\Delta$V [m/s]")

## display solution

n_tof = min_id_out[0]
n_u = min_id_out[1]
tof = tof_vect[n_tof]
rr_arrival = rr_arrival_out[n_tof][n_u]
vv_arrival = vv_arrival_out[n_tof][n_u]
a_to, p_to, e_to, error_lambert, vv_to_dep, vv_to_arr, tpar, theta = lambertMR(rr_parking, rr_arrival, tof, mu_earth, Ncase=0, Nrev=0, optionsLMR=0)
print("-----------------------------------")
print("## Lambert Solver outputs")
print("Result {0}".format(error_lambert))
print("a_to  = {0} km".format(a_to))
print("p_to  = {0} km".format(p_to))
print("e_to  = {0}".format(e_to))
print("tpar  = {0}".format(tpar))
print("theta = {0}".format(theta))
print("-----------------------------------")
print("")


print("Departure Point")
print('Initial velocity:        {0} km/s'.format(vv_parking))
print('     TO velocity:        {0} km/s'.format(vv_to_dep))
print('   DeltaV vector:        {0} km/s'.format(vv_to_dep-vv_parking))
print('          DeltaV:        {0} m/s'.format(np.linalg.norm(vv_to_dep-vv_parking)*1000))

print("Arrival Point")
print('     TO velocity:        {0} km/s'.format(vv_to_arr))
print('  Final velocity:        {0} km/s'.format(vv_arrival))
print('   DeltaV vector:        {0} km/s'.format(vv_arrival-vv_to_arr))
print('          DeltaV:        {0} m/s'.format(np.linalg.norm(vv_arrival-vv_to_arr)*1000))

kp_to = rv2kp(rr_parking, vv_to_dep, mu_earth)
print('Keplerian parameter:     {0}'.format(kp_to))
X0_to = np.hstack((rr_parking, vv_to_dep))
t_t0_to = np.arange(0, tof, dt)
orbit_to = OrbitPropagatorR2BP(X0_to, t_t0_to, earth, perts=perturbations)
rr_to = orbit_to.rr_out
vv_to = orbit_to.vv_out
orbit_to.plot_3D(title="Transfer Orbit")
plt.show()

rr0 = rr_orb[-1, :]
vv0 = vv_orb[-1, :]
X0 = np.hstack((rr0, vv0))
orbit_target = OrbitPropagatorR2BP(X0, t_t0_to, earth, perts=perturbations)
rr_orb = orbit_target.rr_out
vv_orb = orbit_target.vv_out

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
ax_rel_motion_abs.set_title("Relative motion in Equatorial Frame - Target centered")
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
ax_rel_motion_lvlh.set_title("Relative motion in LVLH Frame - Target centered")
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
ax_rel_motion_vrbar.set_title("Relative motion in LVLH Frame - Target centered")
plt.legend()

fig_rel_dist = plt.figure()
ax_rel_dist = fig_rel_dist.add_subplot()
ax_rel_dist.plot(t_t0_to, rr_delta_dist, lw=1, label='Relative Motion')
ax_rel_dist.set_xlabel("time [sec]")
ax_rel_dist.set_ylabel("Distance [km]")
ax_rel_dist.set_title("Relative distance from targer")

plt.show()

