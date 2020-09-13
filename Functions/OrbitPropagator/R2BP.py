import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation
from celluloid import Camera


import poliastro.core.perturbations as pert_fun

from Functions.Utilities.KeplerianParameters import kp2rv, rv2kp
from Functions.Dynamics.R2BP import R2BP_dyn
import Functions.Utilities.SolarSystemBodies as CelBody


def null_perturbation():
    return {
        'J2': False,
        'J3': False,
        'J4': False,
        'J5': False,
        'J6': False,
        'J7': False,
        'aero': False,
        '3rd_body': False,
        '4th_body': False,
        'SRP': False
    }


class OrbitPropagatorR2BP:
    def __init__(self, state0, tspan, primary=CelBody.Earth, coes=False, deg=True, perts=null_perturbation()):
        self.primary = primary
        self.mu = self.primary["mu"]

        if coes:
            self.rr0, self.vv0 = kp2rv(state0, self.mu, deg=deg)
        else:
            self.rr0 = state0[:3]
            self.vv0 = state0[3:]

        self.y0 = self.rr0.tolist() + self.vv0.tolist()
        self.tspan = tspan

        self.perts = perts
        self.n_step = len(self.tspan)

        # initialize variables
        self.y_out = np.zeros([self.n_step, 6])
        self.rr_out = np.zeros([self.n_step, 3])
        self.vv_out = np.zeros([self.n_step, 3])

        # propagate orbit
        self.propagate_orbit()

    def dyn_ode(self, t, y):
        y_dot = R2BP_dyn(t, y, self.primary["mu"])

        if self.perts['J2']:
            a_pert = pert_fun.J2_perturbation(t, y, self.primary["mu"], self.primary["J2"], self.primary["Radius"])
            y_dot[3:] += a_pert

        return y_dot

    def propagate_orbit(self, report=False):
        if report:
            print('Loading: orbit propagation ...')

        self.y_out = odeint(self.dyn_ode, self.y0, self.tspan, rtol=1e-12, atol=1e-14, tfirst=True)
        self.rr_out = self.y_out[:, :3]
        self.vv_out = self.y_out[:, 3:]
        return

    def plot_3D(self, show_plot=False, save_plot=False, title='Test title'):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')

        # plot trajectory
        ax.plot(self.rr_out[:, 0], self.rr_out[:, 1], self.rr_out[:, 2], lw=1, label='Trajectory')
        ax.plot([self.rr_out[0, 0]], [self.rr_out[0, 1]], [self.rr_out[0, 2]], 'o', label='Initial position')
        ax.plot([self.rr_out[-1, 0]], [self.rr_out[-1, 1]], [self.rr_out[-1, 2]], 'd', label='Final position')

        # plot central body
        _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, 15), np.linspace(0, np.pi, 15))
        _x = self.primary["Radius"] * np.cos(_u) * np.sin(_v)
        _y = self.primary["Radius"] * np.sin(_u) * np.sin(_v)
        _z = self.primary["Radius"] * np.cos(_v)
        ax.plot_surface(_x, _y, _z, cmap='Blues')

        # set plot limits for equal axis
        max_val = np.max(np.abs(self.rr_out))
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])

        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')

        ax.set_title(title)
        plt.legend()

        if show_plot:
            plt.show()

        if save_plot:
            plt.savefig(title+'.png', dpi=300)

        return

    def kp_evolution(self, degree=True, report=False):
        if report:
            print('Loading: Keplerian elements computation ...')

        # init array to store keplerian parameters
        self.kp_out = np.zeros([self.n_step, 6])
        for i in range(self.n_step):
            self.kp_out[i, :] = rv2kp(self.rr_out[i, :], self.vv_out[i, :], self.primary["mu"], deg=degree)
        return

    def plot_kp(self, hours=False, days=False, show_plot=False,
                title='Keplerian parameters evolution', fig_size=(18, 9)):
        # create fig and axes
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=fig_size)
        fig.suptitle(title, fontsize=20)

        # scale the time axis
        if hours:
            tt = self.tspan / 3600
            xlab = 't [hours]'
        elif days:
            tt = self.tspan / 86400
            xlab = 't [days]'
        else:
            tt = self.tspan
            xlab = 't [s]'

        # plot semi-major axis
        axs[0, 0].plot(tt, self.kp_out[:, 0])
        axs[0, 0].set_title('Semi-major axis vs. Time')
        axs[0, 0].set_xlabel(xlab)
        axs[0, 0].set_ylabel(r'$a$ [km]')
        axs[0, 0].grid(True)

        # plot eccentricity
        axs[0, 1].plot(tt, self.kp_out[:, 1])
        axs[0, 1].set_title('Eccentricity vs. Time')
        axs[0, 1].set_xlabel(xlab)
        axs[0, 1].set_ylabel(r'$e$ [-]')
        axs[0, 1].grid(True)

        # plot true anomaly
        axs[0, 2].plot(tt, self.kp_out[:, 5])
        axs[0, 2].set_title('True anomaly vs. Time')
        axs[0, 2].set_xlabel(xlab)
        axs[0, 2].set_ylabel(r'$\theta$ [deg]')
        axs[0, 2].grid(True)

        # plot RAAN
        axs[1, 0].plot(tt, self.kp_out[:, 3])
        axs[1, 0].set_title('RAAN vs. Time')
        axs[1, 0].set_xlabel(xlab)
        axs[1, 0].set_ylabel(r'$\Omega$ [deg]')
        axs[1, 0].grid(True)

        # plot inclination
        axs[1, 1].plot(tt, self.kp_out[:, 2])
        axs[1, 1].set_title('Inclination vs. Time')
        axs[1, 1].set_xlabel(xlab)
        axs[1, 1].set_ylabel(r'$i$ [deg]')
        axs[1, 1].grid(True)

        # plot pericenter anomaly
        axs[1, 2].plot(tt, self.kp_out[:, 4])
        axs[1, 2].set_title('Pericenter anomaly vs. Time')
        axs[1, 2].set_xlabel(xlab)
        axs[1, 2].set_ylabel(r'$\omega$ [deg]')
        axs[1, 2].grid(True)

        if show_plot:
            plt.show()
        return


# TEST
if __name__ == '__main__':
    # plt.style.use('dark_background')
    R_earth = CelBody.Earth["Radius"]
    omega_earth = 2 * np.pi / CelBody.Earth["ST_rotation"]

    # Orbit parameters
    altitude = 5500.
    a = R_earth + altitude
    # a = 26600
    eccentricity = 0.1
    incl = 98.0
    Omega = 10.0
    omega = 40.0
    theta = 0.0
    kp0 = [a, eccentricity, incl, Omega, omega, theta]

    t_span = np.linspace(0, 1*86400, 1000)

    op = OrbitPropagatorR2BP(kp0, t_span, coes=True, deg=True)
    op.plot_3D(show_plot=True)

    # try animation
    rr = op.rr_out


    # init figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = p3.Axes3D(fig)
    max_r = np.max(abs(rr))
    ax.set_xlim(- max_r, max_r)
    ax.set_ylim(- max_r, max_r)
    ax.set_zlim(- max_r, max_r)

    """
    ## Using FunAnimate
    dot, = ax.plot([], [], [], 'ko', lw=2)
    line, = ax.plot([], [], [], lw=1)
    _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, 15), np.linspace(0, np.pi, 15))
    _x = R_earth * np.cos(_u) * np.sin(_v)
    _y = R_earth * np.sin(_u) * np.sin(_v)
    _z = R_earth * np.cos(_v)
    planet, = ax.plot_surface(_x, _y, _z, cmap='Blues')
    # fig_title = fig.suptitle('')

    def fig_animation(i, vect, tt, Rp, omega_p, n_tail=100):
        # fig_title.set_text('Time = {:.3f} h'.format(tt[i]/3600))  # for debug purposes

        # dot.set_data([vect[i, 0]], [vect[i, 1]])
        # dot.set_3d_properties([vect[i, 2]])
        iplus = i + 1
        if i < n_tail:
            line.set_data(vect[:iplus, 0], vect[:iplus, 1])
            line.set_3d_properties(vect[:iplus, 2], 'z')
        else:
            i_low = i - n_tail
            line.set_data(vect[i_low:iplus, 0], vect[i_low:iplus, 1])
            line.set_3d_properties(vect[i_low:iplus, 2], 'z')

        # plot central body
        _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, 15), np.linspace(0, np.pi, 15))
        _u = _u + (omega_p * tt[i])
        _x = Rp * np.cos(_u) * np.sin(_v)
        _y = Rp * np.sin(_u) * np.sin(_v)
        _z = Rp * np.cos(_v)
        planet.set_data(_x, _y)
        planet.set_3d_properties(_z, 'z')

        return dot, line, planet

    # choose the interval based on dt and the time to animate one step
    from time import time
    t0 = time()
    fig_animation(0, rr, t_span, R_earth, omega_earth)
    t1 = time()
    dt = 1. / 60  # 60 fps
    interval = 1000 * dt - (t1 - t0)

    anim = animation.FuncAnimation(fig, fig_animation, fargs=(rr, t_span, R_earth, omega_earth),
                                   frames=len(t_span), interval=interval, blit=True)
    plt.show()
    """

    # Using Celluloid Module
    camera = Camera(fig)
    n_tail = 50
    for i in range(len(t_span)):
        # Build each frame

        # plot central body
        _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, 15), np.linspace(0, np.pi, 15))
        _u = _u + (omega_earth * t_span[i])
        _x = R_earth * np.cos(_u) * np.sin(_v)
        _y = R_earth * np.sin(_u) * np.sin(_v)
        _z = R_earth * np.cos(_v)
        ax.plot_surface(_x, _y, _z, cmap='Blues')

        # plot orbit position and tail
        if i < n_tail:
            ax.plot(rr[:i+1, 0], rr[:i+1, 1], rr[:i+1, 2], 'b')
        else:
            i_low = i - n_tail
            ax.plot(rr[i_low:i+1, 0], rr[i_low:i+1, 1], rr[i_low:i+1, 2], 'b')
        ax.plot(rr[i, 0], rr[i, 1], rr[i, 2], 'ko', lw=2)

        camera.snap()

    animation = camera.animate(interval=100, blit=True)
    plt.show()

