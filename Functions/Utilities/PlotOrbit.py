import numpy as np
from matplotlib import pyplot as plt


def plot_n_orbits(rr_orbits, labels, show_plot=True, save_plot=False, title='Multiple orbits title'):
    """
    This function plots "n" orbits given in input

    :param rr_orbits: list of position array with shape (m, 6), where m is the number in which the orbit is discretized
    :param labels: list of labels to be associated to each trajectory
    :param show_plot: bool
    :param save_plot: bool
    :param title: title of the figure

    :return: fig, ax
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    # plot trajectory
    n = 0
    for rr_out in rr_orbits:
        ax.plot(rr_out[:, 0], rr_out[:, 1], rr_out[:, 2], lw=1, label=labels[n])
        ax.plot([rr_out[0, 0]], [rr_out[0, 1]], [rr_out[0, 2]], 'o', label='Initial position')
        n += 1

    # set plot limits for equal axis
    max_val = np.max(np.abs(rr_orbits))
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
        plt.savefig(title + '.png', dpi=300)

    return fig, ax


def plot_planet(Rp, position=[0, 0, 0], sphere_resolution=15):
    # plot central body
    _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, sphere_resolution), np.linspace(0, np.pi, sphere_resolution))
    _x = position[0] + Rp * np.cos(_u) * np.sin(_v)
    _y = position[1] + Rp * np.sin(_u) * np.sin(_v)
    _z = position[2] + Rp * np.cos(_v)
    ax.plot_surface(_x, _y, _z, cmap='Blues')

    # load coastline coords [long, lat]
    coast_coords = np.genfromtxt('/Francesco/OrbitalDynPy/Functions/Utilities/coastlines.csv', delimiter=',')



if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plot_planet(10)
    plt.show()
