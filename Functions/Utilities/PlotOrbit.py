import numpy as np
from matplotlib import pyplot as plt
import PIL


def plot_n_orbits(rr_orbits, labels, show_plot=True, save_plot=False, title='Multiple orbits title'):
    """
    This function plots "n" orbits given in input

    :param rr_orbits:   List of position array with shape (m, 6), where m is the number in which the orbit is discretized
    :param labels:      List of labels to be associated to each trajectory
    :param show_plot:   bool
    :param save_plot:   bool
    :param title:       Title of the figure

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


def plot_sphere(Rp, position=[0, 0, 0], theta=0, sphere_resolution=15, ax=None):
    if not bool(ax):
        ax = plt.gca()

    # plot central body
    _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, sphere_resolution), np.linspace(0, np.pi, sphere_resolution))
    _u = _u + theta
    _x = position[0] + Rp * np.cos(_u) * np.sin(_v)
    _y = position[1] + Rp * np.sin(_u) * np.sin(_v)
    _z = position[2] + Rp * np.cos(_v)
    return ax.plot_surface(_x, _y, _z, cmap='Blues')


def plot_plantet(Rp, texture_path, resolution=7, position=[0, 0, 0], theta=0., ax=None):
    if not bool(ax):
        ax = plt.gca()

    # load texture with PIL
    bm = PIL.Image.open(texture_path)
    # it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept
    bm = np.array(bm.resize([int(d/resolution) for d in bm.size], resample=PIL.Image.BICUBIC)) / 256.

    # coordinates of the image - don't know if this is entirely accurate, but probably close
    lons = theta + np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    x = position[0] + Rp * np.outer(np.cos(lons), np.cos(lats)).T
    y = position[1] + Rp * np.outer(np.sin(lons), np.cos(lats)).T
    z = position[2] + Rp * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    return ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=bm)


if __name__ == '__main__':
    # Constants
    R_earth = 0.63781600000000E+04
    mu_earth = 0.59736990612667E+25 * 6.67259e-20

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    texture_path = './texture/Earth.jpg'
    plot_plantet(R_earth, texture_path, resolution=2)
    plt.show()
