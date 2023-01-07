import numpy as np
from matplotlib import pyplot as plt
import PIL

time_handler = {
    'seconds': {'coeff': 1.0,        'xlabel': 'Time (seconds)'},
    'hours':   {'coeff': 3600.0,     'xlabel': 'Time (hours)'},
    'days':    {'coeff': 86400.0,    'xlabel': 'Time (days)'},
    'years':   {'coeff': 31536000.0, 'xlabel': 'Time (years)'}
}

dist_handler = {
    'm':  1000.0,
    'km': 1.0,
    'ER': 1 / 6378.0,
    'JR': 1 / 71490.0,
    'AU': 6.68459e-9,
    r'$\dfrac{km}{s}$': 1.0
}

COLORS = [
             'm', 'deeppink', 'chartreuse', 'w', 'springgreen', 'peachpuff',
             'white', 'lightpink', 'royalblue', 'lime', 'aqua'] * 100


def plot_n_orbits(rr_orbits, labels=None, args=None, ax=None):
    """
    This function plots "n" orbits given in input

    :param rr_orbits:   List of position array with shape (m, 6), where m is the number in which the orbit is discretized
    :param args:
    :param labels:      List of labels to be associated to each trajectory
    :param show_plot:   bool
    :param save_plot:   bool
    :param title:       Title of the figure

    :return: fig, ax
    """
    _args = {
        'fig_size': (16, 9),
        'title': '',
        'labels': None,
        'colors': ['b', 'r', 'g', 'y', 'm'],
        'xlabel': 'x-axis',
        'ylabel': 'y-axis',
        'zlabel': 'z-axis',
        'ul': 'km',
        'show_plot': True,
        'save_plot': False,
        'filename': "orbit trajectory.png",
        'dpi': 300,
    }
    if args is not None:
        for key in args.keys():
            _args[key] = args[key]

    if labels is None:
        for i_lab in range(len(rr_orbits)):
            labels[i_lab] = ['orbit #' + str(i_lab)]

    # if ax has been given as input, plot the orbit on it. Otherwise create a new figure
    if ax is None:
        fig = plt.figure(figsize=_args['fig_size'])
        ax = fig.add_subplot(projection='3d')

    # plot trajectory
    n = 0
    for rr_out in rr_orbits:
        ax.plot(rr_out[:, 0], rr_out[:, 1], rr_out[:, 2], lw=1, label=labels[n])
        ax.plot(rr_out[0, 0], rr_out[0, 1], rr_out[0, 2], 'o', label='Initial position')
        n += 1

    # set plot equal apect ration
    ax.set_aspect('equal')
    ax.set_xlabel(_args['xlabel'] + ' [' + _args['ul'] + ']')
    ax.set_ylabel(_args['ylabel'] + ' [' + _args['ul'] + ']')
    ax.set_zlabel(_args['zlabel'] + ' [' + _args['ul'] + ']')
    ax.set_title(_args['title'])
    plt.legend()

    if _args['show_plot']:
        plt.show()

    if _args['save_plot']:
        plt.savefig(_args['filename'], dpi=_args['dpi'])

    return


def plot_sphere(Rp=1, position=[0, 0, 0], theta=0, sphere_resolution=15, ax=None):
    if ax is None:
        ax = plt.gca()

    # plot central body
    _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, sphere_resolution), np.linspace(0, np.pi, sphere_resolution))
    _u = _u + theta
    _x = position[0] + Rp * np.cos(_u) * np.sin(_v)
    _y = position[1] + Rp * np.sin(_u) * np.sin(_v)
    _z = position[2] + Rp * np.cos(_v)
    return ax.plot_surface(_x, _y, _z, cmap='Blues')


def plot_plantet(Rp, texture_path, resolution=7, position=[0, 0, 0], theta=0, ax=None, transparency=1):
    if ax is None:
        ax = plt.gca()

    # load texture with PIL
    bm = PIL.Image.open(texture_path)
    # it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept
    bm = np.array(bm.resize([int(d / resolution) for d in bm.size], resample=PIL.Image.BICUBIC)) / 256.

    # coordinates of the image - don't know if this is entirely accurate, but probably close
    lons = theta + np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    x = position[0] + Rp * np.outer(np.cos(lons), np.cos(lats)).T
    y = position[1] + Rp * np.outer(np.sin(lons), np.cos(lats)).T
    z = position[2] + Rp * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    return ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=bm, alpha=transparency)


if __name__ == '__main__':
    # Constants
    R_earth = 0.63781600000000E+04
    mu_earth = 0.59736990612667E+25 * 6.67259e-20

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    texture_path = './texture/Earth.jpg'
    plot_plantet(R_earth, texture_path)
    plt.show()
