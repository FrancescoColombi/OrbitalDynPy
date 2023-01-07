import numpy as np
from numpy import arccos as acos
from matplotlib import pyplot as plt

from src.Utilities.SolarSystemBodies import earth

# Rate of revolution of the Earth
n_earth = 2 * np.pi / (earth["sidereal_year"] * 86400)

def inclination_sunsynch(a, e=0., mu=earth["mu"], Rp=earth["radius"], J2=earth["J2"], Omega_dot=n_earth, deg=True):
    """

    :param a: Semi-major axis
    :param e: Eccentricity
    :param mu: Gravitational parameter
    :param Rp: Radius of the primary
    :param J2: J2 geopotential term of the primary
    :param Omega_dot: Desired precession rate of the Right-Ascension of the Ascending Node
    :param deg: Bool
    :return: incl - Orbit inclination
    """
    # semi-latus rectum of the orbit
    p = a * (1 - e ** 2)

    # mean motion of the orbit
    # n = (mu / a ** 3) ** 0.5

    # inclincation
    incl = acos(- 2 / 3 * Omega_dot / J2 * ((a ** 3 / mu) ** 0.5) * ((p / Rp) ** 2))
    if deg:
        return incl * 180 / np.pi
    else:
        return incl


if __name__ == '__main__':
    h_span = np.linspace(300, 5000)
    r_earth = earth["radius"]

    i_sso = np.zeros(len(h_span))
    for n in range(len(h_span)):
        a = h_span[n] + r_earth
        i_sso[n] = inclination_sunsynch(a)

    fig, ax = plt.subplots()
    ax.plot(h_span, i_sso)
    ax.set_xlim([h_span[0], h_span[-1]])
    ax.set_ylim([90, i_sso[-1]])
    ax.set_xlabel('Altitude [km]')
    ax.set_ylabel('Inclination [deg]')
    ax.set_title('Sun-Synchronous Orbit (e = 0)')
    ax.grid()

    plt.show()
