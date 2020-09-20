import numpy as np
from numpy import arccos as acos
from matplotlib import pyplot as plt

from Functions.Utilities.SolarSystemBodies import Earth

# Rate of revolution of the Earth
n_earth = 2 * np.pi / (Earth["sidereal_year"] * 86400)

def inclination_sunsynch(a, e=0., mu=Earth["mu"], Rp=Earth["Radius"], J2=Earth["J2"], Omega_dot=n_earth, deg=True):
    # semi-latus rectum of the orbit
    p = a * (1 - e ** 2)

    # mean motion of the orbit
    #n = (mu / a ** 3) ** 0.5

    # inclincation
    incl = acos(- 2 / 3 * Omega_dot / J2 * ((a ** 3 / mu) ** 0.5) * ((p / Rp) ** 2))
    if deg:
        return incl * 180 / np.pi
    else:
        return incl


if __name__ == '__main__':
    h_span = np.linspace(300, 1500)
    r_earth = Earth["Radius"]

    i_sso = np.zeros(len(h_span))
    for n in range(len(h_span)):
        a = h_span[n] + r_earth
        i_sso[n] = inclination_sunsynch(a)

    fig, ax = plt.subplots()
    ax.plot(h_span, i_sso)
    ax.set_xlim([300, 1500])
    ax.set_ylim([96, 103])
    ax.set_xlabel('Altitude [km]')
    ax.set_ylabel('Inclination [deg]')
    ax.set_title('Sun-Synchronous Orbit (e = 0)')
    ax.grid()

    plt.show()
