import poliastro.bodies as solar_sys
from numpy import pi

Earth = {
    "name": 'Earth',
    "NAIF_ID": 3,
    "mu": solar_sys.Earth.k.to('km3/s2').value,
    "Radius": solar_sys.Earth.R.to('km').value,
    "sidereal_day": (23 * 60 + 56) * 60 + 4.09,
    "sidereal_year": 365.256363004,
    "J2": solar_sys.Earth.J2.value
}
