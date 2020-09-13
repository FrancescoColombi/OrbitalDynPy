import poliastro.bodies as solar_sys


Earth = {
    "name": 'Earth',
    "NAIF_ID": 3,
    "mu": solar_sys.Earth.k.to('km3/s2').value,
    "Radius": solar_sys.Earth.R.to('km').value,
    "ST_rotation": (23 + (56 + 4.09 / 60) / 60) / 24 * 86400,
    "J2": solar_sys.Earth.J2.value
}
