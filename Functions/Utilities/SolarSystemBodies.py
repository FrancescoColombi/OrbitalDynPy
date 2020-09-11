import poliastro.bodies as solar_sys


Earth = {
    "name": 'Earth',
    "NAIF_ID": 3,
    "mu": solar_sys.Earth.k.to('km3/s2').value,
    "Radius": solar_sys.Earth.R.to('km').value,
    "J2": solar_sys.Earth.J2.value
}
