# Note: read Help of lambertMR
# lambertMR(RI,RF,TOF,MU,orbitType,Nrev,Ncase,optionsLMR)
# as input you need only:
# RI,RF,TOF,MU,
# for the other parameters set:
# orbitType = 0;
# Nrev = 0;
# optionsLMR = 0;

from src.Utilities.lamberts_solver import *
from src.Utilities.SolarSystemBodies import sun

muSun = sun['mu']      # mu Sun
rr0 = np.array([10000, 0, 0])
rr1 = np.array([20000, 0, 0])
ToF = 50*86400                 # Time in [s]
vv0, vv1 = lamberts_universal_variables(rr0, rr1, ToF, muSun)

print(vv0)
print(vv1)
