import matplotlib.pyplot as plt
import numpy as np
import OOPAO

from OOPAO.Source import Source
# to create a natural guide star in I band of magnitude 5
ngs = Source(magnitude = 5,
optBand ='I')

from OOPAO.Telescope import Telescope

# telescope parameters
sensing_wavelength = ngs.wavelength # sensing wavelength of the WFS
n_subaperture = 20 # number of subap accross the diameter
diameter = 8 # diameter of the phase screens in [m]
resolution = n_subaperture*8 # resolution of the phase screens in pix
pixel_size = diameter/resolution # size of the pixels in [m]
obs_ratio = 0.1 # central obstruction ratio
sampling_time = 1/1000 # sampling time of the AO loop in [s]
# initialize the telescope object
tel = Telescope(diameter = diameter,
resolution = resolution,
centralObstruction = obs_ratio,
samplingTime = sampling_time)
ngs*tel
from OOPAO.Atmosphere import Atmosphere
# setting a two layers atmosphere
atm = Atmosphere(telescope = tel,
                r0 = 0.15,# Fried parameter @500 nm in [m]
                L0 = 25,# Outer scale in [m]
                fractionalR0 = [0.7, 0.3 ],# Cn2 profile
                altitude = [0 , 10000],
                windDirection = [0 , 20 ],
                windSpeed = [5 , 10 ])
atm.initializeAtmosphere(telescope=tel)
atm.display_atm_layers()

for i in range(1000):
    atm.update()
atm.display_atm_layers()
plt.show()
