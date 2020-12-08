import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import starterlite

grf = starterlite.simulation.GaussianRandomField()

grf_input = np.load(os.getenv('STARTERLITE')+'/output/grf/grf_samples_x180y1z30_N1.npz', allow_pickle=True)

f_in = grf_input['grf']
f_in = f_in.squeeze()
x_in, z_in, y_in = grf_input['coords']

print('f_in has shape:', f_in.shape)
print('x_in has shape:', x_in.shape)
print('y_in has shape:', y_in.shape)
print('z_in:', z_in)

ksph, psph = grf.AverageAutoPS(f=f_in, x=x_in, y=y_in, z=None, bins=11, log=True, avg_type='sph')


plt.figure(figsize=(6,3))
plt.loglog(ksph, psph, 'ko-', lw=1)
plt.ylim([1e7, 1e9])
plt.xlabel('wavenumber k', fontsize=16)
plt.ylabel('measured P(k)', fontsize=16)
plt.tight_layout()
plt.show()