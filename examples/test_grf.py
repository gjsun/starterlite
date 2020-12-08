import numpy as np
import os
from scipy.interpolate import interp1d

import starterlite

# Read in fiducial true [CII] power spectrum
pscii_data = np.load(os.getenv('STARTERLITE')+'/input/CIIPS_TRUE_z7p1_Mmin_1e8.npz')
ps3d_cii_true = interp1d(pscii_data['k'], pscii_data['ps'], kind='linear', bounds_error=False, fill_value=(0., 0.))

grf = starterlite.simulation.GaussianRandomField()

grf.sens.survey_goemetry = np.array([180, 1, 30])
grf.survey_goemetry = np.array([180, 1, 30])

grf.PowerSpectrum = ps3d_cii_true

grf.GenerateGRF(L_x=grf.sens.beam_size_at_z(z=7.1, physical=True)*2.355*grf.n_beam, 
				L_y=grf.sens.beam_size_at_z(z=7.1, physical=True)*2.355, 
				L_z=grf.sens.bandwidth_LF_Mpch)

