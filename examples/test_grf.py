import numpy as np

import starterlite

# Read in fiducial true [CII] power spectrum
pscii_data = np.load(os.getenv('STARTERLITE')+'/input/CIIPS_TRUE_z7p1_MAR_1.0e+08.npz'))
ps3d_cii_true = interp1d(pscii_data['k'], pscii_data['ps'], kind='linear', bounds_error=False, fill_value=(0., 0.))