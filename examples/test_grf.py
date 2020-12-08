import numpy as np

import starterlite

# Read in fiducial true [CII] power spectrum
pscii_data = np.load(os.getenv('STARTER')+
                     '/examples/TimeScienceMaster/data/CIIPS_TRUE_z%s_MAR_%.1e.npz'%(str(z_ps).replace('.','p'), ps.M_thres))
ps3d_cii_true = interp1d(pscii_data['k'], pscii_data['ps'], kind='linear', bounds_error=False, fill_value=(0., 0.))