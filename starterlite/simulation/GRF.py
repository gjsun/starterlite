import numpy as np
import os

from ..analysis import Sensitivity
from ..physics import Cosmology
from ..physics.Constants import c, cm_per_mpc
from ..util.ParameterFile import ParameterFile
from .FourierSpace import FourierSpace

"""

------------
Instructions
------------
The GRF module allows the user to generate realizations of a Gaussian random field with an input power spectrum, and 
compute power spectrum from a given map of fluctuations in real space. 

* Acknowledgements
This module has benefited a lot from and share many features in common with the imapper2 package developed by Tony Li. 

"""






class GaussianRandomField(FourierSpace):
    def __init__(self, **kwargs):
        FourierSpace.__init__(self, **kwargs)

        self.pf = ParameterFile(**kwargs)
        # Get the redshift of the interested signal
        self._z = self.pf.grf_params['grf_z_signal']
        # Specify the survey geometry
        self._survey_goemetry = np.array([self.pf.grf_params['grf_geom_x'],
                                          self.pf.grf_params['grf_geom_y'],
                                          self.pf.grf_params['grf_geom_z']])
        # Get the wavelength [cm] of the interested signal
        self.wv_signal = self.pf.grf_params['grf_lambda_signal']
        # Get the assumed aperture size (diameter) of dish
        self.d_ap = self.pf.grf_params['grf_d_ap']

        self._powerspectrum_in = self.pf.grf_params['grf_ps_in']


    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(
                omega_m_0=self.pf.cosmo_params['omega_m_0'],
                omega_l_0=self.pf.cosmo_params['omega_l_0'],
                omega_b_0=self.pf.cosmo_params['omega_b_0'],
                hubble_0=self.pf.cosmo_params['hubble_0'],
                helium_by_number=self.pf.cosmo_params['helium_by_number'],
                cmb_temp_0=self.pf.cosmo_params['cmb_temp_0'],
                approx_highz=self.pf.cosmo_params['approx_highz'],
                sigma_8=self.pf.cosmo_params['sigma_8'],
                primordial_index=self.pf.cosmo_params['primordial_index'])

        return self._cosm


    @property
    def sens(self, **kwargs):
        if not hasattr(self, '_sens'):
            self._sens = Sensitivity(**kwargs)
        return self._sens


    @property
    def z(self):
        if not hasattr(self, '_z'):
            raise ValueError('must specify a redshift for which the fluctuations of target signal will be simulated!')
        return self._z

    @z.setter
    def z(self, value):
        if value in [6.0]:
            self._z = value
        else:
            raise ValueError('invalid signal redshift!')


    @property
    def survey_goemetry(self):
        if not hasattr(self, '_survey_goemetry'):
            raise ValueError('must specify a survey geometry for the simulation!')
        return self._survey_goemetry


    @survey_goemetry.setter
    def survey_goemetry(self, value):
        if (np.alltrue(value>0)) and (np.size(value)==3):
            self._survey_goemetry = value
            #print 'updated default geometry to %s!'%(self._survey_goemetry)
        else:
            raise ValueError('input survey geometry invalid!')


    @property
    def powerspectrum_in(self):
        if not hasattr(self, '_powerspectrum_in'):
            raise ValueError('To simulate a GRF, must supply an input PS!')
        return self._powerspectrum_in


    @powerspectrum_in.setter
    def powerspectrum_in(self, value):
        if callable(value):
            self._powerspectrum_in = value
        else:
            raise ValueError('Input power spectrum must be a callable function!')


    @property
    def n_ch_x(self):
        return self.survey_goemetry[0]


    @property
    def n_ch_y(self):
        return self.survey_goemetry[1]


    @property
    def n_ch_z(self):
        return self.survey_goemetry[2]


    def get_x_k_grid(self):

        _theta_FWHM = 1.15 * (self.wv_signal * (1.+self.z)) / self.d_ap   # [radians]
        _rofz = self.cosm.ComovingRadialDistance(0.,self.z) / cm_per_mpc * self.cosm.h70
        self.L_x = _lslab_x = _rofz * _theta_FWHM * self.n_ch_x
        self.L_y = _lslab_y = _rofz * _theta_FWHM * self.n_ch_y
        _zl = c/self.sens.nu2_HF/self.wv_signal - 1.
        _zh = c/self.sens.nu1_LF/self.wv_signal - 1.
        self.L_z = _lslab_z = self.cosm.ComovingRadialDistance(_zl, _zh) / cm_per_mpc * self.cosm.h70

        # Define the large simulation box within which the survey volume is embedded
        _lsim_x = _lsim_y = _lsim_z = _lslab_z   # Mpc h^-1

        _dx = _lslab_x / self.n_ch_x   # Mpc h^-1
        _dy = _lslab_y / self.n_ch_y   # Mpc h^-1
        _dz = _lslab_z / self.n_ch_z   # Mpc h^-1

        _nx_sim = np.round(_lsim_x / _dx).astype(int)
        _ny_sim = np.round(_lsim_y / _dy).astype(int)
        _nz_sim = np.round(_lsim_z / _dz).astype(int)

        self.xs = np.linspace(-_nx_sim/2 + _nx_sim%2, _nx_sim/2 - 1 + _nx_sim%2, _nx_sim) * _dx
        self.ys = np.linspace(-_ny_sim/2 + _ny_sim%2, _ny_sim/2 - 1 + _ny_sim%2, _ny_sim) * _dy
        self.zs = np.linspace(-_nz_sim/2 + _nz_sim%2, _nz_sim/2 - 1 + _nz_sim%2, _nz_sim) * _dz

        sim = np.zeros((_nx_sim, _ny_sim, _nz_sim), float)
        self.npix_cen = _nx_sim / 2
        if self.n_ch_y==1:
            sim[int(self.npix_cen - (self.n_ch_x/2)):int(self.npix_cen + (self.n_ch_x/2)), self.npix_cen, :] = 1.0  # real-space weighting function
        else:
            sim[int(self.npix_cen - (self.n_ch_x/2)):int(self.npix_cen + (self.n_ch_x/2)),
                int(self.npix_cen - (self.n_ch_y/2)):int(self.npix_cen + (self.n_ch_y/2)),
                :] = 1.0

        _kx = 2*np.pi * np.fft.fftfreq(_nx_sim, _dx)
        _ky = 2*np.pi * np.fft.fftfreq(_ny_sim, _dy)
        _kz = 2*np.pi * np.fft.fftfreq(_nz_sim, _dz)

        _dkx = abs(_kx[1] - _kx[0])
        _dky = abs(_ky[1] - _ky[0])
        _dkz = abs(_kz[1] - _kz[0])

        self.k = np.sqrt(_kx[:,np.newaxis,np.newaxis]**2 + _ky[np.newaxis,:,np.newaxis]**2 + _kz[np.newaxis,np.newaxis,:]**2)

        _box_vol = _lsim_x * _lsim_y * _lsim_z
        _pix_vol = _box_vol / (_nx_sim * _ny_sim * _nz_sim)
        self.scale_factor = np.sqrt(_pix_vol** 2 / _box_vol)



    def run(self, n_samples=1):

        print('\n--- GRF sims STARTED ---\n')

        self.fn = 'grf_samples_x%dy%dz%d_N%d' % (self.n_ch_x, self.n_ch_y, self.n_ch_z, n_samples)

        if not callable(self.powerspectrum_in): raise TypeError('Input power spectrum must be a callable function!')

        self.survey_maps = np.zeros((self.n_ch_x, self.n_ch_y, self.n_ch_z, n_samples))

        print('Generating x and k grids...')

        self.get_x_k_grid()

        #print '\n---------- Simulation Specifications ----------'
        #print '- Number of samples:', n_samples
        #print '- Survey geometry:', self.survey_goemetry
        #print '- Survey volume: %.5e, %.5e, %.5e' % (self.L_x, self.L_y, self.L_z)
        #print '-----------------------------------------------\n'

        print('Generating GRF realizations...')

        if self.n_ch_y == 1:
            for i in xrange(n_samples):
                # Generate real and imaginary parts
                rand = np.random.RandomState(seed=(42 + i))
                # real_part = np.sqrt(0.5*Pk) * rand.Cheng(loc=0.0,scale=1.0,size=k.shape) / scale_factor
                # imaginary_part = np.sqrt(0.5*Pk) * rand.Cheng(loc=0.0,scale=1.0,size=k.shape) / scale_factor
                real_part = np.sqrt(self.powerspectrum_in(self.k)) * rand.normal(loc=0.0, scale=1.0, size=self.k.shape) / self.scale_factor
                imaginary_part = np.sqrt(self.powerspectrum_in(self.k)) * rand.normal(loc=0.0, scale=1.0, size=self.k.shape) / self.scale_factor

                # Get map in real space and return
                ft_map = (real_part + imaginary_part * 1.0j)
                ft_map[np.where(self.k==np.zeros_like(self.k))] = 0.0

                noise_map = np.fft.ifftn(ft_map)
                noise_map = np.real(noise_map)

                survey_map = noise_map[int(self.npix_cen - (self.n_ch_x / 2)):int(self.npix_cen + (self.n_ch_x / 2)), self.npix_cen, :]
                self.survey_maps[:, 0, :, i] = survey_map

                print('Iteration %d completed.'%(i + 1))
            self.survey_map_coords = [self.xs[int(self.npix_cen-(self.n_ch_x/2)):int(self.npix_cen+(self.n_ch_x/2))],
                                      None,
                                      self.zs]
        else:
            raise ValueError('help!')

        self.save()

        print('\n--- GRF sims DONE ---\n')



    def save(self, format='npz'):
        """
        Save derived window functions to file
        ----------------------------------------
        :param format: format of output file; {str}
        """
        _path = os.getenv('STARTER') + '/examples/example_output/grf/%s.%s' % (self.fn, format)
        _wf_dict = {'grf': self.survey_maps, 'coords': self.survey_map_coords}
        np.savez(_path, **_wf_dict)