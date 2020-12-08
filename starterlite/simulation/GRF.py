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

This module has benefited a lot from the imapper2 package developed by Tony Li. 

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
    def PowerSpectrum(self):
        if not hasattr(self, '_PowerSpectrum'):
            raise ValueError('To simulate a GRF, must supply an input PS!')
        return self._PowerSpectrum


    @PowerSpectrum.setter
    def PowerSpectrum(self, value):
        if callable(value):
            self._PowerSpectrum = value
        else:
            raise ValueError('Input power spectrum must be a callable function of k!')


    @property
    def n_ch_x(self):
        return self.survey_goemetry[0]


    @property
    def n_ch_y(self):
        return self.survey_goemetry[1]


    @property
    def n_ch_z(self):
        return self.survey_goemetry[2]


    @property
    def n_beam(self):
        return self.survey_goemetry[0] * self.survey_goemetry[1]


    @property
    def n_channel(self):
        return self.survey_goemetry[-1]


    def SetGrid(self, L_x, L_y, L_z):
        """
        Set x (real space) and k (fourier space) grids
        ----------------------------------------
        :param L_x: length of survey volume along 1st dimension; {scalar}
        :param L_y: length of survey volume along 2nd dimension; {scalar}
        :param L_z: length of survey volume along 3rd (LOS) dimension; {scalar}
        :return:
        """

        _lslab_x = L_x
        _lslab_y = L_y
        _lslab_z = L_z

        # Define the large simulation box within which the survey volume is embedded
        _lsim_x = _lsim_y = _lsim_z = _lslab_z   # Mpc h^-1

        _dx = _lslab_x / self.n_ch_x   # Mpc h^-1
        _dy = _lslab_y / self.n_ch_y   # Mpc h^-1
        _dz = _lslab_z / self.n_ch_z   # Mpc h^-1

        self.nx_sim = np.round(_lsim_x / _dx).astype(int)
        self.ny_sim = np.round(_lsim_y / _dy).astype(int)
        self.nz_sim = np.round(_lsim_z / _dz).astype(int)

        self.xs = np.linspace(-self.nx_sim//2 + self.nx_sim%2, self.nx_sim//2 - 1 + self.nx_sim%2, self.nx_sim) * _dx
        self.ys = np.linspace(-self.ny_sim//2 + self.ny_sim%2, self.ny_sim//2 - 1 + self.ny_sim%2, self.ny_sim) * _dy
        self.zs = np.linspace(-self.nz_sim//2 + self.nz_sim%2, self.nz_sim//2 - 1 + self.nz_sim%2, self.nz_sim) * _dz

        self.r = np.sqrt(self.xs[:,np.newaxis,np.newaxis]**2 + self.ys[np.newaxis,:,np.newaxis]**2 + self.zs[np.newaxis,np.newaxis,:]**2)

        sim = np.zeros((self.nx_sim, self.ny_sim, self.nz_sim), float)
        self.npix_cen = self.nx_sim // 2 - 1
        if self.n_ch_y==1:
            # real-space weighting function
            sim[int(self.npix_cen - (self.n_beam // 2)):int(self.npix_cen + (self.n_beam // 2)), self.npix_cen, 0:] = 1.0
        else:
            raise NotImplementedError('help!')

        _kx = 2*np.pi * np.fft.fftfreq(self.nx_sim, _dx)
        _ky = 2*np.pi * np.fft.fftfreq(self.ny_sim, _dy)
        _kz = 2*np.pi * np.fft.fftfreq(self.nz_sim, _dz)

        _dkx = abs(_kx[1] - _kx[0])
        _dky = abs(_ky[1] - _ky[0])
        _dkz = abs(_kz[1] - _kz[0])

        self.k = np.sqrt(_kx[:,np.newaxis,np.newaxis]**2 + _ky[np.newaxis,:,np.newaxis]**2 + _kz[np.newaxis,np.newaxis,:]**2)

        _box_vol = _lsim_x * _lsim_y * _lsim_z
        _pix_vol = _box_vol / (self.nx_sim * self.ny_sim * self.nz_sim)
        self.scale_factor = np.sqrt(_pix_vol**2 / _box_vol)


    def GenerateGRF(self, L_x, L_y, L_z, n_samples=1):
        """
        Generate Gaussian random field according to the provided geometry and power spectrum
        ----------------------------------------
        :param L_x: length of survey volume along 1st dimension; {scalar}
        :param L_y: length of survey volume along 2nd dimension; {scalar}
        :param L_z: length of survey volume along 3rd (LOS) dimension; {scalar}
        :param n_samples: number of GRF realizations to generate
        :return:
        """

        self.fn = 'grf_samples_x%dy%dz%d_N%d' % (self.n_ch_x, self.n_ch_y, self.n_ch_z, n_samples)

        if not callable(self.PowerSpectrum): raise TypeError('Input power spectrum must be a callable function of k!')

        self.survey_maps = np.zeros((self.n_ch_x, self.n_ch_y, self.n_ch_z, n_samples))

        print('\nGenerating x (real space) and k (fourier space) grids...')

        self.SetGrid(L_x=L_x, L_y=L_y, L_z=L_z)

        print('\nReading in power spectrum...')

        try:
            Pk = self.PowerSpectrum(self.k)
            assert Pk[Pk >= 0.0].size == Pk.size
        except:
            raise ValueError('Oops!')

        print('\nGenerating GRF realizations...')

        if self.n_ch_y == 1:
            for i in range(n_samples):
                # Generate real and imaginary parts
                rand = np.random.RandomState(seed=(42 + i))

                realspace_vec_r = rand.normal(loc=0.0, scale=1.0, size=self.r.shape)
                realspace_vec_i = rand.normal(loc=0.0, scale=1.0, size=self.r.shape)
                realspace_map = (realspace_vec_r + realspace_vec_i * 1.0j)

                fourierspace_map = np.fft.fftn(realspace_map) / np.sqrt(self.nx_sim * self.ny_sim * self.nz_sim)
                ft_map = np.sqrt(Pk) * fourierspace_map / self.scale_factor
                ft_map[0, 0, 0] = 0.0

                full_map = np.fft.ifftn(ft_map)
                full_map = np.real(full_map)

                survey_map = full_map[int(self.npix_cen-(self.n_ch_x//2)):int(self.npix_cen+(self.n_ch_x//2)), self.npix_cen, :]
                self.survey_maps[:, :, :, i] = survey_map

                print('%d out of %d realizations completed!'%(i+1, n_samples))

            self.survey_map_coords = [self.xs[int(self.npix_cen-(self.n_ch_x//2)):int(self.npix_cen+(self.n_ch_x//2))], None, self.zs]
        else:
            raise NotImplementedError('help!')

        self.save()

        print('\n--- DONE ---\n')



    def save(self, format='npz'):
        """
        Save derived window functions to file
        ----------------------------------------
        :param format: format of output file; {str}
        """
        _path = os.getenv('STARTERLITE') + '/output/grf/%s.%s' % (self.fn, format)
        _wf_dict = {'grf': self.survey_maps, 'coords': self.survey_map_coords}
        np.savez(_path, **_wf_dict)