import numpy as np
import warnings
import os
from scipy.io.idl import readsav

from ..physics import Cosmology
from ..physics.Constants import c, cm_per_mpc, lambda_CII, lambda_CO_10, \
    lambda_HI21, lambda_NII122, lambda_NII205, lambda_H2S0, lambda_H2S1, lambda_H2S2, lambda_H2S3, lambda_HCN10
from ..util.ParameterFile import ParameterFile

class Sensitivity(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        self.t_obs_survey = self.pf.sensitivity_params['sens_t_obs_survey']
        self.n_feedhorns_inst = self.pf.sensitivity_params['sens_n_feedhorns']
        self.d_ap = self.pf.sensitivity_params['sens_d_ap']

        # Specify the survey geometry
        self._survey_goemetry = np.array([self.pf.sensitivity_params['sens_geom_x'],
                                          self.pf.sensitivity_params['sens_geom_y'],
                                          self.pf.sensitivity_params['sens_geom_z']])

        # Get the wavelength [cm] of the interested signal
        self.wv_signal = self.pf.sensitivity_params['sens_lambda_signal']
        # only support [CII] and CO(3-2), CO(4-3), CO(5-4) now!
        #assert set(self.wv_signal).issubset(set([1.577e-2, lambda_CO_32, lambda_CO_43, lambda_CO_54, lambda_CO_65]))
        self.wv_signal = np.array(self.wv_signal)

        # ---------------------------   Read in TIME instrument data   --------------------------- #
        self.iden_channel = readsav(os.getenv('STARTER') + '/input/time_spec/TP_arizona.sav')
        if self.pf.sensitivity_params['sens_read_tnoise']:
            try: self.nei_data = np.load(os.getenv('STARTER') + '/input/time_spec/20171101_baseline_kp_0.6fudge_nei.npz')
            except: raise NotImplementedError('No thermal-noise information provided!')

        self.detfreq = self.nei_data['freq'].copy()
        self.detnei = self.nei_data['nei'].copy() * 1.0e6   # convert from MJy to Jy


        # Boolean array for high-freq channels
        self.sciband_high = self.iden_channel["scienceband_high"].copy()
        self.sciband_low = self.iden_channel["scienceband_low"].copy()
        # Swap the numbers of HF & LF bands
        self.sciband_high[20::] = 0
        self.sciband_low[20:36] = 1
        # Band ranges
        self.band_HF = self.detfreq[np.where(self.sciband_high)] * 1e9
        self.nu1_HF, self.nu2_HF = min(self.band_HF), max(self.band_HF)   # 230.68e9, 301.99e9 [Hz]
        self.band_LF = self.detfreq[np.where(self.sciband_low)] * 1e9
        self.nu1_LF, self.nu2_LF = min(self.band_LF), max(self.band_LF)   # 199.86e9, 228.58e9 [Hz]
        # Bandwidths
        self.bandwidth_HF = self.nu2_HF - self.nu1_HF
        self.bandcenter_HF = self.nu1_HF + ((self.nu2_HF - self.nu1_HF) / 2.)
        self.bandwidth_LF = self.nu2_LF - self.nu1_LF
        self.bandcenter_LF = self.nu1_LF + ((self.nu2_LF - self.nu1_LF) / 2.)
        # N channels
        self.nchannels_HF = np.size(self.band_HF)   # 30
        self.nchannels_LF = np.size(self.band_LF)   # 14
        self.dnu_HF = self.bandwidth_HF / self.nchannels_HF
        self.dnu_LF = self.bandwidth_LF / self.nchannels_LF
        # z ranges
        self.z1_HF, self.z2_HF = (c/self.nu1_HF/self.wv_signal) - 1., (c/self.nu2_HF/self.wv_signal) - 1.
        self.z_cen_HF = (self.z1_HF + self.z2_HF) / 2.
        self.z1_LF, self.z2_LF = (c/self.nu1_LF/self.wv_signal) - 1., (c/self.nu2_LF/self.wv_signal) - 1.
        self.z_cen_LF = (self.z1_LF + self.z2_LF) / 2.

        if np.size(self.wv_signal) == 1:
            self.zs_HF = (c / self.band_HF/ self.wv_signal) - 1.
            self.zs_LF = (c / self.band_LF / self.wv_signal) - 1.

        # LOS lengths
        self.dco1_HF = self.cosm.ComovingRadialDistance(0.,self.z1_HF) / cm_per_mpc
        self.dco2_HF = self.cosm.ComovingRadialDistance(0.,self.z2_HF) / cm_per_mpc
        self.bandwidth_HF_Mpch = (self.dco1_HF - self.dco2_HF) * self.cosm.h70   # [Mpc/h]
        self.dnu_HF_Mpch = self.bandwidth_HF_Mpch / self.nchannels_HF            # [Mpc/h]
        self.dco1_LF = self.cosm.ComovingRadialDistance(0., self.z1_LF) / cm_per_mpc
        self.dco2_LF = self.cosm.ComovingRadialDistance(0., self.z2_LF) / cm_per_mpc
        self.bandwidth_LF_Mpch = (self.dco1_LF - self.dco2_LF) * self.cosm.h70   # [Mpc/h]
        self.dnu_LF_Mpch = self.bandwidth_LF_Mpch / self.nchannels_LF            # [Mpc/h]

        self.rofz_HF = self.cosm.ComovingRadialDistance(0.,self.z_cen_HF) / cm_per_mpc * self.cosm.h70  # [Mpc/h]
        self.rofz_LF = self.cosm.ComovingRadialDistance(0.,self.z_cen_LF) / cm_per_mpc * self.cosm.h70  # [Mpc/h]
        self.theta_FWHM_HF = 1.15 * (self.wv_signal * (1. + self.z_cen_HF)) / self.d_ap                 # [radians]
        self.theta_FWHM_HF_arcsec = self.theta_FWHM_HF * (180. / np.pi) * 3600.                         # [arcsec]
        self.theta_FWHM_LF = 1.15 * (self.wv_signal * (1. + self.z_cen_LF)) / self.d_ap                 # [radians]
        self.theta_FWHM_LF_arcsec = self.theta_FWHM_LF * (180. / np.pi) * 3600.                         # [arcsec]
        #self.area_beam_HF = (self.theta_FWHM_HF / 2.355 * self.rofz_HF)**2      # HF-band beam area at z_cen [(Mpc/h)^2]
        #self.area_beam_LF = (self.theta_FWHM_LF / 2.355 * self.rofz_LF)**2      # LF-band beam area at z_cen [(Mpc/h)^2]
        self.area_beam_HF = (self.theta_FWHM_HF * self.rofz_HF)**2      # HF-band beam area at z_cen [(Mpc/h)^2]
        self.area_beam_LF = (self.theta_FWHM_LF * self.rofz_LF)**2      # LF-band beam area at z_cen [(Mpc/h)^2]
        self.v_vox_HF = self.area_beam_HF * self.dnu_HF_Mpch            # Comoving volume of HF band [(Mpc/h)^3]
        self.v_vox_LF = self.area_beam_LF * self.dnu_LF_Mpch            # Comoving volume of LF band [(Mpc/h)^3]
        self.a_pix_HF = np.sqrt(self.area_beam_HF) * self.dnu_HF_Mpch   # Comoving x-z pixel area of HF band [(Mpc/h)^2]
        self.a_pix_LF = np.sqrt(self.area_beam_LF) * self.dnu_LF_Mpch   # Comoving x-z pixel area of LF band [(Mpc/h)^2]

        # -----------------------   inputs of thermal noise   ----------------------- #
        # Note: we quote the NEI numbers defined for one single-pol detector (NEI on sky per detector), and there are
        # n_feedhorns_inst=32 of them on sky at a given time. The effective NEI for a dual-pol detector is this
        # NEI/sqrt(2), and the effective number of feedhorns becomes 16.
        self.sigma_N_HF_meas = np.median(self.detnei[np.where(self.sciband_high)])   # HF mean NEI, [Jy/sr s^(1/2)]
        self.sigma_N_LF_meas = np.median(self.detnei[np.where(self.sciband_low)])    # LF mean NEI, [Jy/sr s^(1/2)]
        #self.sigma_N_HF = 1.0e7   # mean from proposal, close to measured value at z=6, [Jy/sr s^(1/2)]
        #self.sigma_N_LF = 5.0e6
        self.sigma_N_HF = self.pf.sensitivity_params['sens_sigma_N_HF']
        self.sigma_N_LF = self.pf.sensitivity_params['sens_sigma_N_LF']

        # -----------------------   cross-correlation helpers   ----------------------- #
        if np.size(self.wv_signal)==2:
            self._zlims_all = np.sort(np.vstack((self.z1_HF, self.z2_HF, self.z1_LF, self.z2_LF)), axis=0)
            # Each signal's index of corresponding z_lim row number (-1 for MAX, 0 for MIN)
            self._rnum = ((self.wv_signal == min(self.wv_signal))*1. - 1.).astype(int)
            self._zlims = np.sort([self._zlims_all[self._rnum[0], 0], self._zlims_all[self._rnum[1], 1]])

            self._z_all = np.outer(c / np.hstack((self.band_HF, self.band_LF)), 1./self.wv_signal) - 1.
            self.n1_overlap = np.size(np.where((self._z_all[:,0] >= self._zlims[0])*(self._z_all[:,0] <= self._zlims[1])))
            self.n2_overlap = np.size(np.where((self._z_all[:,1] >= self._zlims[0])*(self._z_all[:,1] <= self._zlims[1])))
            print('n1: ', self.n1_overlap)
            print('n2: ', self.n2_overlap)
            assert self.n1_overlap == self.n2_overlap
            self.nz_overlap = self.n1_overlap



        # ---------------------------   customized experimental setup   --------------------------- #
        self.custom_delta_nu = 0.5 * 1.0e9   # Frequency resolution in [Hz]
        self.custom_nu_min = 200. * 1.0e9
        self.custom_nu_max = 300. * 1.0e9
        self.custom_nu_list = np.linspace(self.custom_nu_min, self.custom_nu_max, int((self.custom_nu_max-self.custom_nu_min)/self.custom_delta_nu))
        self.custom_NEFD = 5.0e-3   # in [Jy s^(1/2)]

        # Judge the type of signal (auto/cross) and determine the z range for sensitivity analysis
        if np.size(self.wv_signal) == 1:
            self.custom_z_list = (c / self.custom_nu_list / self.wv_signal) - 1.
            self.custom_Delta_z = 1.
        elif np.size(self.wv_signal) == 2:
            self.custom_z_list_a = (c / self.custom_nu_list / self.wv_signal[0]) - 1.
            self.custom_z_list_b = (c / self.custom_nu_list / self.wv_signal[1]) - 1.
        else:
            raise ValueError('Can take either 1 (auto) or 2 (cross) wavelengths!')

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
    def survey_goemetry(self):
        if not hasattr(self, '_survey_goemetry'):
            raise ValueError('must set a survey geometry!')
        return self._survey_goemetry


    @survey_goemetry.setter
    def survey_goemetry(self, value):
        if (np.alltrue(value>0)) and (np.size(value)==3):
            self._survey_goemetry = value
            print('updated default geometry to %s!'%(self._survey_goemetry))
        else:
            raise ValueError('input survey geometry invalid!')


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
    def t_obs_pix(self):
        #return self.t_obs_survey / (self.n_ch_x * self.n_ch_y * 2.355**(np.sum(np.array([self.n_ch_x,self.n_ch_y])>1.)) / self.n_feedhorns_inst)
        return self.t_obs_survey / (self.n_ch_x * self.n_ch_y * 1.**(np.sum(np.array([self.n_ch_x, self.n_ch_y]) > 1.)) / self.n_feedhorns_inst)


    @property
    def p3dnoise_survey_1deg_HF(self):
        """
        3D thermal noise
        ----------------------------------------
        :return: 3D thermal noise (of a given voxel volume) in [(Jy/sr)^2]
        """
        return (self.sigma_N_HF**2) * self.v_vox_HF / self.t_obs_pix


    @property
    def p3dnoise_survey_1deg_LF(self):
        """
        3D thermal noise
        ----------------------------------------
        :return: 3D thermal noise (of a given voxel volume) in [(Jy/sr)^2]
        """
        return (self.sigma_N_LF**2) * self.v_vox_LF / self.t_obs_pix


    def beam_size_at_z(self, z, physical=True):
        """
        Beam size at a given redshift
        ----------------------------------------
        :param z: redshift
        :return: beam size in [cMpc/h] or [radian]
        """
        _rofz = self.cosm.ComovingRadialDistance(0., z) / cm_per_mpc * self.cosm.h70
        if np.size(self.wv_signal) == 1:
            _theta_FWHM = 1.15 * (self.wv_signal * (1. + z)) / self.d_ap
        elif np.size(self.wv_signal) == 2:
            _theta_FWHM = 1.15 * (np.mean(self.wv_signal) * (1. + z)) / self.d_ap
        else:
            raise NotImplementedError('oops!')
        _theta = _theta_FWHM / 2.355
        if physical:
            return _rofz * _theta
        else:
            return _theta


    def custom_V_survey(self, z):
        """
        Survey volume
        ----------------------------------------
        :param z: redshift
        :return: survey volume in [(Mpc/h)^3]
        """
        if np.size(self.wv_signal) == 1:
            _izmin = np.argmin(abs(self.custom_z_list - (z-self.custom_Delta_z/2.)))
            _izmax = np.argmin(abs(self.custom_z_list - (z + self.custom_Delta_z / 2.)))
            _zmin = self.custom_z_list[_izmin]
            _zmax = self.custom_z_list[_izmax]
            _Lz = self.cosm.ComovingRadialDistance(_zmin, _zmax) / cm_per_mpc * self.cosm.h70   # in [cMpc/h]
            _lx = _ly = self.beam_size_at_z(z) * 2.355   # in [cMpc/h]
            return (_lx * self.n_ch_x) * (_ly * self.n_ch_y) * _Lz
        elif np.size(self.wv_signal) == 2:
            _izmin_a = np.argmin(self.custom_z_list_a)
            _izmax_a = np.argmax(self.custom_z_list_a)
            _izmin_b = np.argmin(self.custom_z_list_b)
            _izmax_b = np.argmax(self.custom_z_list_b)
            _zmin = np.max(self.custom_z_list_a[_izmin_a], self.custom_z_list_b[_izmin_b])
            _zmax = np.min(self.custom_z_list_a[_izmax_a], self.custom_z_list_b[_izmax_b])
        else:
            raise NotImplementedError('oops!')


    def custom_V_voxel(self, z):
        """
        Voxel volume
        ----------------------------------------
        :param z: redshift
        :return:
        """
        if np.size(self.wv_signal) == 1:
            _y = self.wv_signal * (1.+z)**2 / self.cosm.HubbleParameter(z)
            _Lz = _y * self.custom_delta_nu  / cm_per_mpc * self.cosm.h70   # in [cMpc/h]
            _Lx = _Ly = self.beam_size_at_z(z)   # in [cMpc/h]
            return _Lx * _Ly * _Lz
        else:
            raise NotImplementedError('oops!')


    def custom_Nmodes(self, k, z, dlnk=0.2):
        """
        Analytical expression for number of modes
        ----------------------------------------
        :param k: wavenumber in [h/Mpc]
        :param z: redshift
        :param dlnk: bin size in lnk
        :return: number of k modes
        """
        return k**3 * dlnk * self.custom_V_survey(z) / (4.*np.pi**2)


    def custom_p3dnoise(self, z):
        """
        3D thermal noise power
        ----------------------------------------
        :return: average 3D power of thermal noise in [(Jy/sr)^2 (Mpc/h)^3]
        """
        _t_obs = self.t_obs_pix
        _sigma_noise = self.custom_NEFD / self.beam_size_at_z(z, physical=False)**2 / np.sqrt(_t_obs)
        return _sigma_noise**2 * self.custom_V_voxel(z)


    def get_kgrid(self, z, inherit=False, return_full=False):
        """
        Return k grid
        ----------------------------------------
        :param z: redshift of target signal, for determination of which sub-band the signal belongs to; {scalar}
        :param inherit: whether to adopt Bade's numbers; {boolean}
        :return: kx, ky, kz (and optionally comoving sizes of survey volume lx, ly, lz); {tuple}
        """

        #assert np.logical_not(inherit)

        # First, make sure the signal(s) indeed falls into TIME's bandpass
        print('check: ', c/self.wv_signal/(1.+z))
        if not all(c/self.wv_signal/(1.+z) > self.nu1_LF) and all(c/self.wv_signal/(1.+z) < self.nu2_HF):
            raise ValueError('At least 1 line of rest-frame wavelength in %s cm from z=%.1f outside TIME bandpass!'%(self.wv_signal,z))

        # Determine which sub-band the signal belongs to
        _hf = c/self.wv_signal/(1.+z) > self.nu1_HF
        print('_hf:', _hf)

        ndim = sum(self.survey_goemetry > 1)

        if ndim==2:
            if inherit:
                _lx = np.array([0.62960]) * self.n_ch_x
                _ly = np.array([0.62960]) * self.n_ch_y
                _lz = 787.930
            else:
                if np.size(self.wv_signal) == 1:
                    if self.wv_signal[0] == lambda_CII:
                        if _hf.flatten():
                            # cheating... this was the inaccurate value used for wf_num
                            _lx = np.round(self.area_beam_HF**0.5, 4) * self.n_ch_x
                            _ly = np.round(self.area_beam_HF**0.5, 4) * self.n_ch_y
                            _lz = 270.37
                        else:
                            _lx = np.round(self.area_beam_LF**0.5, 4) * self.n_ch_x
                            _ly = np.round(self.area_beam_LF**0.5, 4) * self.n_ch_y
                            _lz = 499.05
                    else:
                        _lx = np.round(self.area_beam_HF**0.5, 4) * self.n_ch_x
                        _ly = np.round(self.area_beam_HF**0.5, 4) * self.n_ch_y
                        print('z2_HF:', self.z2_HF, 'z1_LF:', self.z1_LF)
                        _lz = self.cosm.ComovingRadialDistance(self.z2_HF, self.z1_LF) / cm_per_mpc * self.cosm.h70
                else:
                    _lx = _hf * self.area_beam_HF**0.5 * self.n_ch_x + (1.-_hf) * self.area_beam_LF**0.5 * self.n_ch_x
                    _ly = _hf * self.area_beam_HF**0.5 * self.n_ch_y + (1.-_hf) * self.area_beam_LF**0.5 * self.n_ch_y
                    _lz = self.cosm.ComovingRadialDistance(self._zlims[0], self._zlims[1]) / cm_per_mpc * self.cosm.h70

            print('lx:', _lx, 'dx:', _lx/self.n_ch_x)
            print('lz:', _lz)

            _dx = _lx / self.n_ch_x
            _dz = _lz / self.n_ch_z

            _nx_sim = self.n_ch_x
            _nz_sim = self.n_ch_z

            #_kx = 2 * np.pi * np.fft.fftfreq(nx_sim, _dx)
            #_kx = np.roll(_kx, -nx_sim / 2 - nx_sim % 2, axis=0)
            #_kz = 2 * np.pi * np.fft.fftfreq(nz_sim, _dz)
            #_kz = np.roll(_kz, -nz_sim / 2 - nz_sim % 2, axis=0)
            _kx = []
            _ky = []
            _kz = []
            for ii in xrange(np.size(self.wv_signal)):
                _kxi = 2 * np.pi * np.fft.fftfreq(_nx_sim, _dx[ii])
                _kxi = np.roll(_kxi, -_nx_sim / 2 - _nx_sim % 2, axis=0)
                _kx.append(_kxi)

                _kzi = 2 * np.pi * np.fft.fftfreq(_nz_sim, _dz)
                _kzi = np.roll(_kzi, -_nz_sim / 2 - _nz_sim % 2, axis=0)
                _kz.append(_kzi)

            if np.size(self.wv_signal) == 1:
                if return_full: return _kx[0], _kz[0], _lx[0], _ly[0], _lz[0]
                else: return _kx[0], _kz[0]
            else:
                if return_full: return _kx, _kz, _lx, _ly, _lz
                else: return _kx, _kz

        elif ndim==3:
            if inherit:
                _lx = np.array([0.62960]) * self.n_ch_x
                _ly = np.array([0.62960]) * self.n_ch_y
                _lz = 787.930
            else:
                if np.size(self.wv_signal) == 1:
                    if _hf.flatten():
                        _lx = self.area_beam_HF**0.5 * self.n_ch_x
                        _ly = self.area_beam_HF**0.5 * self.n_ch_y
                        _lz = 270.37
                    else:
                        _lx = self.area_beam_LF**0.5 * self.n_ch_x
                        _ly = self.area_beam_LF**0.5 * self.n_ch_y
                        _lz = 499.05
                    #_lz = self.cosm.ComovingRadialDistance(self.z2_HF, self.z1_LF) / cm_per_mpc * self.cosm.h70
                else:
                    _lx = _hf * self.area_beam_HF**0.5 * self.n_ch_x + (1.-_hf) * self.area_beam_LF**0.5 * self.n_ch_x
                    _ly = _hf * self.area_beam_HF**0.5 * self.n_ch_y + (1.-_hf) * self.area_beam_LF**0.5 * self.n_ch_y
                    _lz = self.cosm.ComovingRadialDistance(self._zlims[0], self._zlims[1]) / cm_per_mpc * self.cosm.h70

            print('lx:', _lx, _lx / self.n_ch_x)
            print('ly:', _ly)
            print('lz:', _lz)

            _nx_sim = self.n_ch_x
            _ny_sim = self.n_ch_y
            # For cross-correlation, we must consider only the overlapped spectral channels
            if hasattr(self, 'n1_overlap'): _nz_sim = self.nz_overlap
            else: _nz_sim = self.n_ch_z

            _dx = _lx / _nx_sim
            _dy = _ly / _ny_sim
            _dz = _lz / _nz_sim

            _kx = []
            _ky = []
            _kz = []
            for ii in xrange(np.size(self.wv_signal)):
                _kxi = 2 * np.pi * np.fft.fftfreq(_nx_sim, _dx[ii])
                _kxi = np.roll(_kxi, -_nx_sim / 2 - _nx_sim % 2, axis=0)
                _kx.append(_kxi)
                _kyi = 2 * np.pi * np.fft.fftfreq(_ny_sim, _dy[ii])
                _kyi = np.roll(_kyi, -_ny_sim / 2 - _ny_sim % 2, axis=0)
                _ky.append(_kyi)
                _kzi = 2 * np.pi * np.fft.fftfreq(_nz_sim, _dz)
                _kzi = np.roll(_kzi, -_nz_sim / 2 - _nz_sim % 2, axis=0)
                _kz.append(_kzi)

            if np.size(self.wv_signal)==1:
                if return_full: return _kx[0], _ky[0], _kz[0], _lx, _ly, _lz
                else: return _kx[0], _ky[0], _kz[0]
            else:
                if return_full: return _kx, _ky, _kz, _lx, _ly, _lz
                else: return _kx, _ky, _kz
        else:
            raise ValueError('Dimension (either 2D or 3D) of survey geometry does not match!')


    def get_kgrid_ciilae(self, z, z_l, z_h, return_full=False):
        """
        Return k grid for [CII]-LAE cross correlation
        ----------------------------------------
        :param z: redshift of target signal, for determination of which sub-band the signal belongs to; {scalar}
        :return: kx, ky, kz (and optionally comoving sizes of survey volume lx, ly, lz); {tuple}
        """

        # First, make sure the signal(s) indeed falls into TIME's bandpass
        if not all(c/self.wv_signal/(1.+z) > self.nu1_LF) and all(c/self.wv_signal/(1.+z) < self.nu2_HF):
            raise ValueError('At least 1 line of rest-frame wavelength in %s cm from z=%.1f outside TIME bandpass!'%(self.wv_signal,z))

        ndim = sum(self.survey_goemetry > 1)

        _lx = np.round(self.area_beam_HF ** 0.5, 4) * self.n_ch_x
        _ly = np.round(self.area_beam_HF ** 0.5, 4) * self.n_ch_y
        _lz = self.cosm.ComovingRadialDistance(z_l, z_h) / cm_per_mpc * self.cosm.h70

        if ndim == 1:
            _dx = _lx / self.n_ch_x

            nx_sim = self.n_ch_x

            _kx = 2 * np.pi * np.fft.fftfreq(nx_sim, _dx)
            _kx = np.roll(_kx, -nx_sim/2 - nx_sim%2, axis=0)

            if return_full:
                return _kx, _lx, _ly, _lz
            else:
                return _kx
        else:
            _dx = _lx / self.n_ch_x
            _dy = _ly / self.n_ch_y

            nx_sim = self.n_ch_x
            ny_sim = self.n_ch_y

            _kx = 2 * np.pi * np.fft.fftfreq(nx_sim, _dx)
            _kx = np.roll(_kx, -nx_sim/2 - nx_sim%2, axis=0)
            _ky = 2 * np.pi * np.fft.fftfreq(ny_sim, _dy)
            _ky = np.roll(_ky, -ny_sim/2 - ny_sim%2, axis=0)

            if return_full:
                return _kx, _ky, _lx, _ly, _lz
            else:
                return _kx, _ky


    def get_Nmodes_3D_custom(self, kx_HF, ky_HF, kz_HF, k3d_binctr, zapping=False):
        """
        Number counts of Fourier modes for a 3D (kx, ky, kz) grid and user-specified |k| bin centers
        ----------------------------------------
        :param kx_HF: kx vector; {1d arr}
        :param ky_HF: ky vector; {1d arr}
        :param kz_HF: kz vector; {1d arr}
        :param k3d_binctr: centers of k magnitude bins, doesn't have to be uniform; {1d arr}
        :return: |k| bin centers, numbers of modes, lower/higher |k| bin edge; {tuple}
        """

        # Must be called for an actual 3D survey
        assert (self.n_ch_x > 1) and (self.n_ch_y > 1) and (self.n_ch_z > 1)

        nkx_HF = np.size(kx_HF)
        nky_HF = np.size(ky_HF)
        nkz_HF = np.size(kz_HF)

        # Judge if the total number of combinations is odd or even
        is_odd = (nkx_HF * nky_HF * nkz_HF) % 2 == 1
        print('3D is_odd:', is_odd)

        nbins = np.size(k3d_binctr)
        nmode_HF = np.zeros(nbins)

        kbin_centers = k3d_binctr

        dlogkbin = np.log10(k3d_binctr[1]) - np.log10(k3d_binctr[0])
        logkbin_min = np.log10(k3d_binctr[0]) - dlogkbin/2.

        logkbin = np.hstack((np.log10(kbin_centers) - dlogkbin/2., np.log10(kbin_centers[-1]) + dlogkbin/2.))
        kbin = 10**logkbin

        # now we are ready to count and bin modes.
        # we step through the 2-d slab with k magnitudes, and place every k magnitude we encoutner in the appropriate bin
        # important note: Count only independent modes. Modes from the lower half plane modes are discarded,
        # since they provide only redundant information on the P(k) from the upper half plane. This is because the
        # observed field is real-valued: delta*(k_x,k_y,k_z) = delta(-k_x,-k_y,-k_z)

        if is_odd:
            counter = []
            for l in range(nkx_HF):
                for m in range(nky_HF):
                    for n in range(nkz_HF):
                        if ([kx_HF[l], ky_HF[m], kz_HF[n]] not in counter) and ([-kx_HF[l], -ky_HF[m], -kz_HF[n]] not in counter):
                            counter.append([kx_HF[l], ky_HF[m], kz_HF[n]])
                            kmagnitude_HF = np.sqrt(kx_HF[l]**2 + ky_HF[m]**2 + kz_HF[n]**2)
                            if kmagnitude_HF > 0.:
                                loc = int(np.floor(abs((np.log10(kmagnitude_HF) - logkbin_min)) / dlogkbin))
                                if loc < nbins:
                                    nmode_HF[loc] += 1
                                else:
                                    warnings.warn("kmag not included in the specified k3d_binctr range!")
                                    pass
        else:
            counter = []
            axis_to_halve = np.where(np.array([nkx_HF, nky_HF, nkz_HF])%2 == 0)[0][0]
            zero_locs = np.array([np.where(kx_HF==0.), np.where(ky_HF==0.), np.where(kz_HF==0.)]).squeeze()
            counter_start = (np.zeros(3)).astype(int)
            counter_start[axis_to_halve] = zero_locs[axis_to_halve].astype(int)
            for l in range(counter_start[0], nkx_HF):
                for m in range(counter_start[1], nky_HF):
                    for n in range(counter_start[2], nkz_HF):
                        if zapping:
                            if abs(kz_HF[n]) <= min(abs(kz_HF[counter_start[2]::])) or \
                                np.sqrt(kx_HF[l]**2 + ky_HF[m]**2) <= min(np.sqrt(kx_HF[counter_start[0]::][:,np.newaxis]**2 +
                                                                                  ky_HF[counter_start[1]::][np.newaxis,:]**2).flatten()):
                                pass
                            else:
                                counter.append([kx_HF[l], ky_HF[m], kz_HF[n]])
                                kmagnitude_HF = np.sqrt(kx_HF[l] ** 2 + ky_HF[m] ** 2 + kz_HF[n] ** 2)
                                if kmagnitude_HF > 0.:
                                    loc = int(np.floor(abs((np.log10(kmagnitude_HF) - logkbin_min)) / dlogkbin))
                                    if loc < nbins:
                                        nmode_HF[loc] += 1
                                    else:
                                        warnings.warn("kmag not included in the specified k3d_binctr range!")
                                        pass
                        else:
                            counter.append([kx_HF[l], ky_HF[m], kz_HF[n]])
                            kmagnitude_HF = np.sqrt(kx_HF[l]**2 + ky_HF[m]**2 + kz_HF[n]**2)
                            if kmagnitude_HF > 0.:
                                loc = int(np.floor(abs((np.log10(kmagnitude_HF) - logkbin_min)) / dlogkbin))
                                if loc < nbins:
                                    nmode_HF[loc] += 1
                                else:
                                    warnings.warn("kmag not included in the specified k3d_binctr range!")
                                    pass
        #print 'counter:', counter
        n_output = nmode_HF[np.where(nmode_HF > 0)]
        k_output = kbin_centers[np.where(nmode_HF > 0)]
        k_edge_output_l = kbin[np.where(nmode_HF > 0)]
        k_edge_output_h = kbin[np.where(nmode_HF > 0)[0]+1]

        return [k_output, n_output, k_edge_output_l, k_edge_output_h]


    def get_Nmodes_2D_custom(self, k1, k2, k2d_binctr, zapping=False):
        """
        Number counts of Fourier modes for a 2D (k1, k2) grid and user-specified |k| bin centers
        ----------------------------------------
        :param k1: k1 vector; {1d arr}
        :param k2: k2 vector; {1d arr}
        :param k2d_binctr: centers of k magnitude bins, doesn't have to be uniform; {1d arr}
        :return: |k| bin centers, numbers of modes, lower/higher |k| bin edge; {tuple}
        """

        # Must be called for an actual 2D survey
        assert int(self.n_ch_x > 1) + int(self.n_ch_y > 1) + int(self.n_ch_z > 1) == 2

        nk1 = np.size(k1)
        nk2 = np.size(k2)

        # Judge if the total number of combinations is odd or even
        is_odd = (nk1 * nk2) % 2 == 1
        print('2D is_odd:', is_odd)

        nbins = np.size(k2d_binctr)
        nmode = np.zeros(nbins)

        kbin_centers = k2d_binctr

        dlogkbin = np.log10(k2d_binctr[1]) - np.log10(k2d_binctr[0])
        logkbin_min = np.log10(k2d_binctr[0]) - dlogkbin/2.

        logkbin = np.hstack((np.log10(kbin_centers) - dlogkbin/2., np.log10(kbin_centers[-1]) + dlogkbin/2.))
        kbin = 10**logkbin

        # now we are ready to count and bin modes.
        # we step through the 2-d slab with k magnitudes, and place every k magnitude we encoutner in the appropriate bin
        # important note: Count only independent modes. Modes from the lower half plane modes are discarded,
        # since they provide only redundant information on the P(k) from the upper half plane. This is because the
        # observed field is real-valued: delta*(k_x,k_y,k_z) = delta(-k_x,-k_y,-k_z)

        if is_odd:
            counter = []
            for l in range(nk1):
                for m in range(nk2):
                    if ([k1[l], k2[m]] not in counter) and ([-k1[l], -k2[m]] not in counter):
                        counter.append([k1[l], k2[m]])
                        kmagnitude_HF = np.sqrt(k1[l]**2 + k2[m]**2)
                        if kmagnitude_HF > 0.:
                            loc = int(np.floor(abs((np.log10(kmagnitude_HF) - logkbin_min)) / dlogkbin))
                            nmode[loc] += 1
        else:
            counter = []
            axis_to_halve = np.where(np.array([nk1, nk2])%2 == 0)[0][0]
            zero_locs = np.array([np.where(k1==0.), np.where(k2==0.)]).squeeze()
            counter_start = (np.zeros(2)).astype(int)
            counter_start[axis_to_halve] = zero_locs[axis_to_halve].astype(int)
            for l in range(counter_start[0], nk1):
                    for m in range(counter_start[1], nk2):
                        if zapping:
                            if abs(k2[m]) <= min(abs(k2[counter_start[1]::])) or \
                               abs(k1[l]) <= min(abs(k1[counter_start[0]::]).flatten()):
                                #print 'yes, zapped!'
                                pass
                            else:
                                counter.append([k1[l], k2[m]])
                                kmagnitude_HF = np.sqrt(k1[l]**2 + k2[m]**2)
                                if kmagnitude_HF > 0.:
                                    loc = int(np.floor(abs((np.log10(kmagnitude_HF) - logkbin_min)) / dlogkbin))
                                    if loc < nbins:
                                        nmode[loc] += 1
                                    else:
                                        warnings.warn("kmag = %.3f not included in the specified k2d_binctr range!"%kmagnitude_HF)
                                        pass
                        else:
                            counter.append([k1[l], k2[m]])
                            kmagnitude_HF = np.sqrt(k1[l]**2 + k2[m]**2)
                            if kmagnitude_HF > 0.:
                                loc = int(np.floor(abs((np.log10(kmagnitude_HF) - logkbin_min)) / dlogkbin))
                                if loc < nbins:
                                    nmode[loc] += 1
                                else:
                                    warnings.warn("kmag = %.3f not included in the specified k2d_binctr range!"%kmagnitude_HF)
                                    pass

        n_output = nmode[np.where(nmode > 0)]
        k_output = kbin_centers[np.where(nmode > 0)]
        k_edge_output_l = kbin[np.where(nmode > 0)]
        k_edge_output_h = kbin[np.where(nmode > 0)[0]+1]

        return [k_output, n_output, k_edge_output_l, k_edge_output_h]


    def get_Nmodes_1D_custom(self, k1, k1d_binctr, zapping=False):
        """
        Number counts of Fourier modes for a 1D (k1,) grid and user-specified |k| bin centers
        ----------------------------------------
        :param k1: k1 vector; {1d arr}
        :param k1d_binctr: centers of k magnitude bins, doesn't have to be uniform; {1d arr}
        :return: |k| bin centers, numbers of modes, lower/higher |k| bin edge; {tuple}
        """

        # Must be called for an actual 2D survey
        assert int(self.n_ch_x > 1) + int(self.n_ch_y > 1) + int(self.n_ch_z > 1) == 1

        nk1 = np.size(k1)

        nbins = np.size(k1d_binctr)
        nmode = np.zeros(nbins)

        kbin_centers = k1d_binctr

        dlogkbin = np.log10(k1d_binctr[1]) - np.log10(k1d_binctr[0])
        logkbin_min = np.log10(k1d_binctr[0]) - dlogkbin/2.

        logkbin = np.hstack((np.log10(kbin_centers) - dlogkbin/2., np.log10(kbin_centers[-1]) + dlogkbin/2.))
        kbin = 10**logkbin

        # now we are ready to count and bin modes.
        # we step through the 2-d slab with k magnitudes, and place every k magnitude we encoutner in the appropriate bin
        # important note: Count only independent modes. Modes from the lower half plane modes are discarded,
        # since they provide only redundant information on the P(k) from the upper half plane. This is because the
        # observed field is real-valued: delta*(k_x,k_y,k_z) = delta(-k_x,-k_y,-k_z)

        counter = []
        for l in range(nk1):
            if (k1[l] not in counter) and (-k1[l] not in counter):
                counter.append(k1[l])
                kmagnitude_HF = np.sqrt(k1[l]**2)
                if kmagnitude_HF > 0.:
                    loc = int(np.floor(abs((np.log10(kmagnitude_HF) - logkbin_min)) / dlogkbin))
                    nmode[loc] += 1

        n_output = nmode[np.where(nmode > 0)]
        k_output = kbin_centers[np.where(nmode > 0)]
        k_edge_output_l = kbin[np.where(nmode > 0)]
        k_edge_output_h = kbin[np.where(nmode > 0)[0]+1]

        return [k_output, n_output, k_edge_output_l, k_edge_output_h]