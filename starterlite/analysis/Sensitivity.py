import numpy as np
import warnings
import os
from scipy.io.idl import readsav

from ..physics import Cosmology
from ..physics.Constants import c, cm_per_mpc
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