# Here we use the last column of Table 4 of "Planck 2015 Results: XIII. Cosmological Parameters"
_cosmo_params = \
{
'omega_m_0': 0.3089,
'omega_b_0': round(0.0223/0.6774**2, 5),
'omega_l_0': 1. - 0.3089,
'hubble_0': 0.6774,
'helium_by_number': 0.0813,
'helium_by_mass': 0.2453,
'cmb_temp_0': 2.7255,
'sigma_8': 0.8159,
'primordial_index': 0.9667,
'approx_highz': False
}

#'''
# USE THIS ONE FOR TIME SCIENCE PAPER
_hmf_params = \
{
#'hmf_tbl': '/input/hmf_tbl/TimeScience/hmf_ST_wrt_mean_logM_1000_6-16_z_301_0-30.npz',
'hmf_tbl': '/input/hmf_tbl/TimeScience/hmf_Tinker08_wrt_vir_logM_1000_6-16_z_301_0-30.npz',
'hmf_analytic': False,
#'hmf_model': 'ST',
'hmf_model': 'Tinker08',
#'hmf_delta_wrt': 'mean',
'hmf_delta_wrt': 'vir',
'hmf_logMmin': 6.0,
'hmf_logMmax': 16.0,
'hmf_zmin': 0.0,
'hmf_zmax': 30.0,
'hmf_dlogM': 0.01,
'hmf_dz': 0.1,
'hmf_dlna': 2e-6,
'hmf_dlnk': 1e-2,
'hmf_lnk_min': -20.,
'hmf_lnk_max': 10.,
'hmf_transfer_k_per_logint': 11,
'hmf_transfer_kmax': 100.,  # hmf default value is 5
'hmf_profile_p': 0.3,
'hmf_profile_q': 0.75
}
#'''

'''
# USE THIS ONE FOR MULTI-TRACER PAPER
_hmf_params = \
{
'hmf_tbl': '/input/hmf_tbl/TimeScience/hmf_Tinker08_wrt_vir_logM_1000_6-16_z_301_0-30.npz',
'hmf_analytic': False,
'hmf_model': 'Tinker08',
'hmf_delta_wrt': 'vir',
'hmf_logMmin': 7.0,
'hmf_logMmax': 16.0,
'hmf_zmin': 0.0,
'hmf_zmax': 30.0,
'hmf_dlogM': 0.01,
'hmf_dz': 0.1,
'hmf_dlna': 2e-6,
'hmf_dlnk': 1e-2,
'hmf_lnk_min': -20.,
'hmf_lnk_max': 10.,
'hmf_transfer_k_per_logint': 11,
'hmf_transfer_kmax': 100.,  # hmf default value is 5
'hmf_profile_p': 0.3,
'hmf_profile_q': 0.75
}
'''

_cibmodel_params = \
{
'cib_model': 'CIB:Cheng',
'cib_L0': [0.0135, 0.02],   # Note Heidi has 0.0135, while Yun-Ting has 0.02
'cib_T0': [24.4, 25.3],
'cib_alpha': [0.36, 0.0],
'cib_delta': [3.6, 2.6],
'cib_sigmasq_LM': [0.5, 0.5],
'cib_M_eff': [10**12.6, 10**12.6],
'cib_beta': [1.75, 1.5],
'cib_gamma': [1.7, 2.0],
'cib_zmin': [0.1, 0.1],   # minimum z CIB model is valid
'cib_zmax': [10.1, 10.1],   # maximum z CIB model is valid
}


_dust_params = \
{
'dust_mw_dg': 0.01,
'dust_sed_nu_ref': 8.57e11,
'dust_sed_emissivity_ref': 4.3e-21,
}


_sensitivity_params = \
{
'sens_t_obs_survey': 1.0e3 * 3600.,   # default exposure time [s]
'sens_n_feedhorns': 32.,              # number of feedhorns
'sens_d_ap': 12.0 * 1e2,              # effective aperture size (diameter) [cm]
'sens_read_tnoise': True,             # whether to read thermal noise from file
'sens_geom_x': 156,
'sens_geom_y': 1,
'sens_geom_z': 42,
'sens_lambda_signal': [1.578e-2],       # wavelength of the target (pair of) signal(s); {list}
'sens_sigma_N_HF': 1.0e7,
'sens_sigma_N_LF': 5.0e6,
}


_grf_params = \
{
'grf_d_ap': 12.0 * 1e2,              # effective aperture size (diameter) [cm]
'grf_geom_x': 156,
'grf_geom_y': 1,
'grf_geom_z': 42,
'grf_lambda_signal': 1.577e-2,       # wavelength of the target (pair of) signal(s); {scalar}
'grf_z_signal': 6.0,
'grf_ps_in': None,
}


_wf_params = \
{
'wf_type': 'analytical',
'wf_z_signal': 6.0,
'wf_n_logkbins': 20,
}


_ham_params = \
{
'uvlf_model': 'bouwens2015',
'dustcorr_method': None,   # or 'meurer1999', 'pettini1998', 'capak2015'
'dustcorr_beta': 'bouwens2014',
'dustcorr_scatter_A': 0.,
'dustcorr_scatter_B': 0.34,
'logMh_min': 8.,
'logMh_max': 14.,
'dmag': 0.1,
}