import numpy as np

# general constants
c = 29979245800.0 				                                # Speed of light - [c] = cm/s
G = 6.673e-8     				                                # Gravitational constant - [G] = cm^3/g/s^2
e = 1.60217646e-19   			                                # Electron charge - [e] = C
e_cgs = 4.803204e-10                                            # Electron charge - [e] = statC
m_e = 9.10938188e-28     		                                # Electron mass - [m_e] = g
m_p = 1.67262158e-24    		                                # Proton mass - [m_p] = g
m_n = 1.67492729e-24                                            # Neutron mass - [m_n] = g
k_B = 1.38065e-16                                               # Boltzmann constant - [k_B] = erg/K
k_B_SI = 1.38065e-23                                            # Boltzmann constant in SI units
h = h_p = h_P = 6.626068e-27                                    # Planck's constant - [h] = erg*s
h_p_SI = 6.626068e-34                                           # Planck's constant in SI units
h_bar = h / 2 / np.pi   			                            # H-bar - [h_bar] = erg*s
sigma_SB = 2.0 * np.pi**5 * k_B**4 / 15.0 / c**2 / h**3         # Stefan-Boltzmann constant - [sigma_SB] = erg/cm^2/deg^4/s
Ryd = 2.1798719e-11                                             # Rydberg in erg
sigma_T = 6.652e-25                                             # Thomson scattering cross section - [sigma_T] = cm^2

# --------------------------------------   Conversions   -------------------------------------- #

# lengths
km_per_pc = 3.08568e13
km_per_mpc = km_per_pc*1e6
km_per_gpc = km_per_mpc*1e3
cm_per_pc = km_per_pc*1e5
cm_per_kpc = cm_per_pc*1e3
cm_per_mpc = cm_per_pc*1e6
cm_per_gpc = cm_per_mpc*1e3
cm_per_km = 1e5
cm_per_rsun = 695500. * cm_per_km
cm_per_rEarth = 637100000.
cm_per_au = 1.49597871e13

# masses
g_per_amu = 1.660538921e-24
g_per_msun = 1.98892e33
mH_amu = 1.00794
mHe_amu = 4.002602

# energies
erg_per_j = 1e-7
erg_per_ev = e / erg_per_j
erg_per_kev = 1e3 * erg_per_ev

# times
s_per_yr = 365.25*24*3600
s_per_kyr = s_per_yr*1e3
s_per_myr = s_per_kyr*1e3
s_per_gyr = s_per_myr*1e3


# miscellaneous
lsun = 3.839e33                     # Solar luminosity - erg / s
watt_to_cgs = 1.0e7                 # watt to erg/s
jansky_to_cgs = 1.0e-23

# ------------------------ Properties of Emission Lines ------------------------ #
# Hydrogen
E_LL = Ryd / erg_per_ev
E_LyA = E_LL * (1. - 1. / 2**2)
E_LyB = E_LL * (1. - 1. / 3**2)
nu_alpha = E_LyA * erg_per_ev / h

# Other lines
nu_HI21 = 1.420406e9                # HI 21cm rest-frame frequency - Hz
lambda_HI21 = c / nu_HI21           # HI 21cm rest-frame wavelength - Hz
nu_CO_10 = 1.15271208e11            # CO(1-0) rest-frame frequency - Hz
lambda_CO_10 = c / nu_CO_10         # CO(1-0) rest-frame wavelength - cm
lambda_CO_32 = 8.669e-2             # CO(3-2) rest-frame wavelength - cm
nu_CO_32 = c / lambda_CO_32         # CO(3-2) rest-frame frequency - Hz
lambda_CO_43 = 6.502e-2             # CO(4-3) rest-frame wavelength - cm
nu_CO_43 = c / lambda_CO_43         # CO(4-3) rest-frame frequency - Hz
lambda_CO_54 = 5.202e-2             # CO(5-4) rest-frame wavelength - cm
nu_CO_54 = c / lambda_CO_54         # CO(5-4) rest-frame frequency - Hz
lambda_CO_65 = 4.335e-2             # CO(6-5) rest-frame wavelength - cm
nu_CO_65 = c / lambda_CO_65         # CO(6-5) rest-frame frequency - Hz
lambda_CO_76 = 3.716e-2             # CO(7-6) rest-frame wavelength - cm
nu_CO_76 = c / lambda_CO_76         # CO(7-6) rest-frame frequency - Hz

lambda_CI = 6.090e-2                # [CI] rest-framewavelength - cm
nu_CI = c / lambda_CI

lambda_CII = 157.8e-4               # [CII] rest-frame wavelength - cm
nu_CII = c / lambda_CII             # [CII] rest-frame frequency - Hz
lambda_NII122 = 121.89e-4           # [NII] 122 rest-frame wavelength - cm
nu_NII122 = c / lambda_NII122       # [NII] 122 rest-frame frequency - Hz
lambda_NII205 = 205.19e-4           # [NII] 205 rest-frame wavelength - cm
nu_NII205 = c / lambda_NII205       # [NII] 205 rest-frame frequency - Hz
lambda_H2S0 = 28.219e-4             # H2 S(0) rest-frame wavelength - cm
nu_H2S0 = c / lambda_H2S0           # H2 S(0) rest-frame frequency - Hz
lambda_H2S1 = 17.035e-4             # H2 S(1) rest-frame wavelength - cm
nu_H2S1 = c / lambda_H2S1           # H2 S(1) rest-frame frequency - Hz
lambda_H2S2 = 12.279e-4             # H2 S(2) rest-frame wavelength - cm
nu_H2S2 = c / lambda_H2S2           # H2 S(2) rest-frame frequency - Hz
lambda_H2S3 = 9.665e-4              # H2 S(3) rest-frame wavelength - cm
nu_H2S3 = c / lambda_H2S3           # H2 S(3) rest-frame frequency - Hz
lambda_HCN10 = 3.383e-1             # HCN(1-0) rest-frame wavelength - cm
nu_HCN10 = c / lambda_HCN10         # HCN(1-0) rest-frame frequency - Hz