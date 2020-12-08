import numpy as np
from scipy.integrate import quad

from ..util.ParameterFile import ParameterFile
from .Constants import c, km_per_mpc, cm_per_mpc, g_per_msun, G, cm_per_kpc


class Cosmology(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

        self.omega_m_0 = self.pf.cosmo_params['omega_m_0']
        self.omega_b_0 = self.pf.cosmo_params['omega_b_0']
        self.omega_l_0 = self.pf.cosmo_params['omega_l_0']
        self.hubble_0 = self.pf.cosmo_params['hubble_0'] * 100 / km_per_mpc
        self.h70 = self.pf.cosmo_params['hubble_0']
        self.helium_by_number = self.pf.cosmo_params['helium_by_number']
        self.helium_by_mass = self.pf.cosmo_params['helium_by_mass']
        self.primordial_index = self.pf.cosmo_params['primordial_index']
        self.cmb_temp_0 = self.pf.cosmo_params['cmb_temp_0']
        self.sigma_8 = self.pf.cosmo_params['sigma_8']
        self.approx_highz = self.pf.cosmo_params['approx_highz']

        # Matter/Lambda equality
        #if self.omega_l_0 > 0:
        self.a_eq = (self.omega_m_0 / self.omega_l_0)**(1./3.)
        self.z_eq = 1. / self.a_eq - 1.

        self.CriticalDensityNow = self.rho_crit_0 = \
            (3 * self.hubble_0**2) / (8 * np.pi * G)

        # Mean total matter density in [Msun h^2 Mpc^-3]
        self.mean_density0 = self.omega_m_0 * self.rho_crit_0 * cm_per_mpc**3 / g_per_msun / self.h70**2

        # Mean baryonic matter density in [Msun h^2 Mpc^-3]
        self.mean_bdensity0 = self.omega_b_0 * self.rho_crit_0 * cm_per_mpc**3 / g_per_msun / self.h70**2


    def EvolutionFunction(self, z):
        return self.omega_m_0 * (1.0 + z) ** 3 + self.omega_l_0


    def HubbleParameter(self, z):
        if self.approx_highz:
            return self.hubble_0 * np.sqrt(self.omega_m_0) * (1. + z) ** 1.5
        return self.hubble_0 * np.sqrt(self.EvolutionFunction(z))


    def OmegaMatter(self, z):
        if self.approx_highz:
            return 1.0
        return self.omega_m_0 * (1. + z) ** 3 / self.EvolutionFunction(z)


    def OmegaLambda(self, z):
        if self.approx_highz:
            return 0.0


    def Dp(self, z):
        _integrand = lambda zp: (1. + zp) / (self.HubbleParameter(zp) / self.HubbleParameter(0.))**3
        if np.isscalar(z):
            temp = quad(_integrand, z, 3000.)[0]
        else:
            temp = np.array([quad(_integrand, i, 3000.)[0] for i in z])
        temp *= 2.5 * self.omega_m_0 * self.HubbleParameter(z) / self.HubbleParameter(0.)
        return temp


    def D(self, z):
        """
        Growth factor
        ----------------------------
        :param z: redshift
        :return: growth factor
        """
        return self.Dp(z) / self.Dp(0.)


    def t_of_z(self, z):
        """
        Time-redshift relation for a matter + lambda Universe.

        References
        ----------
        Ryden, Equation 6.28

        Returns
        -------
        Time since Big Bang in seconds.

        """
        # if self.approx_highz:
        #    pass
        # elif self.approx_lowz:
        #    pass

        # Full calculation
        a = 1. / (1. + z)
        t = (2. / 3. / np.sqrt(1. - self.omega_m_0)) \
            * np.log((a / self.a_eq) ** 1.5 + np.sqrt(1. + (a / self.a_eq) ** 3.)) \
            / self.hubble_0

        return t


    def dtdz(self, z):
        return 1. / self.HubbleParameter(z) / (1. + z)


    def LookbackTime(self, z_i, z_f):
        """
        Returns lookback time from z_i to z_f in seconds, where z_i < z_f.
        """
        return self.t_of_z(z_i) - self.t_of_z(z_f)


    def LuminosityDistance(self, z):
        """
        Returns luminosity distance in cm.  Assumes we mean distance from us (z = 0).
        """
        integr = quad(lambda z: self.hubble_0 / self.HubbleParameter(z),
                      0.0, z)[0]
        return integr * c * (1. + z) / self.hubble_0


    def ComovingRadialDistance(self, z0, z):
        """
        Return comoving distance between redshift z0 and z, z0 < z.
        ----------------------------------------
        :param z0: reference redshift
        :param z: source redshift
        :return: comoving radial distance in [cm]
        """
        if np.isscalar(z):
            if self.approx_highz:
                temp = 2. * c * ((1. + z0)**-0.5 - (1. + z)**-0.5) / self.hubble_0 / np.sqrt(self.omega_m_0)
            else:
                # Otherwise, do the integral - normalize to H0 for numerical reasons
                integrand = lambda z: self.hubble_0 / self.HubbleParameter(z)
                temp = c * quad(integrand, z0, z)[0] / self.hubble_0
            return temp
        else:
            temp = np.zeros_like(z)
            for i, z_i in enumerate(z):
                if self.approx_highz:
                    temp[i] = 2. * c * ((1. + z0)**-0.5 - (1. + z_i)**-0.5) / self.hubble_0 / np.sqrt(self.omega_m_0)
                else:
                    # Otherwise, do the integral - normalize to H0 for numerical reasons
                    integrand_i = lambda z: self.hubble_0 / self.HubbleParameter(z)
                    temp[i] = c * quad(integrand_i, z0, z_i)[0] / self.hubble_0
            return temp


    def ProperRadialDistance(self, z0, z):
        return self.ComovingRadialDistance(z0, z) / (1. + z0)


    def CriticalDensity(self, z):
        return (3.0 * self.HubbleParameter(z)**2) / (8.0 * np.pi * G)


    def CriticalDensityForCollapse(self, z):
        """
        Generally denoted (in LaTeX format) \Delta_c, fit from
        Bryan & Norman (1998), w.r.t. critical density
        """
        d = self.OmegaMatter(z) - 1.
        return 18. * np.pi**2 + 82. * d - 39. * d**2


    def VirialMass(self, T, z, mu=0.6):
        """
        Virial mass
        """
        m_vir = (1e8/self.h70) * (T/1.98e4)**1.5 * (mu/0.6)**-1.5
        m_vir *= (self.omega_m_0 * self.CriticalDensityForCollapse(z) / self.OmegaMatter(z) / 18. / np.pi**2)**-0.5
        m_vir *= ((1.+z) / 10.)**-1.5

        return m_vir


    def VirialRadius(self, M, z):
        """
        Virial radius in cgs units for a halo collapsing at z>>1
        """
        r_vir = (self.omega_m_0 / self.OmegaMatter(z) * self.CriticalDensityForCollapse(z) / (18.*np.pi**2))**(-1./3.)
        r_vir *= (M / 1.0e8)**(1./3.) * ((1.+z)/10.)**-1
        r_vir *= 0.784 * self.h70**(-2./3.) * cm_per_kpc

        return r_vir
