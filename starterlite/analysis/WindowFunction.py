import numpy as np
import os
import time
from scipy.special import spherical_jn
from scipy.interpolate import interp1d

from starter.analysis.Sensitivity import Sensitivity

"""

------------
Instructions
------------
The WindowFunction module serves to provide essential routines to compute the window function for user-specified survey
geometries, in the form of interpolated, callable functions or transfer matrices (must supply a binning scheme)

* Acknowledgements
Thanks Bade Uzgil for pioneering the development of this part.

"""

class WindowFunction(Sensitivity):
    def __init__(self, **kwargs):
        Sensitivity.__init__(self, **kwargs)

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
    def n_beam(self):
        return self.survey_goemetry[0] * self.survey_goemetry[1]


    @property
    def n_channel(self):
        return self.survey_goemetry[-1]


    def phi(self, kx, ky, kz, k_i, L_x, L_y, L_z):
        """
        F.T. of real-space, **top-hat** selection functions
        ----------------------------------------
        :param kx: x-coord of wavenumber of sky mode
        :param ky: x-coord of wavenumber of sky mode
        :param kz: x-coord of wavenumber of sky mode
        :param k_i: wavenumber of instrument mode
        :param L_x: length of survey volume along 1st dimension; {scalar}
        :param L_y: length of survey volume along 2nd dimension; {scalar}
        :param L_z: length of survey volume along 3rd (LOS) dimension; {scalar}
        :return: F.T. of real-space selection function
        """

        j0_x = spherical_jn(0, (k_i[0] - kx) * L_x / 2.)
        j0_y = spherical_jn(0, -ky * L_y / 2.)
        j0_z = spherical_jn(0, (k_i[1] - kz) * L_z / 2.)

        return j0_x * j0_y * j0_z


    def AnalyticalWF(self, k, k_i, L_x, L_y, L_z, n_angsmpl=1000):
        """
        Compute (fourier-space) window function for (real-space) top-hat selection function analytically
        ----------------------------------------
        :param k: wavenumber of sky mode; {scalar}
        :param k_i: wavenumber of instrument mode; {list}
        :param L_x: length of survey volume along 1st dimension; {scalar}
        :param L_y: length of survey volume along 2nd dimension; {scalar}
        :param L_z: length of survey volume along 3rd (LOS) dimension; {scalar}
        :return: WF(k) for a given instrument mode k_i; {scalar}
        """

        _theta_list = np.linspace(0., np.pi, n_angsmpl)
        _phi_list = np.linspace(0., 2.*np.pi, n_angsmpl)

        def integrand0(Phi, Theta):
            if np.isscalar(Phi) or np.isscalar(Theta):
                j0_x = spherical_jn(0, (k_i[0] - k * np.sin(Theta) * np.cos(Phi)) * L_x / 2.)
                j0_y = spherical_jn(0, -k * np.sin(Theta) * np.sin(Phi) * L_y / 2.)
                j0_z = spherical_jn(0, (k_i[1] - k * np.cos(Theta)) * L_z / 2.)
                temp = (j0_x * j0_y * j0_z) ** 2
                temp *= np.sin(Theta)
            else:
                j0_x = spherical_jn(0,(k_i[0] - k * np.sin(Theta[np.newaxis, :]) * np.cos(Phi[:, np.newaxis])) * L_x / 2.)
                j0_y = spherical_jn(0, -k * np.sin(Theta[np.newaxis, :]) * np.sin(Phi[:, np.newaxis]) * L_y / 2.)
                j0_z = spherical_jn(0, (k_i[1] - k * np.cos(Theta[np.newaxis, :])) * L_z / 2.)
                temp = (j0_x * j0_y * j0_z) ** 2
                temp *= np.sin(Theta[np.newaxis, :])
            return temp

        def integrand1(Phi):
            temp = integrand0(Phi, _theta_list)
            temp = np.trapz(temp, _theta_list)
            return temp

        ans = np.array([integrand1(i) for i in _phi_list])
        ans = np.trapz(ans, _phi_list)
        ans /= 4. * np.pi

        return ans


    def RunAnalyticalWF(self, L_x, L_y, L_z):

        # set up the k-space array for binning power spectra
        nbins = 96

        inst_mode_min = min((2. * np.pi / L_x), (2. * np.pi / L_z))
        inst_mode_max = np.sqrt((self.n_beam * np.pi / L_x)**2 + (self.n_channel * np.pi / L_z)**2)

        dlogk = (np.log10(inst_mode_max) - np.log10(inst_mode_min)) / nbins

        logkrange = np.zeros(nbins, float)

        for q in range(0, nbins):
            logkrange[q] = np.log10(inst_mode_min) + q * dlogk

        krange = 10. ** logkrange

        krange_extra1 = 10. ** (logkrange[len(krange) - 1] + dlogk)
        krange_extra2 = 10. ** (logkrange[len(krange) - 1] + (2. * dlogk))
        krange_extra3 = 10. ** (logkrange[len(krange) - 1] + (3. * dlogk))
        krange_extra4 = 10. ** (logkrange[len(krange) - 1] + (4. * dlogk))
        krange_extra5 = 10. ** (logkrange[len(krange) - 1] + (5. * dlogk))

        krange = np.append(krange, [krange_extra1, krange_extra2, krange_extra3, krange_extra4, krange_extra5])

        k_anal = (krange[0:-1] + krange[1::]) / 2.

        nshell = len(krange) - 1  # 100

        # ----- compute 2D K grid for TIME ----- #
        kx_2d = 2 * np.pi * np.fft.fftfreq(self.n_beam, L_x / self.n_beam)
        kx_2d = np.roll(kx_2d, -self.n_beam // 2 - self.n_beam % 2, axis=0)

        kz_2d = 2 * np.pi * np.fft.fftfreq(self.n_channel, L_z / self.n_channel)
        kz_2d = np.roll(kz_2d, -self.n_channel // 2 - self.n_channel % 2, axis=0)

        wf_of_k3d = np.zeros((np.size(kx_2d), np.size(kz_2d), nshell, 2))

        for iii in range(np.size(kx_2d)):
            for jjj in range(np.size(kz_2d)):
                k_inst = [kx_2d[iii], kz_2d[jjj]]
                k_i_x_mag1 = k_inst[0]
                k_i_z_mag1 = k_inst[1]
                pathname = os.getenv('STARTERLITE') + '/output/wf/HF/raw/'
                filename = 'analytical_wf_kx%.5f_kz%.5f.npz' % (k_i_x_mag1, k_i_z_mag1)
                if os.path.exists(pathname + filename):
                    print('K mode [%.5f, %.5f] already calculated, pass!' % (k_i_x_mag1, k_i_z_mag1))
                    pass
                else:
                    print('Calculating wf of k_inst:', k_inst)
                    wf_anal = np.zeros_like(k_anal)

                    for i in range(nshell):
                        wf_anal[i] = self.AnalyticalWF(k=k_anal[i], k_i=k_inst, L_x=L_x, L_y=L_y, L_z=L_z)

                    wf_of_k3d[iii, jjj, :, 0] = k_anal
                    wf_of_k3d[iii, jjj, :, 1] = wf_anal

                    ana_wf_data_ij = {
                        'kx_2d': kx_2d,
                        'kz_2d': kz_2d,
                        'wf_of_k3d': wf_of_k3d[iii, jjj, :, :]
                    }

                    np.savez(pathname + filename, **ana_wf_data_ij)


    def CalculateTMatrix(self, L_x, L_y, L_z):

        # ----- compute 2D K grid for TIME ----- #
        kx_2d = 2 * np.pi * np.fft.fftfreq(self.n_beam, L_x / self.n_beam)
        kx_2d = np.roll(kx_2d, -self.n_beam // 2 - self.n_beam % 2, axis=0)

        kz_2d = 2 * np.pi * np.fft.fftfreq(self.n_channel, L_z / self.n_channel)
        kz_2d = np.roll(kz_2d, -self.n_channel // 2 - self.n_channel % 2, axis=0)

        nx_samples = np.size(kx_2d)
        nz_samples = np.size(kz_2d)
        n_k2d = nx_samples * nz_samples

        n_shell = 20
        T_matrix = np.zeros((n_k2d, n_shell))
        k3d_matrix = np.zeros((n_k2d, n_shell))
        K2D_matrix = np.zeros((n_k2d, 2))

        pathname = os.getenv('STARTERLITE') + '/output/wf/HF/raw/'

        for ii in range(nx_samples):
            for jj in range(nz_samples):
                filename = 'analytical_wf_kx%.5f_kz%.5f.npz'%(kx_2d[ii], kz_2d[jj])
                ana_wf_data = np.load(pathname+filename)
                k_num = ana_wf_data['wf_of_k3d'][:, 0]
                wf_num = ana_wf_data['wf_of_k3d'][:, 1]

                ln_k_num = np.log(k_num)
                dln_k_num = abs(ln_k_num[1] - ln_k_num[0])

                #integrand = L_x * L_z * k_num**3 / 2. / np.pi**2 * wf_num
                #ans = np.trapz(integrand, ln_k_num)

                # Use trapezoidal-rule sum!!!
                norm = L_x * L_y * L_z * (wf_num * (k_num**3 + np.concatenate([[0.], k_num[1:-1]**3, [0.]])) / 2. / 2. / np.pi**2 * dln_k_num).reshape(1, np.size(k_num))
                norm = np.sum(norm)
                xxx = L_x * L_z * (wf_num * (k_num**3 + np.concatenate([[0.], k_num[1:-1]**3, [0.]])) / 2. / 2. / np.pi**2 * dln_k_num).reshape(1, np.size(k_num))
                xxx /= norm
                #yyy = np.ones_like(k_num).reshape(np.size(k_num), 1)

                T_matrix[ii * nz_samples + jj, :] = xxx
                k3d_matrix[ii * nz_samples + jj, :] = k_num
                K2D_matrix[ii * nz_samples + jj, 0] = kx_2d[ii]
                K2D_matrix[ii * nz_samples + jj, 1] = kz_2d[jj]

        T_matrix_data = {'T_matrix': T_matrix,
                         'k3d_bins': k_num,
                         'K2D_matrix': K2D_matrix,}

        np.savez(os.getenv('STARTERLITE') + '/output/wf/HF/TMatrix/T_matrix_HF_x%dz%d.npz' % (self.n_beam, self.n_channel), **T_matrix_data)