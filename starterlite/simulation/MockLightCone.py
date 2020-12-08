import numpy as np
import os

from .FourierSpace import FourierSpace
from ..analysis import Sensitivity
from ..physics import Cosmology
from ..physics.Constants import c, cm_per_mpc, k_B, lsun, jansky_to_cgs
from ..util.ParameterFile import ParameterFile

class LIMGrid(object):

    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)


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


    def observed_grid(self, x_angs=None, y_angs=None, z_fs=None):
        """
        Return the spatial and spectral bin edges given the centers
        ----------------------------------------
        :param x_angs: spatial bin centers along x direction in arcmin, of shape Nx
        :param y_angs: spatial bin centers along y direction in arcmin, of shape Ny
        :param z_fs: spectral bin centers along z direction in [GHz], of shape Nz
        :return: bin edges along x, y, z directions; tuple of shape (Nx+1, Ny+1, Nz+1)
        """
        x_bin_edges = np.concatenate([[x_angs[0]-(x_angs[1]-x_angs[0])/2.],
                                      (x_angs[0:-1]+x_angs[1::])/2.,
                                      [x_angs[-1] + (x_angs[-1] - x_angs[-2])/2.]])
        y_bin_edges = np.concatenate([[y_angs[0]-(y_angs[1]-y_angs[0])/2.],
                                      (y_angs[0:-1]+y_angs[1::])/2.,
                                      [y_angs[-1] + (y_angs[-1] - y_angs[-2])/2.]])
        z_bin_edges = np.concatenate([[z_fs[0]-(z_fs[1]-z_fs[0])/2.],
                                      (z_fs[0:-1]+z_fs[1::])/2.,
                                      [z_fs[-1] + (z_fs[-1] - z_fs[-2])/2.]])

        return x_bin_edges, y_bin_edges, z_bin_edges


    def observed_to_comoving_grid_edges(self, obs_grid, nu0, avg_redshift):
        """
        Return the spatial and spectral bin edges in comoving distances
        ----------------------------------------
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :param avg_redshift: whether or not to calculate the comoving distances using a mean redshift; boolean
        :return: bin edges along x, y, z directions in terms of comoviding distances in [Mpc/h]; tuple
        """
        _xs, _ys, _zs = obs_grid
        _redshift_bin_edges = nu0 / _zs - 1.
        _mean_redshift = np.mean(_redshift_bin_edges)
        if avg_redshift:
            _z_comov_bin_edges = self.cosm.ComovingRadialDistance(z0=0., z=_redshift_bin_edges) / cm_per_mpc * self.cosm.h70
            # Make the z binning uniform, as required by the F.T. pipeline
            _z_comov_bin_edges = np.linspace(min(_z_comov_bin_edges), max(_z_comov_bin_edges), np.size(_z_comov_bin_edges))
            _z_comov_mean = self.cosm.ComovingRadialDistance(z0=0., z=_mean_redshift) / cm_per_mpc * self.cosm.h70
            _x_comov_bin_edges = _xs / 60. * (np.pi / 180.) * _z_comov_mean
            _y_comov_bin_edges = _ys / 60. * (np.pi / 180.) * _z_comov_mean
            return _x_comov_bin_edges, _y_comov_bin_edges, _z_comov_bin_edges
        else:
            _z_comov_bin_edges = self.cosm.ComovingRadialDistance(z0=0., z=_redshift_bin_edges) / cm_per_mpc * self.cosm.h70
            _x_comov_bin_edges = _xs[:,np.newaxis] / 60. * (np.pi / 180.) * _z_comov_bin_edges[np.newaxis,:]# * self.cosm.h70
            _y_comov_bin_edges = _ys[:,np.newaxis] / 60. * (np.pi / 180.) * _z_comov_bin_edges[np.newaxis,:]# * self.cosm.h70
            return _x_comov_bin_edges, _y_comov_bin_edges, _z_comov_bin_edges


    def observed_to_comoving_grid_ctrs(self, obs_grid, nu0, avg_redshift=True):
        """
        Return the spatial and spectral bin centers in comoving distances
        ----------------------------------------
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :param avg_redshift: whether or not to calculate the comoving distances using a mean redshift; boolean
        :return: bin centers along x, y, z directions in terms of comoving distances in [Mpc/h]; tuple
        """
        if avg_redshift:
            _x, _y, _z = self.observed_to_comoving_grid_edges(obs_grid, nu0, avg_redshift)
            _x_ctr = (_x[0:-1] + _x[1::]) / 2.
            _y_ctr = (_y[0:-1] + _y[1::]) / 2.
            _z_ctr = (_z[0:-1] + _z[1::]) / 2.
            return _x_ctr, _y_ctr, _z_ctr
        else:
            raise NotImplementedError('oops!')


    def observed_to_redshift_grid(self, obs_grid, nu0):
        """
        Return a grid of central redshifts for a given observed grid
        ----------------------------------------
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :return: a grid of redshifts of voxels; 3d array of shape (Nx, Ny, Nz)
        """
        _xs, _ys, _zs = obs_grid
        _redshift_bin_edges = nu0/_zs - 1.
        _redshift_bin_ctrs = (_redshift_bin_edges[0:-1] + _redshift_bin_edges[1::])/2.
        _redshift_ctr_grid = np.tile(_redshift_bin_ctrs, (_xs.size-1, _ys.size-1, 1))

        return _redshift_ctr_grid


    def observed_to_Vcomoving_grid(self, obs_grid, nu0):
        """
        Return a grid of central comoving volume of voxels for a given observed grid
        ----------------------------------------
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :return: a grid of comoving volume of voxels in (Mpc/h)^3; 3d array of shape (Nx, Ny, Nz)
        """
        _xs, _ys, _zs = obs_grid
        _redshift_bin_edges = nu0/_zs - 1.   # decreasing from 8.5 to 5.3
        _z_comov_bin_edges = self.cosm.ComovingRadialDistance(z0=0., z=_redshift_bin_edges) / cm_per_mpc
        _z_comov_bin_ctrs = ((_z_comov_bin_edges[0:-1] + _z_comov_bin_edges[1::])/2.)[np.newaxis, np.newaxis, :]
        _dz_comov_ctr = abs(_z_comov_bin_edges[1::] - _z_comov_bin_edges[0:-1])[np.newaxis, np.newaxis, :]
        _dx_comov_ctr = abs(_xs[1::] - _xs[0:-1])[:, np.newaxis, np.newaxis] / 60. * (np.pi/180.) * _z_comov_bin_ctrs
        _dy_comov_ctr = abs(_ys[1::] - _ys[0:-1])[np.newaxis, :, np.newaxis] / 60. * (np.pi/180.) * _z_comov_bin_ctrs

        return _dx_comov_ctr * _dy_comov_ctr * _dz_comov_ctr * self.cosm.h70**3






class MockLightCone(LIMGrid, FourierSpace):

    def __init__(self, **kwargs):
        LIMGrid.__init__(self, **kwargs)


    def get_halo_voxid(self, x_halo, y_halo, z_halo, obs_grid, nu0):
        """
        For given halo (x,y,z) positions, return the IDs of the voxels they belong to
        ----------------------------------------
        :param x_halo: angular x position of N halos in arcmin
        :param y_halo: angular y position of N halos in arcmin
        :param z_halo: redshift of N halos
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :return: flat indices of N halos; 1d array
        """
        _xs, _ys, _zs = obs_grid
        nxbins, nybins, nzbins = [i.size - 1 for i in (_xs, _ys, _zs)]  # Number of voxels along each direction
        nbins = nxbins * nybins * nzbins                                # Total number of voxels
        _redshift_bin_edges = nu0 / _zs - 1.
        id_x = np.digitize(x_halo, _xs) - 1
        id_y = np.digitize(y_halo, _ys) - 1
        id_z = np.digitize(z_halo, _redshift_bin_edges) - 1
        # flat index is calculated as id_x * nybins * nzbins + id_y * nzbins + id_z
        voxid = np.ravel_multi_index((id_x, id_y, id_z), (nxbins, nybins, nzbins), mode='clip')
        invalid = (id_x < 0) | (id_x > nxbins - 1) | (id_y < 0) | (id_y > nybins - 1) | (id_z < 0) | (id_z > nzbins - 1)
        if any(invalid):
            voxid[np.where(invalid)] = nbins
        else:
            pass
        return voxid


    def Ngalvox_grid(self, x_halo, y_halo, z_halo, mstar_halo, obs_grid, nu0, mstar_cut):
        """
        Return a grid of galaxy counts within voxels for given halo (x,y,z) positions
        ----------------------------------------
        :param x_halo: angular x position of N halos in arcmin
        :param y_halo: angular y position of N halos in arcmin
        :param z_halo: redshift of N halos
        :param mstar_halo: stellar mass in [Msun] of N halos, used as a cutoff for galaxy number counts
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :param mstar_cut: stellar mass cutoff
        :return: a grid of voxel galaxy number counts; 3d array of shape (Nx, Ny, Nz)
        """
        _xs, _ys, _zs = obs_grid
        nxbins, nybins, nzbins = [i.size - 1 for i in (_xs, _ys, _zs)]  # Number of voxels along each direction
        nbins = nxbins * nybins * nzbins                                # Total number of voxels
        _redshift_bin_edges = nu0 / _zs - 1.
        id_x = np.digitize(x_halo, _xs) - 1
        id_y = np.digitize(y_halo, _ys) - 1
        id_z = np.digitize(z_halo, _redshift_bin_edges) - 1
        _weights = np.zeros_like(mstar_halo)
        _weights[np.where(mstar_halo >= mstar_cut)] = 1.
        # flat index is calculated as id_x * nybins * nzbins + id_y * nzbins + id_z
        voxid = np.ravel_multi_index((id_x, id_y, id_z), (nxbins, nybins, nzbins), mode='clip')   # a 1d array corresponding to the N halos
        invalid = (id_x < 0) | (id_x > nxbins - 1) | (id_y < 0) | (id_y > nybins - 1) | (id_z < 0) | (id_z > nzbins - 1)
        if any(invalid):
            voxid[np.where(invalid)] = nbins
        else:
            pass
        ngalvox = np.bincount(voxid, weights=_weights, minlength=nbins)[:nbins].reshape(nxbins, nybins, nzbins)

        return ngalvox


    def deltaNDgal_grid(self, x_halo, y_halo, z_halo, mstar_halo, obs_grid, nu0, mstar_cut):
        """
        Return a grid of overdensity of voxel galaxy counts for given halo (x,y,z) positions
        ----------------------------------------
        :param x_halo: angular x position of N halos in arcmin
        :param y_halo: angular y position of N halos in arcmin
        :param z_halo: redshift of N halos
        :param mstar_halo: stellar mass in [Msun] of N halos, used as a cutoff for galaxy number counts
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :param mstar_cut: stellar mass cutoff
        :return:
        """
        _Ngal_grid = self.Ngalvox_grid(x_halo, y_halo, z_halo, mstar_halo, obs_grid, nu0, mstar_cut)
        #_Ngal_grid_true = self.Ngalvox_grid(x_halo, y_halo, z_halo, mstar_halo, obs_grid, nu0, mstar_cut=1.0e6)
        _NDgal_grid = _Ngal_grid / self.observed_to_Vcomoving_grid(obs_grid, nu0)
        #print '\n--------------'
        #print 'Number of galaxies:', np.log10(mstar_cut), np.sum(_Ngal_grid.flatten())
        #print '--------------\n'
        #_deltaNDgal_mean = np.sum(_Ngal_grid.flatten()) / np.sum(self.observed_to_Vcomoving_grid(obs_grid, nu0).flatten())
        _NDgal_mean = np.mean(_NDgal_grid)
        _deltaNDgal_grid = _NDgal_grid / _NDgal_mean - 1.

        return _deltaNDgal_grid


    def Lvox_grid(self, x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0):
        """
        Return a grid of voxel luminosities for given halo (x,y,z) positions
        ----------------------------------------
        :param x_halo: angular x position of N halos in arcmin
        :param y_halo: angular y position of N halos in arcmin
        :param z_halo: redshift of N halos
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :return: a grid of voxel luminosities; 3d array of shape (Nx, Ny, Nz)
        """
        _xs, _ys, _zs = obs_grid
        nxbins, nybins, nzbins = [i.size - 1 for i in (_xs, _ys, _zs)]  # Number of voxels along each direction
        nbins = nxbins * nybins * nzbins                                # Total number of voxels
        _redshift_bin_edges = nu0 / _zs - 1.
        id_x = np.digitize(x_halo, _xs) - 1
        id_y = np.digitize(y_halo, _ys) - 1
        id_z = np.digitize(z_halo, _redshift_bin_edges) - 1
        # flat index is calculated as id_x * nybins * nzbins + id_y * nzbins + id_z
        voxid = np.ravel_multi_index((id_x, id_y, id_z), (nxbins, nybins, nzbins), mode='clip')   # a 1d array corresponding to the N halos
        invalid = (id_x < 0) | (id_x > nxbins - 1) | (id_y < 0) | (id_y > nybins - 1) | (id_z < 0) | (id_z > nzbins - 1)
        if any(invalid):
            voxid[np.where(invalid)] = nbins
        else:
            pass
        lvox = np.bincount(voxid, weights=lum_halo, minlength=nbins)[:nbins].reshape(nxbins, nybins, nzbins)

        return lvox


    def Ivox_grid(self, x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0):
        """
        Return a grid of voxel intensities for given halo (x,y,z) positions
        ----------------------------------------
        :param x_halo: angular x position of N halos in arcmin
        :param y_halo: angular y position of N halos in arcmin
        :param z_halo: redshift of N halos
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :return: a grid of voxel intensities in [Jy/sr]; 3d array of shape (Nx, Ny, Nz)
        """
        Vvox = self.observed_to_Vcomoving_grid(obs_grid, nu0) * (cm_per_mpc / self.cosm.h70)**3
        zvox = self.observed_to_redshift_grid(obs_grid, nu0)
        Ivox = self.Lvox_grid(x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0) * lsun / Vvox / self.cosm.HubbleParameter(zvox)
        Ivox *= c / 4. / np.pi / (nu0*1.0e9)
        Ivox /= jansky_to_cgs

        return Ivox


    def Tvox_grid(self, x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0):
        """
        Return a grid of voxel brightness temperatures for given halo (x,y,z) positions
        ----------------------------------------
        :param x_halo: angular x position of N halos in [arcmin]
        :param y_halo: angular y position of N halos in [arcmin]
        :param z_halo: redshift of N halos
        :param lum_halo: line luminosity of N halos in [Lsun]
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :return: a grid of voxel brightness temperatures in [muK]; 3d array of shape (Nx, Ny, Nz)
        """
        Vvox = self.observed_to_Vcomoving_grid(obs_grid, nu0) * (cm_per_mpc / self.cosm.h70)**3
        zvox = self.observed_to_redshift_grid(obs_grid, nu0)
        Tvox = self.Lvox_grid(x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0) * lsun * (1.+zvox)**2 / Vvox / self.cosm.HubbleParameter(zvox)
        Tvox *= c**3/8./np.pi/k_B/(nu0*1.0e9)**3
        Tvox *= 1.0e6   # convert K to muK

        return Tvox


    def ThreeDimensionalAPS(self, x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0):
        """
        Return the 3D power spectrum as a function of (kx, ky, kz)
        ----------------------------------------
        :param x_halo: angular x position of N halos in [arcmin]
        :param y_halo: angular y position of N halos in [arcmin]
        :param z_halo: redshift of N halos
        :param lum_halo: line luminosity of N halos in [Lsun]
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :return: power spectrum of target function, and k space samples; tuple
        """
        _x_real, _y_real, _z_real = self.observed_to_comoving_grid_ctrs(obs_grid, nu0)
        return self.RealToAPS(f=self.Ivox_grid(x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0),
                              x=_x_real, y=_y_real, z=_z_real)


    def ThreeDimensionalCPS(self, x_halo, y_halo, z_halo, lum_halo, mstar_halo, obs_grid, nu0, **kwargs):
        """
        Return the 3D power spectrum as a function of (kx, ky, kz)
        ----------------------------------------
        :param x_halo: angular x position of N halos in [arcmin]
        :param y_halo: angular y position of N halos in [arcmin]
        :param z_halo: redshift of N halos
        :param lum_halo: line luminosity of N halos in [Lsun]
        :param mstar_halo: stellar mass in [Msun] of N halos, used as a cutoff for galaxy number counts
        :param obs_grid: spatial and spectral bin edges given the centers; tuple of 1d arrays
        :param nu0: rest-frame frequency of emission line in [GHz]
        :return: power spectrum of target function, and k space samples; tuple
        """
        _x_real, _y_real, _z_real = self.observed_to_comoving_grid_ctrs(obs_grid, nu0)
        return self.RealToCPS(f1=self.Ivox_grid(x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0),
                              f2=self.deltaNDgal_grid(x_halo, y_halo, z_halo, mstar_halo, obs_grid, nu0, **kwargs),
                              x=_x_real, y=_y_real, z=_z_real)


    def SphericallyAveragedAPS(self, x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0, bins=None, log=False):
        _x_real, _y_real, _z_real = self.observed_to_comoving_grid_ctrs(obs_grid, nu0)
        return self.AverageAPS(f=self.Ivox_grid(x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0),
                               x=_x_real, y=_y_real, z=_z_real, bins=bins, log=log)


    def SphericallyAveragedGalAPS(self, x_halo, y_halo, z_halo, mstar_halo, obs_grid, nu0, bins=None, log=False, **kwargs):
        _x_real, _y_real, _z_real = self.observed_to_comoving_grid_ctrs(obs_grid, nu0)
        return self.AverageAPS(f=self.deltaNDgal_grid(x_halo, y_halo, z_halo, mstar_halo, obs_grid, nu0, **kwargs),
                               x=_x_real, y=_y_real, z=_z_real, bins=bins, log=log)


    def SphericallyAveragedCPS(self, x_halo, y_halo, z_halo, lum_halo, mstar_halo, obs_grid, nu0, bins=None, log=False, **kwargs):
        _x_real, _y_real, _z_real = self.observed_to_comoving_grid_ctrs(obs_grid, nu0)
        return self.AverageCPS(f1=self.Ivox_grid(x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0),
                               f2=self.deltaNDgal_grid(x_halo, y_halo, z_halo, mstar_halo, obs_grid, nu0, **kwargs),
                               x=_x_real, y=_y_real, z=_z_real, bins=bins, log=log)


    def CoeffCC(self, x_halo, y_halo, z_halo, lum_halo, mstar_halo, obs_grid, nu0, bins=None, log=False, ctype='LxG', **kwargs):
        if ctype=='LxG':
            _k, _r = self.SphericallyAveragedCPS(x_halo, y_halo, z_halo, lum_halo, mstar_halo, obs_grid, nu0, bins, log, **kwargs)
            _r /= self.SphericallyAveragedAPS(x_halo, y_halo, z_halo, lum_halo, obs_grid, nu0, bins, log)[1]**0.5
            _r /= self.SphericallyAveragedGalAPS(x_halo, y_halo, z_halo, mstar_halo, obs_grid, nu0, bins, log, **kwargs)[1]**0.5
            return _k, _r
        else:
            raise NotImplementedError('oops!')

