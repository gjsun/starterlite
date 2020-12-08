import numpy as np
import os

from ..analysis import Sensitivity
from ..physics import Cosmology
from ..physics.Constants import c, cm_per_mpc
from ..util.ParameterFile import ParameterFile



class FourierSpace(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

    def RealToFourier(self, f, x, y, z=None):
        """
        Return the fourier transform of a 2D or 3D real-space function
        ----------------------------------------
        :param f: target function defined in real space; {2d or 3d arr}
        :param x: 1st-dimension samples of real space; {1d arr}
        :param y: 2nd-dimension samples of real space; {1d arr}
        :param z: (optional) 3rd-dimension samples of real space; {1d arr}
        :return: F.T. of target function, and k space samples; {tuple}
        """
        if np.ndim(f) == 2:
            if not (self.EvenlySpaced(x) and self.EvenlySpaced(y)):
                raise ValueError('Must supply evenly spaced sample grids in real space!')
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            ft = np.fft.rfftn(f.T).T
            ft *= dx * dy   # Normalize (convert DFT to continuous FT)
            # Note that here (in 1D) we multiply ft by dx, and for the PS we further divide by Lx.
            # This is essentially dividing by Nx!!!
            kx = 2 * np.pi * np.fft.rfftfreq(x.size, d=dx)
            ky = 2 * np.pi * np.fft.fftfreq(y.size, d=dy)
            return ft, kx, ky
        elif np.ndim(f) == 3:
            if not (self.EvenlySpaced(x) and self.EvenlySpaced(y) and self.EvenlySpaced(z)):
                raise ValueError('Must supply evenly spaced sample grids in real space!')
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            dz = z[1] - z[0]
            ft = np.fft.rfftn(f.T).T
            ft *= dx * dy * dz   # Normalize (convert DFT to continuous FT)
            kx = 2 * np.pi * np.fft.rfftfreq(x.size, d=dx)
            ky = 2 * np.pi * np.fft.fftfreq(y.size, d=dy)
            kz = 2 * np.pi * np.fft.fftfreq(z.size, d=dz)
            return ft, kx, ky, kz
        else:
            raise NotImplementedError('Only 2d and 3d real spaces are supported!')


    def RealToAutoPS(self, f, x, y, z=None):
        """
        Return the power spectrum (that describes the fluctuations) of a real-space function
        ----------------------------------------
        :param f: target function defined in real space; {2d or 3d arr}
        :param x: 1st-dimension samples of real space; {1d arr}
        :param y: 2nd-dimension samples of real space; {1d arr}
        :param z: (optional) 3rd-dimension samples of real space; {1d arr}
        :return: power spectrum of target function, and k space samples; {tuple}
        """
        if np.ndim(f) == 2:
            ft, kx, ky = self.RealToFourier(f, x, y)
            vol = abs((x[-1] - x[0]) * (y[-1] - y[0]))
            # power spectrum = |F.T.(f)|^2 / V
            ps = abs(ft)**2 / vol
            return ps, kx, ky
        elif np.ndim(f) == 3:
            ft, kx, ky, kz = self.RealToFourier(f, x, y, z)
            vol = abs((x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0]))
            # power spectrum = |F.T.(f)|^2 / V
            ps = abs(ft)**2 / vol
            return ps, kx, ky, kz
        else:
            raise NotImplementedError('Only 2d and 3d real spaces are supported!')


    def RealToCrossPS(self, f1, f2, x, y, z=None):
        """
        Return the power spectrum (that describes the fluctuations) of a real-space function
        ----------------------------------------
        :param f1: 1st target function defined in real space; {2d or 3d arr}
        :param f2: 2nd target function defined in real space; {2d or 3d arr}
        :param x: 1st-dimension samples of real space; {1d arr}
        :param y: 2nd-dimension samples of real space; {1d arr}
        :param z: (optional) 3rd-dimension samples of real space; {1d arr}
        :return: power spectrum of target function, and k space samples; {tuple}
        """
        if (np.ndim(f1) == 2) and (np.ndim(f2) == 2):
            f1t, kx, ky = self.RealToFourier(f1, x, y)
            f2t = self.RealToFourier(f2, x, y)[0]
            vol = abs((x[-1] - x[0]) * (y[-1] - y[0]))
            # power spectrum = |F.T.(f)|^2 / V
            ps = abs(f1t*f2t) / vol
            return ps, kx, ky
        elif (np.ndim(f1) == 3) and (np.ndim(f2) == 3):
            f1t, kx, ky, kz = self.RealToFourier(f1, x, y, z)
            f2t = self.RealToFourier(f2, x, y, z)[0]
            vol = abs((x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0]))
            # power spectrum = |F.T.(f)|^2 / V
            ps = abs(f1t*f2t) / vol
            return ps, kx, ky, kz
        else:
            raise NotImplementedError('Only 2d and 3d real spaces are supported!')


    def CartesianToRadialBins(self, x, y, z=None, bins=None, log=False):
        """
        Convert Cartesian binning into radial binning
        ----------------------------------------
        :param x: 1st-dimension samples of Cartesian grid; {1d arr}
        :param y: 2nd-dimension samples of Cartesian grid; {1d arr}
        :param z: (optional) 3rd-dimension samples of Cartesian grid; {1d arr}
        :param bins: number of radial bins, if None, set according to dr = max(dx, dy, dz); {int or None}
        :return: radial bin edges; {1d arr}
        """
        if z is None:
            rmin = 0.
            rmax = max(np.amax(np.abs(x)), np.amax(np.abs(y)))
            if bins == None:
                dr = max(
                    [np.min(np.abs(q[q != 0])) for q in (x, y)])  # Smallest nonzero, absolute x,y,z coordinate value
                bins = int(np.ceil((rmax - rmin) / dr))
                if log:
                    _rmin = min(np.min(np.abs(x)), np.min(np.abs(y)))
                    if _rmin == 0.:
                        _rmin = min(np.partition(np.abs(x), 2)[1], np.partition(np.abs(y), 2)[1])
                        _rmin = _rmin / 1.01
                    rsphbins = np.logspace(np.log10(_rmin), np.log10(bins * dr), bins + 1)
                else:
                    rsphbins = np.linspace(0, bins * dr, bins + 1)
            else:
                if log:
                    _rmin = min(np.min(np.abs(x)), np.min(np.abs(y)))
                    if _rmin == 0.:
                        _rmin = min(np.partition(np.abs(x), 2)[1], np.partition(np.abs(y), 2)[1])
                        _rmin = _rmin / 1.01
                    rsphbins = np.logspace(np.log10(_rmin), np.log10(rmax), bins + 1)
                else:
                    rsphbins = np.linspace(rmin, rmax, bins + 1)
        else:
            rmin = 0.
            # rmax = max(np.amax(np.abs(x)), np.amax(np.abs(y)))
            rmax = np.sqrt(np.amax(np.abs(x))**2 + np.amax(np.abs(y))**2 + np.amax(np.abs(z))**2)
            if bins == None:
                dr = max([np.min(np.abs(q[q != 0])) for q in (x, y, z)])  # Smallest nonzero, absolute x,y,z coordinate value
                bins = int(np.ceil((rmax - rmin) / dr))
                _rmin = min(np.min(np.abs(x)), np.min(np.abs(y)), np.min(np.abs(z)))
                if _rmin == 0.:
                    _rmin = min(np.partition(np.abs(x), 2)[1], np.partition(np.abs(y), 2)[1], np.partition(np.abs(z), 2)[1])
                    _rmin /= 0.5
                if log:
                    rsphbins = np.logspace(np.log10(_rmin), np.log10(bins * dr), bins + 1)
                else:
                    rsphbins = np.linspace(_rmin, bins * dr, bins + 1)
            else:
                _rmin = min(np.min(np.abs(x)), np.min(np.abs(y)), np.min(np.abs(z)))
                if _rmin == 0.:
                    _rmin = min(np.partition(np.abs(x), 2)[1], np.partition(np.abs(y), 2)[1], np.partition(np.abs(z), 2)[1])
                    _rmin /= 0.5
                if log:
                    rsphbins = np.logspace(np.log10(_rmin), np.log10(rmax), bins + 1)
                else:
                    rsphbins = np.linspace(_rmin, rmax, bins + 1)

        return rsphbins


    def AverageAutoPS(self, f, x, y, z, bins=None, log=False, avg_type='sph'):
        """
        Return the averaged (as specified) auto power spectrum
        ----------------------------------------
        :param f: target function defined in real space; {2d or 3d arr}
        :param x: 1st-dimension samples of real space; {1d arr}
        :param y: 2nd-dimension samples of real space; {1d arr}
        :param z: (optional) 3rd-dimension samples of real space; {1d arr}
        :param avg_type: ; {str}
        :return:
        """
        if avg_type == 'sph':
            if np.ndim(f) == 2:
                pxyz, kx, ky = self.RealToAutoPS(f, x, y)
                ksphbins = self.CartesianToRadialBins(kx, ky, bins=bins, log=log)
                # 3d grid of r (distance from origin), note ORDER!
                rr = np.sqrt(sum(kk**2 for kk in np.meshgrid(kx, ky, indexing='ij')))
                # selection for r>0 (do not include origin)
                gt0 = np.where(rr > 0)
                # average within individual bins
                psph = np.histogram(rr[gt0], bins=ksphbins, weights=pxyz[gt0])[0] / np.histogram(rr[gt0], bins=ksphbins)[0]
                ksph = (ksphbins[0:-1] + ksphbins[1::])/2.
                return ksph, psph
            elif np.ndim(f) == 3:
                pxyz, kx, ky, kz = self.RealToAutoPS(f, x, y, z)
                ksphbins = self.CartesianToRadialBins(kx, ky, kz, bins=bins, log=log)
                # 3d grid of r (distance from origin), note ORDER!
                rr = np.sqrt(sum(kk**2 for kk in np.meshgrid(kx, ky, kz, indexing='ij')))
                # selection for r>0 (do not include origin)
                gt0 = np.where(rr > 0)
                # average within individual bins
                psph = np.histogram(rr[gt0], bins=ksphbins, weights=pxyz[gt0])[0] / np.histogram(rr[gt0], bins=ksphbins)[0]
                ksph = (ksphbins[0:-1] + ksphbins[1::])/2.
                return ksph, psph
            else:
                raise NotImplementedError('help!')
        else:
            raise NotImplementedError('help!')


    def AverageCrossPS(self, f1, f2, x, y, z, bins=None, log=False, avg_type='sph'):
        """
        Return the averaged (as specified) cross power spectrum
        ----------------------------------------
        :param f1: 1st target function defined in real space; {2d or 3d arr}
        :param f2: 2nd target function defined in real space; {2d or 3d arr}
        :param x: 1st-dimension samples of real space; {1d arr}
        :param y: 2nd-dimension samples of real space; {1d arr}
        :param z: (optional) 3rd-dimension samples of real space; {1d arr}
        :param avg_type: ; {str}
        :return:
        """
        if avg_type == 'sph':
            pxyz, kx, ky, kz = self.RealToCrossPS(f1, f2, x, y, z)
            ksphbins = self.CartesianToRadialBins(kx, ky, kz, bins=bins, log=log)
            rr = np.sqrt(sum(kk ** 2 for kk in np.meshgrid(kx, ky, kz,
                                                           indexing='ij')))  # 3d grid of r (distance from origin), note ORDER!
            gt0 = np.where(rr > 0)  # selection for r>0 (do not include origin)
            # average within individual bins
            psph = np.histogram(rr[gt0], bins=ksphbins, weights=pxyz[gt0])[0] / \
                   np.histogram(rr[gt0], bins=ksphbins)[0]
            ksph = (ksphbins[0:-1] + ksphbins[1::]) / 2.
            return ksph, psph
        else:
            raise NotImplementedError('help!')


    # ----------------   helpers   ---------------- #

    def EvenlySpaced(self, a):
        """
        Determine whether an array is evenly spaced
        ----------------------------------------
        :param a: input array; {1d arr}
        :return: evenly spaced or not; {boolean}
        """
        return np.allclose((a[1:] - a[:-1]), (a[1] - a[0]))


    def BinMidpoints(self, bin_edges):
        return 0.5*(bin_edges[:-1] + bin_edges[1:])