import numpy as np
import os
import time

from scipy.special import spherical_jn

######################################################################
#   Analytical Method
######################################################################

nbeams = 180   # number of beams in the transverse direction
nchannels = 14 # number of freq channels (science channels only)

L_x = lslab_x = 0.56860 * nbeams   # Mpc h^-1
L_y = lslab_y = 0.56860            # Mpc h^-1
L_z = lslab_z = 270.370            # Mpc h^-1

N_angsmpl = 1000
theta_list = np.linspace(0, np.pi, N_angsmpl)
phi_list = np.linspace(0, 2.*np.pi, N_angsmpl)


def phi(kx, ky, kz, k_i):
    j0_x = spherical_jn(0,(k_i[0]-kx)*L_x/2.)
    j0_y = spherical_jn(0,-ky*L_y/2.)
    j0_z = spherical_jn(0,(k_i[1]-kz)*L_z/2.)
    return j0_x * j0_y * j0_z


def WF(k, k_i):

    def integrand0(Phi, Theta):
        if np.isscalar(Phi) or np.isscalar(Theta):
                j0_x = spherical_jn(0,(k_i[0]-k*np.sin(Theta)*np.cos(Phi))*L_x/2.)
                j0_y = spherical_jn(0,-k*np.sin(Theta)*np.sin(Phi)*L_y/2.)
                j0_z = spherical_jn(0,(k_i[1]-k*np.cos(Theta))*L_z/2.)
                temp = (j0_x * j0_y * j0_z)**2
                temp *= np.sin(Theta)
        else:
                j0_x = spherical_jn(0,(k_i[0]-k*np.sin(Theta[np.newaxis,:])*np.cos(Phi[:,np.newaxis]))*L_x/2.)
                j0_y = spherical_jn(0,-k*np.sin(Theta[np.newaxis,:])*np.sin(Phi[:,np.newaxis])*L_y/2.)
                j0_z = spherical_jn(0,(k_i[1]-k*np.cos(Theta[np.newaxis,:]))*L_z/2.)
                temp = (j0_x * j0_y * j0_z)**2
                temp *= np.sin(Theta[np.newaxis,:])
        return temp

    def integrand1(Phi):
    	temp = integrand0(Phi, theta_list)
    	temp = np.trapz(temp, theta_list)
    	return temp

    ans = integrand1(phi_list)
    ans = np.trapz(ans, phi_list)
    ans /= 4.*np.pi

    return ans



print('\n======== Calculating WF analytically ========\n')

# set up the k-space array for binning power spectra
nbins = 96

inst_mode_min = min((2.*np.pi/lslab_x), (2.*np.pi/lslab_z))
inst_mode_max = np.sqrt((nbeams*np.pi / lslab_x)**2. + (nchannels*np.pi / lslab_z)**2.)

dlogk = (np.log10(inst_mode_max) - np.log10(inst_mode_min)) / nbins

logkrange = np.zeros(nbins, float)

for q in range(0, nbins):
    logkrange[q] = np.log10(inst_mode_min) + q*dlogk

krange = 10.**logkrange

krange_extra1 = 10.**(logkrange[len(krange)-1]+dlogk)
krange_extra2 = 10.**(logkrange[len(krange)-1]+(2.*dlogk))
krange_extra3 = 10.**(logkrange[len(krange)-1]+(3.*dlogk))
krange_extra4 = 10.**(logkrange[len(krange)-1]+(4.*dlogk))
krange_extra5 = 10.**(logkrange[len(krange)-1]+(5.*dlogk))

krange = np.append(krange, [krange_extra1, krange_extra2, krange_extra3, krange_extra4, krange_extra5])

k_anal = (krange[0:-1] + krange[1::])/2.

nshell = len(krange)-1   # 100



# ----- compute 2D K grid for TIME ----- #
kx_2d = 2 * np.pi * np.fft.fftfreq(int(nbeams), lslab_x/nbeams)
kx_2d = np.roll(kx_2d, -int(nbeams) // 2 - int(nbeams) % 2, axis=0)

kz_2d = 2 * np.pi * np.fft.fftfreq(int(nchannels), lslab_z/nchannels)
kz_2d = np.roll(kz_2d, -int(nchannels) // 2 - int(nchannels) % 2, axis=0)


wf_of_k3d = np.zeros((np.size(kx_2d), np.size(kz_2d), nshell, 2))

for iii in range(np.size(kx_2d)):
	for jjj in range(np.size(kz_2d)):
		k_inst = [kx_2d[iii], kz_2d[jjj]]
		k_i_x_mag1 = k_inst[0]
		k_i_z_mag1 = k_inst[1]
		pathname = os.getenv('STARTERLITE') + '/output/wf/HF/raw/'
		filename = 'analytical_wf_kx%.5f_kz%.5f.npz'%(k_i_x_mag1,k_i_z_mag1)
		if os.path.exists(pathname+filename):
			print('K mode [%.5f, %.5f] already calculated, pass!'%(k_i_x_mag1,k_i_z_mag1))
			pass
		else:
			print('Calculating wf of k_inst:', k_inst)
				
			wf_anal = np.zeros_like(k_anal)
		
			for i in range(nshell):
				wf_anal[i] = WF(k=k_anal[i], k_i=k_inst)
				#print 'k =', k_anal[i], '%d/%d'%(i+1, nshell)
				
			wf_of_k3d[iii,jjj,:,0] = k_anal
			wf_of_k3d[iii,jjj,:,1] = wf_anal
			
			ana_wf_data_ij = {
			'kx_2d': kx_2d,
			'kz_2d': kz_2d,
			'wf_of_k3d': wf_of_k3d[iii,jjj,:,:]
			}

			np.savez(pathname+filename, **ana_wf_data_ij)
