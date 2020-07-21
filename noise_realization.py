import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

data_path = '/home/wudl/project/ILC/fits_files/wmap/'
band_name = ['K', 'Ka', 'Q', 'V', 'W']
n_band = len(band_name)
nu = 0 #mean of noise

'''
sigma from wmap, one can change the value of sigma_0 for
specific project
'''
sigma_0 = np.array([1.429,1.466,2.188,3.131,6.544])   # unit: mk
N_obs = [hp.read_map(data_path+'wmap_band_smth_imap_r9_3yr_'+band+'_v2.fits', field=(1)) for band in band_name]
sigma = sigma_0.reshape(5,-1)/np.sqrt(N_obs)
n_realization = 100
noise = 0
len_pix=len(sigma[0])
lmax = 1024
nside = 512
fwhm = [0.88, 0.66, 0.51, 0.35, 0.22]  # degree
n_band = len(band_name)
alm_comb_n_smoothed = []

alm_comb_n = np.array([hp.map2alm(np.random.normal(nu,sigma[i],\
    (n_realization,len_pix)),lmax=lmax,pol=False) for i in np.arange(n_band)])
np.save('alm_comb_noise', alm_comb_n)

def beam_func(fwhm):
    '''
    return a gaussian beam with specific fwhm and lmax
    '''
    return hp.gauss_beam(fwhm=np.radians(fwhm), lmax=lmax)

#one can also try hp.almxfl function to compute alm times B_ell
def main():
    '''
    smoothing noise maps to the same angular resolution (1 degree)
    '''
    for i in np.arange(n_band):
        print(i)
        for j in np.arange(n_realization):
            alm = alm_comb_n[i, j]
            B_ell = beam_func(fwhm[i])
            B_ell_new = beam_func(1.)  # smooth to 1 degree
            for ell in np.arange(lmax+1):
                for m in np.arange(ell+1):
                    alm[hp.Alm.getidx(lmax=lmax, l=ell, m=m)] = alm[hp.Alm.getidx(
                        lmax=lmax, l=ell, m=m)]/B_ell[ell]*B_ell_new[ell]
            alm_comb_n_smoothed.append(alm)
    np.save('alm_comb_noise_smoothed_with_bl_own', np.array(
        alm_comb_n_smoothed).reshape(n_band, n_realization, -1))

main()
