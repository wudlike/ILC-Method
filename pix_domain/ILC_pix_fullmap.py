import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

bandname = ['K','Ka','Q','V','W']
lmax = 1024
maps = []
alms = []
mk2uk = 1000
map_path = r'/home/wudl/project/ILC/fits_files/wmap/'

def beam_func(fwhm):
    return hp.gauss_beam(fwhm=np.radians(fwhm), lmax=lmax)

for i in np.arange(5):
    '''
    load 5 maps, note that all the maps are smoothed to 1 degree.
    '''
    T_map = hp.read_map(map_path+'wmap_band_smth_imap_r9_3yr_{0}_v2.fits'.format(bandname[i]))
    maps.append(T_map)
    alms.append(hp.map2alm(T_map))
    hp.mollview(T_map,sub=(2,3,i+1),unit=r'mK$_\mathrm{CMB}$',norm='hist',min=-0.4,max=0.4,title= bandname[i] + "-band")

map_cov = np.cov(maps)
print('map_cov:>>> ',map_cov)
weight = []
map_cov_inv = np.linalg.inv(map_cov)
for i in range(5):
    weight.append(np.sum(map_cov_inv[i])/np.sum(map_cov_inv))

#print weight
print(np.array(maps).shape) # shape of the data array
print(weight,np.sum(weight))

# get clean map
map_clean = np.sum(np.array(weight).reshape(5,-1)*maps,axis=0)
#np.save('wmap_clean_map',map_clean)

# add a temperature mask
mask_ = hp.read_map(map_path+'wmap_temperature_analysis_mask_r9_7yr_v4.fits')
map_clean_masked = hp.ma(map_clean)
map_clean_masked.mask = np.logical_not(mask_)
hp.mollview(map_clean_masked.filled(),unit=r'mK$_\mathrm{CMB}$',norm='hist',min=-0.4,max=0.4,title='WMAP masked clean T_map')

'''
cleaned map from wmap team via ILC method, note that the map have been smoothed to 1 degree
and the inputted 5 original maps are divided into 12 regions, and the boundardies are smoothed
with 1.5 degree.
please see https://lambda.gsfc.nasa.gov/product/map/dr5/ilc_map_info.cfm
'''
ILC_wmap = hp.read_map(
    '/home/wudl/project/ILC/fits_files/wmap/wmap_ilc_3yr_v2.fits')
Cl_wmap = hp.anafast(ILC_wmap, lmax=lmax)*mk2uk**2   # mk ---> uk
ell = np.arange(Cl_wmap.size)
Dl_wmap = ell*(ell+1)/2/np.pi*Cl_wmap  
Dl_wmap_no_beam = Dl_wmap/beam_func(1.)**2  # remove the effects of beam

#show the cleaned_map by ILC method 
plt.figure()
hp.mollview(ILC_wmap, sub=(1,2,1),unit=r'mK$_\mathrm{CMB}$', norm='hist', min=-
            0.4, max=0.4, title='Cleaned T_map from wmap team (ILC-12 regions)')
hp.mollview(map_clean, sub=(1, 2, 2),unit=r'mK$_\mathrm{CMB}$', norm='hist',
            min=-0.4, max=0.4, title='cleaned T_map with ILC')

# cleaned Cl
cleaned_Cl = hp.anafast(map1=map_clean, lmax=lmax)*mk2uk**2
Cleaned_Dl = ell*(ell+1)/np.pi/2*cleaned_Cl  
Cleaned_Dl_no_beam = Cleaned_Dl/beam_func(1.)**2  # remove the effects of beam
fig,ax=plt.subplots(1,2)
ax[0].plot(ell[2:400], Cleaned_Dl_no_beam[2:400],
           label='cleaned_Dl with ILC (no beam)')
ax[0].plot(ell[2:400], Dl_wmap_no_beam[2:400],
           label='cleaned_Dl from wmap team (no beam)')
ax[1].loglog(ell[2:600], Cleaned_Dl_no_beam[2:600],
           label='cleaned_Dl with ILC (no beam)')
ax[1].loglog(ell[2:600], Dl_wmap_no_beam[2:600],
           label='cleaned_Dl from wmap team (no beam)')

plt.legend()
plt.show()




