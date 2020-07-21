import healpy as hp
import numpy as np
import time
import matplotlib.pyplot as plt

start_t = time.time()
map_path = '/home/wudl/project/ILC/fits_files/wmap/'
map_name = ['K','Ka','Q','V','W']
lmax = 1024
mk2uk = 10**3
nside=512
map_num = len(map_name)

#load 5 band maps, which are smoothed to 1 degree by wmap team 
map_list = [hp.read_map(map_path + 'wmap_band_smth_imap_r9_3yr_' +\
                       map_name[i]+'_v2.fits') for i in np.arange(map_num)]
alm_comb = np.array([hp.map2alm(map_list[i], lmax=lmax) for i in np.arange(map_num)])

# load wmap simulated noise from realization
alm_comb_n = np.load('alm_comb_noise_smoothed_with_bl_own.npy')
n_realization = np.shape(alm_comb_n)[1]

def w_ell():
    Cl_list = []
    for i in np.arange(map_num):
        for j in np.arange(map_num):
            Cl_list.append(hp.anafast(map1=map_list[i],map2=map_list[j],lmax=lmax))
    Cl = np.array(Cl_list).T.reshape(-1,map_num,map_num)
    #save Cl
    np.save('Cl_matrix',Cl)
    print('Cl_shape:>>> ',np.shape(Cl))
    Cl_inv = np.linalg.inv(Cl)
    Cl_numr = np.sum(Cl_inv,axis=2)
    Cl_den = np.sum(Cl_numr,axis=1)
    wl = Cl_numr/Cl_den.reshape(-1,1)
    # save weights
    np.save('weights_ell',wl)
    print('shape of weight>>>: ',np.shape(wl))
    return wl,Cl

'''
# calculate cleaned Cl, Dl (another way to compute cleaned_Cl)
def cleaned_Dl():
   wl,Cl = w_ell()
   shape_wl = np.shape(wl)
   wl = wl.reshape(shape_wl[0],shape_wl[1],1)
   wl_T = wl.transpose(0,2,1)
   clean_Cl = []
   for i in np.arange(shape_wl[0]):
       clean_Cl.append(np.dot(np.dot(wl_T[i],Cl[i]),wl[i]))
   clean_Cl = np.array(clean_Cl).reshape(1,-1)[0]
   ell = np.arange(len(clean_Cl))
   print('shape of Clean Cl>>>: ',np.shape(clean_Cl))
   clean_Dl = ell*(ell+1)/2/np.pi*clean_Cl
   return clean_Cl,clean_Dl
'''

#one can also try hp.almxfl function to compute alm times B_ell
def cleaned_map():
    weights = w_ell()[0]
    for ell in np.arange(lmax+1):
        print('ell>>>',ell)
        for m in np.arange(ell+1):
            for i in np.arange(map_num):
                alm_comb[i,hp.Alm.getidx(lmax = lmax,l=ell,m=m)]=alm_comb[i,hp.Alm.getidx(lmax = lmax,l=ell,m=m)]*weights[ell,i]
            for j in np.arange(n_realization):
                alm_comb_n[:,j,hp.Alm.getidx(lmax = lmax,l=ell,m=m)]=alm_comb_n[:,j,hp.Alm.getidx(lmax = lmax,l=ell,m=m)]*weights[ell]
    cleaned_alm = np.sum(alm_comb,axis=0)
    noise_bias = np.sum(alm_comb_n,axis=0)
    Cl_noise = np.sum(
        hp.alm2cl(noise_bias, nspec=n_realization), axis=0)/n_realization
    np.save('Cl_noise_sim',Cl_noise)
    print('shape_cleaned_alm',np.shape(cleaned_alm))
    cleaned_Cl = hp.alm2cl(cleaned_alm)
    Cl_debias = cleaned_Cl - Cl_noise
    cleaned_map = hp.alm2map(cleaned_alm,nside=512)
    ell = np.arange(len(cleaned_Cl))
    cleaned_Dl = ell*(ell+1)/2/np.pi*cleaned_Cl
    cleaned_Dl_debias = ell*(ell+1)/2/np.pi*Cl_debias
    return cleaned_map,cleaned_Dl,cleaned_Dl_debias

def beam_func(fwhm,data_lmax):
    return hp.gauss_beam(fwhm=np.radians(fwhm),lmax=data_lmax)

#show original WMAP maps
for i in np.arange(map_num):
    hp.mollview(map_list[i],sub=(2,3,i+1),unit=r'mK$_\mathrm{CMB}$',norm='hist',min=-0.4,max=0.4,title= map_name[i] + "-band")

#show weights w_l
plt.figure()
wl = w_ell()[0]
for band in np.arange(map_num):
    plt.subplot(2,3,band+1,title='weights of '+map_name[band]+'-band')
    plt.semilogx(np.arange(len(wl[:,band])),wl[:,band])

#ILC processing
clean_map,clean_Dl,clean_Dl_debias = cleaned_map()
clean_Dl = clean_Dl/beam_func(1., lmax)**2*mk2uk**2                          #unit: mk--->uk, remove beam
clean_Dl_debias = clean_Dl_debias/beam_func(1.,lmax)**2*mk2uk**2             #unit: mk--->uk, remove beam
np.save('cleaned_map',clean_map)
np.save('cleaned_Dl',clean_Dl)
np.save('cleaned_Dl_debias',clean_Dl_debias)
ell = np.arange(len(clean_Dl))

#compared Dl to wmap team and planck data
data_planck = np.loadtxt(map_path+'COM_PowerSpect_CMB-TT-full_R3.01.txt')  #planck best-fit tt
ell_planck = data_planck[:,0]
Dl_planck = data_planck[:,1]

'''
wmap cleaned_map
the cleaned_map is generated by cross power spectra to remove
noise and using templetes to remove foregrounds (not by ILC)
'''
data_wmap_3y = np.loadtxt(map_path+'wmap_combined_tt_powspec_3y.txt')
ell_wmap = data_wmap_3y[:,0]
Dl_wmap_cross = data_wmap_3y[:, 1]

'''
wmap ILC result
cleaned map from wmap team via ILC method, note that the map have been smoothed to 1 degree
and the inputted 5 original maps are divided into 12 regions, and the boundardies are smoothed
with 1.5 degree.
please see https://lambda.gsfc.nasa.gov/product/map/dr5/ilc_map_info.cfm
'''
ILC_wmap = hp.read_map(
    '/home/wudl/project/ILC/fits_files/wmap/wmap_ilc_3yr_v2.fits')
Cl_wmap_ILC = hp.anafast(ILC_wmap, lmax=lmax)*mk2uk**2   # mk ---> uk
Dl_wmap_ILC = ell*(ell+1)/2/np.pi*Cl_wmap_ILC/beam_func(1.,lmax)**2    #remove the effects of beam

# planck best fit cmb TT
planck_bf = np.loadtxt('/home/wudl/project/ILC/fits_files/planck/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')
ell_bf = planck_bf[:,0]
Dl_bf = planck_bf[:,1]

#linear plot (without bin)
plt.figure()
plt.plot(ell_wmap, Dl_wmap_cross, label='wmap_CMB_TT (via cross powspec)')
plt.plot(ell, Dl_wmap_ILC, label='ILC cleaned_Dl from wmap team')
plt.plot(ell_bf,Dl_bf,label='planck best-fit CMB TT')
plt.plot(ell, clean_Dl, label='before debias(ILC, ell_domain)')
plt.plot(ell, clean_Dl_debias, label='after debias(ILC, ell_domain)')
plt.xlim(2,600)
plt.ylim(-0,20000)
plt.xlabel('$\ell$')
plt.ylabel('$D_\ell(\mu K^2)$')
plt.title('cleaned Dl (linear scale)')
plt.legend()

#log plot
plt.figure()
plt.loglog(ell_wmap, Dl_wmap_cross, label='wmap_CMB_TT (via cross powspec)')
plt.loglog(ell, Dl_wmap_ILC, label='ILC cleaned_map from wmap team')
plt.loglog(ell_bf, Dl_bf, label='planck best-fit CMB TT')
plt.loglog(ell, clean_Dl, label='before debias(ILC, ell_domain)')
plt.loglog(ell, clean_Dl_debias, label='after debias(ILC, ell_domain)')
plt.xlim(2, 600)
plt.ylim(10**(2), 10**6)
plt.xlabel('$\ell$')
plt.ylabel('$D_\ell(\mu K^2)$')
plt.title('cleaned Dl (log scale)')
plt.legend()

'''
#linear plot (bin)
plt.figure()
plt.plot(ell_wmap,Dl_wmap_smoothed,label='wmap_CMB_TT')
plt.plot(ell_bf,Dl_bf_smoothed,label='planck best-fit CMB TT')
bin_size = 5
Dl_debias_bin = np.mean(clean_Dl_debias[:len(clean_Dl_debias)//bin_size*bin_size].reshape(-1,bin_size),axis=1)
ell_bin = np.round(np.mean(ell[:len(ell)//bin_size*bin_size].reshape(-1,bin_size),axis=1))
plt.plot(ell_bin, Dl_debias_bin, label='cleaned_Dl_debias (bin_size=5)')
plt.xlim(2, 900)
plt.xlabel('$\ell$')
plt.ylabel('$D_\ell(\mu K^2)$')
plt.title('cleaned Dl')
plt.legend()

#log plot (bin)
plt.figure()
#plt.loglog(ell,clean_Dl*mk2uk**2,label='cleaned_Dl from ILC')
plt.loglog(ell_wmap, Dl_wmap_smoothed, label='wmap_CMB_TT')
#plt.loglog(ell_planck,Dl_planck_smoothed,label='planck_CMB_TT')
plt.loglog(ell_bf, Dl_bf_smoothed, label='planck best-fit CMB TT')
plt.loglog(ell_bin, Dl_debias_bin, label='cleaned_Dl_debias')
plt.xlim(2, 600)
plt.xlabel('$\ell$')
plt.ylabel('$D_\ell(\mu K^2)$')
plt.title('cleaned Dl')
plt.legend()
'''

#show cleaned map
hp.mollview(clean_map,norm='hist',unit=r'mK$_\mathrm{CMB}$',title='cleaned_map (ILC)')

#show noise map after multiplying by weights W_l
#hp.mollview(hp.alm2map(alms=noise_bias,nside=nside),norm='hist',unit=r'mK$_\mathrm{CMB}$',title='wmap_sim_noise_multiplied_by_wl')
end_t = time.time()
print('time cose>>>: '+str((end_t-start_t)/60)+' mins')
plt.show()
