import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import time
import numpy as np
from astropy.io import fits
from multiprocessing import Pool
from multiprocessing import Process
import argparse
from astropy.constants import G,c,M_sun,pc
import emcee
from models_profiles import *
from colossus.cosmology import cosmology  
from colossus.halo import concentration
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MyCosmo', params)
cosmo = cosmology.setCosmology('MyCosmo')
from astropy.cosmology import LambdaCDM
from astropy.constants import G,c,M_sun,pc
cosmo_ap = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

cmodel = 'diemer19'


#parameters
cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)


pro = fits.open('../Kappa_Submuestra1_5deg.fits')[1].data

pro2t = fits.open('../KappaProyectado_Submuestra1_Tesis.fits')[1].data


p_mice = fits.open('../profile_CMB_test2.fits')[1].data
cov    = fits.open('../profile_CMB_test2.fits')[2].data

CovS      = np.sqrt(np.diag(cov.COV_S.reshape(len(p_mice),len(p_mice))))
CovS_cos  = np.sqrt(np.diag(cov.COV_Scos.reshape(len(p_mice),len(p_mice))))


theta = pro['Radio']
kappa = pro['Media Radial']
kappa_par = pro['Media Paralelo']
kappa_per = pro['Media Perpendicular']
err = pro['Error Radial']

theta2t = pro2t['Radio [arcmin]']
kappa2t = pro2t['Media Radial']



zl = 0.42
zs = 1100.

dl  = cosmo_ap.angular_diameter_distance(zl).value
ds  = cosmo_ap.angular_diameter_distance(zs).value
dls = cosmo_ap.angular_diameter_distance_z1z2(zl, zs).value

KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0

BETA_array = dls/ds

Dl = dl*1.e6*pc
sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)

Sigma  = kappa*sigma_c
Sigma2t  = kappa2t*sigma_c
Sigma_par  = kappa_par*sigma_c
Sigma_per  = kappa_per*sigma_c
eSigma  = err*sigma_c
r      = (theta*60.*KPCSCALE)/1.e3
r2t      = (theta2t*60.*KPCSCALE)/1.e3

folder    = '../'

logM = 14.55
c200 = concentration.concentration(10**logM, '200c', zl, model = 'diemer19')

mr  = r2t < 50.
mr2 = r < 50.

s2      = S2_quadrupole(p_mice.Rp[:-3],zl,M200 = 10**logM,c200=c200,cosmo_params=params,terms='1h',pname='Einasto')
s2_2h   = S2_quadrupole(p_mice.Rp[:-3],zl,M200 = 10**logM,c200=c200,cosmo_params=params,terms='2h',pname='Einasto')
s       = Sigma_Ein_2h(p_mice.Rp[:-3],zl,M200 = 10**logM,c200=c200,alpha=0.3,cosmo_params=params,terms='1h')
s_2h    = Sigma_Ein_2h(p_mice.Rp[:-3],zl,M200 = 10**logM,c200=c200,alpha=0.3,cosmo_params=params,terms='2h')

plt.figure()
plt.xlabel(r'$R [Mpc/h]$')
plt.ylabel(r'$e \times \Sigma_2 [M_{\odot}pc^{-2} h ]$')
plt.savefig('../profile_S2.png',bbox_inches='tight')
plt.plot(r2t,Sigma2t,'C0',lw=3)
plt.plot(p_mice.Rp[:-3],0.2*s2,'C1',label='S2 - 1halo - e=0.2')
plt.plot(p_mice.Rp[:-3],0.4*s2_2h,'C1--',label='S2 - 2halo - e=0.4')
plt.plot(p_mice.Rp[:-3],0.2*s2+0.4*s2_2h,'C3',label='S2 - 1h + 2h')
plt.plot(p_mice.Rp,p_mice.SIGMA_cos,'k',label='MICE')
plt.fill_between(p_mice.Rp,p_mice.SIGMA_cos+CovS_cos,p_mice.SIGMA_cos-CovS_cos,color='C7',alpha=0.4)
plt.axis([0,50,-1,10])
plt.legend()
plt.savefig('../profile_S2.png',bbox_inches='tight')



plt.figure()
plt.xlabel(r'$R [Mpc/h]$')
plt.ylabel(r'$\Sigma [M_{\odot}pc^{-2} h ]$')
plt.plot(r,Sigma,'C0',label='radial',lw=3)
plt.plot(r,Sigma_par,'C0--',label='paralelo',lw=3)
plt.plot(r,Sigma_per,'C0:',label='perpendicular',lw=3)
plt.axis([0.5,50,1,300])
plt.loglog()
plt.plot(p_mice.Rp[:-3],s,'C1',label='S2 - 1halo')
plt.plot(p_mice.Rp[:-3],s_2h,'C1--',label='S2 - 2halo')
plt.plot(p_mice.Rp[:-3],s+s_2h,'C3',label='S2 - 1halo + 2halo')
plt.plot(p_mice.Rp,p_mice.Sigma,'k',label='MICE')
plt.plot(p_mice.Rp,p_mice.SIGMA_par,'k--',label='MICE - par')
plt.plot(p_mice.Rp,p_mice.SIGMA_per,'k:',label='MICE - per')
plt.fill_between(p_mice.Rp,p_mice.Sigma+CovS,p_mice.Sigma-CovS,color='C7',alpha=0.4)
plt.legend(frameon=False)
plt.savefig('../profile_S.png',bbox_inches='tight')



'''	
nit       = 250
ncores    = 32
RIN       = 100
ROUT      = 10000


outfile     = 'fitresults_'+str(int(RIN))+'_'+str(int(ROUT))+'.fits'



print('fitting profiles')
print('ncores = ',ncores)
print('RIN ',RIN)
print('ROUT ',ROUT)
print('nit', nit)
print('outfile',outfile)



### compute dilution


def log_likelihood(logM, R, S, eS):
    
    c200 = concentration.concentration(10**logM, '200c', zl, model = cmodel)
    
    s   = Sigma_NFW_2h(R,zl,M200 = 10**logM,c200=c200,cosmo_params=params,terms='1h+2h')    
    
    sigma2 = eDS**2
    return -0.5 * np.sum((DS - ds)**2 / sigma2 + np.log(2.*np.pi*sigma2))


def log_probability(logM, R, S, eS):
    
    
    if 11. < logM < 15.:
        return log_likelihood(logM, R, S, eS)
        
    return -np.inf

# initializing

pos = np.array([np.random.uniform(14.,14.5,20)]).T

nwalkers, ndim = pos.shape

#-------------------
# running emcee

maskr   = (r > (RIN/1000.))*(r < (ROUT/1000.))


t1 = time.time()


pool = Pool(processes=(ncores))    
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                args=(r[maskr],Sigma[maskr],eSigma[maskr]),
                                pool = pool)
				
sampler.run_mcmc(pos, nit, progress=True)
pool.terminate()
    
    
print('TOTAL TIME FIT')    
print((time.time()-t1)/60.)

#-------------------
# saving mcmc out

mcmc_out = sampler.get_chain(flat=True)

table = [fits.Column(name='logM', format='E', array=mcmc_out)]

tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(table))

logM    = np.percentile(mcmc_out[2500:], [16, 50, 84])
c200 = concentration.concentration(10**logM[1], '200c', zmean, model = cmodel)



h = fits.Header()
h.append(('c200',np.round(c200,4)))

h.append(('lM200',np.round(logM[1],4)))
h.append(('elM200_min',np.round(np.diff(logM)[0],4)))
h.append(('elM200_max',np.round(np.diff(logM)[1],4)))
primary_hdu = fits.PrimaryHDU(header=h)

hdul = fits.HDUList([primary_hdu, tbhdu])
hdul.writeto(folder+outfile,overwrite=True)

print('SAVED FILE')



'''
