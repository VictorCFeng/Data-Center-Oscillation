"""Service bands, quadrature weights, and the engineering-controller transfers."""
import json
import numpy as np
from grid_reference import ref_transfer
from case_models import OUT

DEST = OUT / 'service_validated'

def spec(kind):
    return json.loads((DEST/(kind+'_target_v2_admission.json')).read_text())

def masks(f,kind):
    iw=(f>=1)&(f<=20); ig=np.zeros(len(f),bool)
    for lo,hi in spec(kind)['service_bands_Hz']: ig|=(f>=lo)&(f<=hi)
    return iw,ig

def weights(f,mask):
    # Quadrature of mean squared error per log-frequency interval. No dependence
    # on an arbitrary count of plotting points, and exactly zero outside band.
    w=np.zeros(len(f)); x=np.log(f[mask]); d=np.diff(x)
    w[mask]=np.r_[d[0]/2,(d[:-1]+d[1:])/2,d[-1]/2]/(x[-1]-x[0])
    return w

def scales(n):
    return np.r_[np.full(n,.003),np.full(n,.001),np.full(n,.01)]

def direct_transfers(theta,f,n,reference):
    s=2j*np.pi*f; b=ref_transfer(reference,f,n)*np.asarray(theta[3:3+n])[None,:]
    kd,tz,fc=theta[:3];wc=2*np.pi*fc
    a=200/(s+200)*kd*s/(s+2*np.pi*.002)*(1+.001*tz*s)*(wc/(s+wc))**2
    return a,b
