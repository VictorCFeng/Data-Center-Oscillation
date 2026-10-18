"""Frequency-support reference transfer function."""
import numpy as np

def ref_transfer(theta, f, n):
    s = 2j*np.pi*np.asarray(f)
    base = .2*s/(s+.1)*4*np.pi/(s+4*np.pi)
    # Broad POD branch: gain/site, plus shared lead-lag zero and pole (Hz).
    gains = np.asarray(theta[:n]); z,p = ref_phase(theta,n)
    pod = (s/(s+np.pi)*24*np.pi/(s+24*np.pi))[:,None]*(1+s[:,None]/(2*np.pi*z))/(1+s[:,None]/(2*np.pi*p))
    return (base[:,None]+pod*gains)*200/(s+200)[:,None]

def ref_phase(theta,n):
    return (np.full(n,theta[n]),np.full(n,theta[n+1])) if len(theta)==n+2 else (np.asarray(theta[n:2*n]),np.asarray(theta[2*n:]))
