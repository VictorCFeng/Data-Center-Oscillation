"""Frequency responses and nonlinear trajectory metrics for the paper cases."""
import argparse
import json
import numpy as np
from scipy.io import loadmat
from scipy.linalg import matrix_balance
from scipy.integrate import cumulative_trapezoid

from case_models import BASE, CAPS, interconnected, load_case, response
from data_utils import ctrl, dump, items
from design_specs import masks, scales
from trajectory_io import collect

OUT = BASE.parent / 'bilateral_final_comparison'
METHODS = ['uncontrolled','power_only','rbc_adapted','rbc_support_added','lead_support','lead_support_optimized','centered_q_g3_u20']
LABELS = ['No HESS','Power-only feedback','RBC (adapted)','RBC + support','Lead + support','Lead + support (optimized)','Proposed']

def read(p):return json.loads(p.read_text())

def scalar(x):return float(x.toarray().item()) if hasattr(x,'toarray') else float(x)

def frequency(kind,out=None,methods=None,labels=None):
    out=out or OUT;methods=methods or METHODS;labels=labels or LABELS
    models,f,H,target=load_case(kind);n=int(models[0]['nd']);iw,ig=masks(f,kind);io=(~iw)&(f<=200);results=[]
    for name,label in zip(methods,labels):
        file=out/f'{kind}_{name}_controller.mat'
        if not file.exists():continue
        d=items(loadmat(file,simplify_cells=True)['designs'])[0];a,b,c=ctrl(d,n)
        k=response(dict(A=a,B=b,C=c,D=np.zeros((n,3*n))),f)
        N=np.linalg.solve(np.eye(n)-k[None]@H[:,:,:3*n,n+1:],np.broadcast_to(k,(len(H),)+k.shape))
        U=N@H[:,:,:3*n,:n+1];Y=H[:,:,:,:n+1]+H[:,:,:,n+1:]@U
        iso=np.linalg.norm(Y[:,:,:n,:n],axis=2)/np.maximum(np.linalg.norm(H[:,:,:n,:n],axis=2),1e-4)
        ge=np.linalg.norm(Y[:,:,:n,n]-target[:,:,:n],axis=2)/np.maximum(np.linalg.norm(target[:,:,:n],axis=2),.05)
        noise=100*np.linalg.svd(N*scales(n),compute_uv=False)[...,0]
        pts=[]
        for op,m in enumerate(models):
            cl=interconnected(m,(a,b,c));ab,_=matrix_balance(cl['A']);p=float(np.linalg.eigvals(ab).real.max())
            z=dict(op=op+1,isolation=float(iso[op,iw].max()),grid_error=float(ge[op,ig].max()),noise_MW=float(noise[op].max()),outband=float(iso[op,io].max()),pole_max=p)
            z['pass']=bool(p<0 and z['isolation']<=.65*(1+2e-6) and z['grid_error']<=CAPS[kind][0]*(1+2e-6) and z['outband']<=2.8*(1+2e-6) and z['noise_MW']<=CAPS[kind][1]*np.sqrt(n)*(1+2e-6));pts.append(z)
        np.savez_compressed(out/f'{kind}_{name}_uniform_frequency.npz',frequency=f,Y=Y,U=U,N=N,target=target,isolation=iso,grid_error=ge,noise=noise)
        results.append(dict(name=name,label=label,controller_states=0 if name=='uncontrolled' else len(a),points=pts,pass_count=sum(p['pass'] for p in pts)))
        print('FREQUENCY',kind,name,results[-1]['pass_count'],flush=True)
    dump(out/f'{kind}_uniform_frequency_metrics.json',results)

def time_metrics(kind):
    rec,failed=collect(OUT,kind);jobs=read(OUT/f'{kind}_jobs.json')['jobs']
    extra=OUT.parent/'bilateral_power_only_verification'
    if (extra/f'{kind}_jobs.json').exists():
        more,bad=collect(extra,kind);rec.update(more);failed+=bad;jobs+=read(extra/f'{kind}_jobs.json')['jobs']
    missing=[j['key'] for j in jobs if j['key'] not in rec]
    assert not missing,(kind,'missing',len(missing))
    def get(name,op,event):return next((v for v in rec.values() if v['name']==name and v['op']==op and v['event']==event),None)
    n=1 if kind=='single_sm' else 2;rows=[]
    for name in METHODS:
        for op in range(1,14):
            ref=get('service_reference',op,'grid')
            if ref is None:continue
            for ev in ['server','grid','combined']:
                r=get(name,op,ev)
                if r is None:continue
                t=r['t']
                expected=np.zeros((len(t),n)) if ev=='server' else ref['y'][:,:n]
                lo,hi={'server':(2,26),'grid':(5,30),'combined':(2,30)}[ev];sel=(t>=lo)&(t<=hi)
                err=100*(r['y'][:,:n]-expected);e2=np.sum(err**2,axis=1);u=100*np.asarray(r['u']).reshape(len(t),n)
                energy=cumulative_trapezoid(u,t,axis=0,initial=0)/3.6
                rows.append(dict(name=name,op=op,event=ev,rms_MW=float(np.sqrt(np.trapz(e2[sel],t[sel])/(hi-lo))),
                                 error_peak_MW=float(np.sqrt(e2[sel].max())),hess_peak_MW=float(abs(u).max()),
                                 energy_excursion_kWh=float(np.ptp(energy,axis=0).max()),machine_drop_Hz=float(max(0,-r['y'][:,-2].min())),
                                 voltage_peak_pu=float(abs(r['y'][:,n:2*n]).max()),kcl_max=scalar(r['kcl_max']),clipping_max=scalar(r['clipping_max'])))
    report=dict(kind=kind,completed=len(rec),expected=len(jobs),failures=[v for v in failed if v['key'] not in rec],rows=rows)
    dump(OUT/f'{kind}_uniform_time_metrics.json',report);return report,rec

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['frequency','time']);p.add_argument('kind',choices=['single_sm','ieee9_two_dc']);a=p.parse_args()
    if a.action=='frequency':frequency(a.kind)
    else:time_metrics(a.kind)
