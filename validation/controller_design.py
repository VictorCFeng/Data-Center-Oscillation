"""Reference-centered affine controller synthesis, Eqs. (27)-(29)."""
import argparse
import json
import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, matrix_balance
from scipy.io import savemat

from case_models import BASE, CAPS, TRAIN, interconnected, load_case, response
from data_utils import ctrl, dump
from design_specs import direct_transfers, masks, scales, spec, weights

OUT = BASE.parent / 'bilateral_final_comparison'

def affine_rows(mapping, x, offset=None):
    expr = mapping.reshape(-1, mapping.shape[-1])@x
    if offset is not None: expr = expr + np.asarray(offset).reshape(-1)
    return expr

def rownorm(expr, rows, width):
    return cp.norm(cp.reshape(expr, (rows, width), order='C'), axis=1)

def setup(kind):
    models,f,H,target=load_case(kind);n=int(models[0]['nd'])
    th=np.array(json.loads((BASE/'benchmark_parameters.json').read_text())[kind])
    ref=spec(kind)['theta'];wc=2*np.pi*th[2]
    rates=([[.1,4*np.pi,wc]] if n==1 else
           [[.1,4*np.pi,np.pi,24*np.pi,2*np.pi*ref[4],wc],[.1,4*np.pi,np.pi,2*np.pi*ref[5],wc]])
    s=2j*np.pi*f;w=200*s/((s+200)*(s+2*np.pi*.002));phi=[]
    for rr in rates:phi.append(w[:,None]*np.column_stack([np.ones(len(f)),*[a/(s+a) for a in rr],(rr[-1]/(s+rr[-1]))**2]))
    allowed=[(i,j,k) for i in range(n) for j in [i,n+i,2*n+i] for k in range(phi[i].shape[1])]
    qm=np.zeros((len(f),n,3*n,len(allowed)),complex)
    for col,(i,j,k) in enumerate(allowed):qm[:,i,j,col]=phi[i][:,k]*(10 if n<=j<2*n else 1)
    a,b=direct_transfers(th,f,n,ref);qb=np.zeros((len(f),n,3*n),complex)
    for i in range(n):qb[:,i,i]=-a;qb[:,i,2*n+i]=b[:,i]
    mat=qm.reshape(-1,len(allowed));xb=np.linalg.lstsq(np.vstack([mat.real,mat.imag]),np.r_[qb.real.ravel(),qb.imag.ravel()],rcond=None)[0]
    h0=np.zeros((len(f),3*n,n),complex);h0[:,:n]=-np.eye(n)
    return dict(kind=kind,models=models,f=f,H=H,target=target,n=n,rates=rates,qm=qm,allowed=allowed,xb=xb,qb=qb,h0=h0)

def affine_maps(d):
    H,qm,h0,target,n=d['H'],d['qm'],d['h0'],d['target'],d['n']
    pu=H[:,:,:n,n+1:];pi=np.linalg.inv(pu);diff=H[:,:,:3*n,n+1:]-h0
    em=np.einsum('fija,ofjk->ofika',qm,diff)
    eo=np.einsum('ofij,ofjka,ofkl->ofila',pu,em,pi)
    pq=np.einsum('ofij,fjka->ofika',pu,qm)
    base=H[:,:,:n,:n+1].copy();base[:,:,:,n]-=target[:,:,:n]
    nm=np.einsum('ofija,ofjk->ofika',eo,base)-np.einsum('ofija,ofjk->ofika',pq,H[:,:,:3*n,:n+1])
    sm=-np.einsum('ofija,fjk,ofkl->ofila',pq,h0,pi)
    eb=d['qb'][None]@diff;ebo=pu@eb@pi
    inv=np.linalg.inv(np.eye(n)+eb);invo=np.linalg.inv(np.eye(n)+ebo)
    am=np.einsum('ofij,ofjka->ofika',inv,em);aom=np.einsum('ofij,ofjka->ofika',invo,eo)
    nmw=np.einsum('ofij,ofjka->ofika',invo,nm)
    nw0=np.einsum('ofij,ofjk->ofik',invo,base)
    swm=np.einsum('ofij,ofjka->ofika',invo,sm)
    qq=np.einsum('ofij,fjka->ofika',inv,qm)*scales(n)[None,None,None,:,None]
    q0=inv@d['qb'][None]*scales(n)
    d.update(pu=pu,pi=pi,em=em,eo=eo,base=base,nm=nm,sm=sm,inv=inv,invo=invo,am=am,aom=aom,nmw=nmw,nw0=nw0,swm=swm,qq=qq,q0=q0)
    return d

def components(d):
    f,H,n=d['f'],d['H'],d['n'];iw,_=masks(f,d['kind'])
    ws=weights(f,iw);wn=weights(f,np.ones(len(f),bool));df=np.diff(f)
    wg=np.r_[df[0]/2,(df[:-1]+df[1:])/2,df[-1]/2]/(2*np.pi*f)**2
    wg/=np.mean(np.sum(np.sum(abs(d['target'][TRAIN,:,:n])**2,axis=2)*wg,axis=1))
    nm=d['nm'];base=d['base'];rr=[]
    for what in ('server','grid','q'):
        chunks=[]
        if what=='q':
            a=d['qm']*scales(n)[None,None,:,None]*np.sqrt(wn)[:,None,None,None]
            chunks=[np.c_[a.reshape(-1,a.shape[-1]),np.zeros(a.size//a.shape[-1])]]
        else:
            for j in (range(n) if what=='server' else [n]):
                den=np.maximum(np.linalg.norm(H[:,:,:n,j],axis=2),1e-4) if j<n else np.ones(H.shape[:2])
                ww=(ws/n if j<n else wg)/len(TRAIN)
                a=nm[TRAIN,:,:,j,:]/den[TRAIN,:,None,None]*np.sqrt(ww)[None,:,None,None]
                b=base[TRAIN,:,:,j]/den[TRAIN,:,None]*np.sqrt(ww)[None,:,None]
                chunks.append(np.c_[a.reshape(-1,a.shape[-1]),b.ravel()])
        z=np.vstack(chunks);rr.append(np.linalg.qr(np.vstack([z.real,z.imag]),mode='r'))
    return rr

def herm_dd(a):
    h=(a+np.swapaxes(a.conj(),-1,-2))/2
    diag=np.diagonal(h,axis1=-2,axis2=-1).real
    return np.min(2*diag-np.sum(abs(h),axis=-1),axis=-1)

def ratios(d,x):
    n,f=d['n'],d['f'];iw,ig=masks(f,d['kind']);io=(~iw)&(f<=200)
    a=d['inv']+d['am']@x;ao=d['invo']+d['aom']@x
    du=herm_dd(a);dy=herm_dd(ao);safe=np.maximum(dy,1e-10)
    nn=d['nw0']+d['nmw']@x
    ratios={'accretive_input':1e-5/np.maximum(du,1e-12),'accretive_output':1e-5/np.maximum(dy,1e-12)}
    for j in range(n+1):
        den=np.maximum(np.linalg.norm(d['H'][:,:,:n,j] if j<n else d['target'][:,:,:n],axis=2),1e-4 if j<n else .05)
        ratios['server'+str(j) if j<n else 'grid']=np.where(iw if j<n else ig,np.linalg.norm(nn[:,:,:,j],axis=2)/den/safe/(.65 if j<n else CAPS[d['kind']][0]),0)
    sw=d['invo']+d['swm']@x
    rows=np.max(np.sum(abs(sw),axis=3),axis=2);cols=np.max(np.sum(abs(sw),axis=2),axis=2)
    ratios['sensitivity']=np.where(io,np.maximum(rows,cols)/safe/2.8,0)
    noise0=np.linalg.svd(d['q0'],compute_uv=False)[...,0]
    noiseup=noise0+np.linalg.norm(d['qq']@x-d['q0'],axis=(2,3))
    ratios['noise']=noiseup/np.maximum(du,1e-10)/(CAPS[d['kind']][1]*np.sqrt(n)/100)
    return ratios

def realization(d,x):
    n=d['n'];blocks=[]
    for i,rr in enumerate(d['rates']):
        coef=np.zeros((3,len(rr)+2))
        for val,(ii,j,k) in zip(x,d['allowed']):
            if ii==i:coef[[i,n+i,2*n+i].index(j),k]=val*(10 if j==n+i else 1)
        m=len(rr)+1;aa=np.diag(-np.r_[rr,rr[-1]]);aa[m-2,m-1]=rr[-1]
        bb=np.vstack([np.asarray(rr)[:,None]*coef[:,1:-1].T,rr[-1]*coef[:,-1]])
        cc=np.r_[np.ones(m-1),0.][None];dd=coef[:,0][None];wash=2*np.pi*.002
        aq=np.block([[aa,np.zeros((m,2))],[wash*cc,np.array([[-wash,0.]])],[200*cc,np.array([[-200.,-200.]])]])
        bq=np.vstack([bb,wash*dd,200*dd]);cq=np.zeros((1,m+2));cq[0,-1]=1
        hu0=np.array([[-1.],[0.],[0.]]);ac=aq+bq@hu0@cq;gc=-bq@hu0;out=-cq
        ac,tr=matrix_balance(ac);blocks.append((ac,np.linalg.solve(tr,bq),out@tr,np.linalg.solve(tr,gc)))
    a=block_diag(*[v[0] for v in blocks]);b=np.zeros((len(a),3*n));c=np.zeros((n,len(a)));g=np.zeros((len(a),n));off=0
    for i,(aa,bb,cc,gg) in enumerate(blocks):
        sl=slice(off,off+len(aa));b[sl,[i,n+i,2*n+i]]=bb;c[i,sl]=cc;g[sl,i]=gg[:,0];off+=len(aa)
    return dict(Ac=a,Bc=b,Cc=c,Gu=g)

def evaluate_controller(d,x,name,out=None):
    out=out or OUT;n,f,H=d['n'],d['f'],d['H'];q=d['qm']@x;eta=d['em']@x
    N=-np.linalg.solve(np.eye(n)+eta,np.broadcast_to(q,(len(H),)+q.shape))
    U=N@H[:,:,:3*n,:n+1];Y=H[:,:,:,:n+1]+H[:,:,:,n+1:]@U
    iw,ig=masks(f,d['kind']);io=(~iw)&(f<=200)
    iso=np.linalg.norm(Y[:,:,:n,:n],axis=2)/np.maximum(np.linalg.norm(H[:,:,:n,:n],axis=2),1e-4)
    ge=np.linalg.norm(Y[:,:,:n,n]-d['target'][:,:,:n],axis=2)/np.maximum(np.linalg.norm(d['target'][:,:,:n],axis=2),.05)
    noise=100*np.linalg.svd(N*scales(n),compute_uv=False)[...,0];dc=realization(d,x);pts=[]
    for op,m in enumerate(d['models']):
        cl=interconnected(m,ctrl(dc,n));a,_=matrix_balance(cl['A']);p=float(np.linalg.eigvals(a).real.max())
        pt=dict(op=op+1,isolation=float(iso[op,iw].max()),grid_error=float(ge[op,ig].max()),outband=float(iso[op,io].max()),noise_MW=float(noise[op].max()),pole_max=p)
        pt['pass']=bool(p<0 and pt['isolation']<=.65*(1+2e-6) and pt['grid_error']<=CAPS[d['kind']][0]*(1+2e-6) and pt['outband']<=2.8*(1+2e-6) and pt['noise_MW']<=CAPS[d['kind']][1]*np.sqrt(n)*(1+2e-6));pts.append(pt)
    rec=dict(name=name,kind=d['kind'],controller_states=len(dc['Ac']),points=pts,pass_count=sum(p['pass'] for p in pts))
    dump(out/f'{d["kind"]}_{name}_performance.json',rec);dc.update(name=name,kind=d['kind'])
    savemat(out/f'{d["kind"]}_{name}_controller.mat',dict(designs=np.array([dc],dtype=object)))
    np.savez_compressed(out/f'{d["kind"]}_{name}_frequency.npz',frequency=f,Y=Y,U=U,N=N,Q=q,eta=eta,isolation=iso,grid_error=ge,noise=noise)
    return rec

def solve(kind,wg=3.,wu=20.,data=None,out=None,evaluate_result=True):
    out=out or OUT;out.mkdir(parents=True,exist_ok=True)
    d=affine_maps(data if data is not None else setup(kind));f=d['f'];n=d['n'];rr=components(d)
    name=f'centered_q_g{wg:g}_u{wu:g}'.replace('.','p')
    stack=np.vstack([rr[0][:,:-1],np.sqrt(wg)*rr[1][:,:-1],np.sqrt(wu)*rr[2][:,:-1]])
    _,ss,vt=np.linalg.svd(stack,full_matrices=False);transform=(vt.T/ss)/100
    ix=set(int(np.argmin(abs(f-v))) for v in np.r_[np.geomspace(.001,1e4,85),np.geomspace(.005,20,65),np.linspace(1,5.5,65),.005,1,20,200])
    rounds=[];val=None
    for step in range(7):
        ids=np.array(sorted(ix));idsop=np.ix_(TRAIN,ids);count=len(TRAIN)*len(ids);v=cp.Variable(len(d['allowed']));x=d['xb']+transform@v;du=cp.Variable(count);dy=cp.Variable(count)
        cs=[du>=1e-5,dy>=1e-5];iw,ig=masks(f[ids],kind);io=(~iw)&(f[ids]<=200)
        def aff(key,offset=None):return affine_rows(d[key][idsop],x,None if offset is None else d[offset][idsop])
        for key,offset,delta in [('am','inv',du),('aom','invo',dy)]:
            a=cp.reshape(aff(key,offset),(count,n*n),order='C')
            if n==1:cs.append(cp.real(a[:,0])>=delta)
            else:
                off=cp.abs((a[:,1]+cp.conj(a[:,2]))/2)
                cs.extend([cp.real(a[:,0])>=delta+off,cp.real(a[:,3])>=delta+off])
        for j in range(n+1):
            den=np.maximum(np.linalg.norm(d['H'][idsop][:,:,:n,j] if j<n else d['target'][idsop][:,:,:n],axis=2),1e-4 if j<n else .05)
            a=d['nmw'][idsop][:,:,:,j,:]/den[:,:,None,None];b=d['nw0'][idsop][:,:,:,j]/den[:,:,None]
            z=rownorm(affine_rows(a,x,b),count,n);use=np.broadcast_to(iw if j<n else ig,(len(TRAIN),len(ids))).ravel()
            cs.append(z[use]<=dy[use]*(.65 if j<n else CAPS[kind][0]))
        sw=cp.reshape(aff('swm','invo'),(count,n*n),order='C');use=np.broadcast_to(io,(len(TRAIN),len(ids))).ravel()
        # Bound both maximum row and column sums, hence the spectral norm.
        for i in range(n):
            cs.append(cp.sum(cp.abs(sw[:,i*n:(i+1)*n]),axis=1)[use]<=2.8*dy[use])
            cs.append(cp.sum(cp.abs(sw[:,i::n]),axis=1)[use]<=2.8*dy[use])
        noise0=np.linalg.svd(d['q0'][idsop],compute_uv=False)[...,0].ravel()
        noisechange=rownorm(affine_rows(100*d['qq'][idsop],x,-100*d['q0'][idsop]),count,3*n*n)
        cs.append(noisechange+100*noise0<=du*(CAPS[kind][1]*np.sqrt(n)))
        costs=[cp.sum_squares(z[:,:-1]@x+z[:,-1]) for z in rr];prob=cp.Problem(cp.Minimize(1e4*(costs[0]+wg*costs[1]+wu*costs[2])),cs)
        print('SOLVE',kind,name,step,len(ids),flush=True)
        try:
            prob.solve(solver='CLARABEL',tol_feas=1e-7,tol_gap_abs=1e-7,tol_gap_rel=1e-7,max_iter=250,max_threads=1)
        except cp.error.SolverError:
            prob.solve(solver='CLARABEL',tol_feas=5e-7,tol_gap_abs=5e-7,tol_gap_rel=5e-7,max_iter=350,max_threads=1,equilibrate_enable=False,warm_start=False)
        if x.value is None:raise RuntimeError((kind,prob.status))
        val=x.value.copy();rat=ratios(d,val);peaks={k:float(a[TRAIN].max()) for k,a in rat.items()}
        rounds.append(dict(status=prob.status,frequencies=len(ids),ratios=peaks));print('CUT',kind,peaks,flush=True)
        if max(peaks.values())<=1+2e-6:break
        for a in rat.values():ix.update(np.argsort(a[TRAIN].max(axis=0))[-12:].tolist())
    dump(out/f'{kind}_{name}_solve.json',dict(name=name,kind=kind,theta=val.tolist(),allowed=d['allowed'],rates=d['rates'],grid_weight=wg,control_weight=wu,rounds=rounds))
    if evaluate_result:print('PERFORMANCE',evaluate_controller(d,val,name,out),flush=True)
    return name

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('--grid-weight',type=float,default=3.);p.add_argument('--control-weight',type=float,default=20.)
    a=p.parse_args();solve(a.kind,a.grid_weight,a.control_weight)
