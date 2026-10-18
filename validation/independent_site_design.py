"""Independent local controller synthesis and interconnected-system evaluation."""
import argparse
import json
from shutil import copyfile
import numpy as np
from scipy.io import loadmat
from scipy.integrate import cumulative_trapezoid

import controller_design as c
from case_metrics import frequency
from case_models import CAPS, response
from data_utils import ctrl, dump, items
from trajectory_io import collect

KIND = 'ieee9_two_dc'
SOURCE = c.OUT
OUT = SOURCE.parent / 'independent_site_design_local_target'
NAME = 'independent_local_q'

def local_data(full, site):
    rows = [site, 2+site, 4+site]
    cols = [site, 2, 3+site]
    keep = [k for k, (i, j, ell) in enumerate(full['allowed']) if i == site]
    d = dict(full)
    d.update(n=1, H=full['H'][:, :, rows, :][:, :, :, cols],
             target=full['target'][:, :, site:site+1],
             rates=[full['rates'][site]], models=[],
             allowed=[(0, rows.index(full['allowed'][k][1]), full['allowed'][k][2]) for k in keep],
             qm=full['qm'][:, site:site+1, :, :][:, :, rows, :][:, :, :, keep],
             qb=full['qb'][:, site:site+1, rows], xb=full['xb'][keep],
             h0=np.broadcast_to(np.array([-1., 0., 0.])[None, :, None], (len(full['f']), 3, 1)).copy())
    # Independent target from the same frozen per-site service LAW, with the
    # other HESS off. Closing both reference laws yields the unchanged global
    # reference used for final evaluation. No new service gains are tuned.
    ref = items(loadmat(SOURCE/f'{KIND}_service_reference_controller.mat', simplify_cells=True)['designs'])[0]
    aa, bb, cc = ctrl(ref, 2)
    kk = response(dict(A=aa, B=bb, C=cc, D=np.zeros((2,6))), full['f'])
    kl = kk[:,site:site+1,rows]
    hu = d['H'][:,:,:3,2:3]
    ng = np.linalg.solve(1-kl[None]@hu, np.broadcast_to(kl,(len(d['H']),)+kl.shape))
    d['target'] = d['H'][:,:,:1,1] + (d['H'][:,:,:1,2:3]@ng@d['H'][:,:,:3,1:2])[...,0]
    return d, keep

def design(site):
    d, keep = local_data(c.setup(KIND), site)
    dest = OUT / f'bus{[5,8][site]}'
    dest.mkdir(parents=True, exist_ok=True)
    # The 1.414-MW bound is a NETWORK norm, not a 1-MW/site bound. Each
    # independent solve receives that same ceiling; the assembled network is
    # separately checked against the unchanged global ceiling.
    old_caps = CAPS[KIND]
    CAPS[KIND] = (old_caps[0], old_caps[1]*np.sqrt(2))
    try:
        name = c.solve(KIND, data=d, out=dest, evaluate_result=False)
    finally:
        CAPS[KIND] = old_caps
    rec = json.loads((dest/f'{KIND}_{name}_solve.json').read_text())
    rec.update(site=site, bus=[5,8][site], full_coefficient_indices=keep)
    dump(dest/'local_design.json', rec)

def assemble():
    full = c.affine_maps(c.setup(KIND))
    x = np.zeros(len(full['allowed']))
    for site, bus in enumerate([5,8]):
        rec = json.loads((OUT/f'bus{bus}'/'local_design.json').read_text())
        x[rec['full_coefficient_indices']] = rec['theta']
    OUT.mkdir(exist_ok=True)
    performance = c.evaluate_controller(full, x, NAME, out=OUT)
    copyfile(SOURCE/'all_nonlinear_plants.mat', OUT/'all_nonlinear_plants.mat')
    jobs = [dict(job, name=NAME, key=job['key'].replace(job['name'], NAME),
                 target=OUT.name, controller_file=f'{KIND}_{NAME}_controller.mat')
            for job in json.loads((SOURCE/f'{KIND}_jobs.json').read_text())['jobs']
            if job['name'] == 'lead_support_optimized']
    dump(OUT/f'{KIND}_jobs.json', dict(jobs=jobs))
    frequency(KIND, out=OUT, methods=[NAME], labels=['Independent local synthesis'])
    print('ASSEMBLED', performance, flush=True)

def finish():
    records, failed = collect(OUT, KIND)
    baseline, _ = collect(SOURCE, KIND)
    rows = []
    for r in records.values():
        ref = next(v for v in baseline.values() if v['name']=='service_reference' and v['op']==r['op'] and v['event']=='grid')
        t = r['t']
        target = np.zeros((len(t),2)) if r['event']=='server' else ref['y'][:,:2]
        err = 100*(r['y'][:,:2]-target)
        lo, hi = {'server':(2,26),'grid':(5,30),'combined':(2,30)}[r['event']]
        sel = (t>=lo)&(t<=hi)
        rms_site = np.sqrt(np.trapz(err[sel]**2,t[sel],axis=0)/(hi-lo))
        u = 100*np.asarray(r['u']).reshape(len(t),2)
        energy = cumulative_trapezoid(u,t,axis=0,initial=0)/3.6
        rows.append(dict(name=NAME,op=int(r['op']),event=r['event'],rms_MW=float(np.linalg.norm(rms_site)),rms_by_site_MW=rms_site.tolist(),
                         hess_peak_MW=float(abs(u).max()), energy_excursion_kWh=float(np.ptp(energy,axis=0).max()),
                         frequency_drop_Hz=float(max(0.,-r['y'][:,-2].min())),clipping_max=float(r['clipping_max']),kcl_max=float(r['kcl_max'])))
    fa=json.loads((OUT/f'{KIND}_uniform_frequency_metrics.json').read_text())[0]
    summary=json.loads((SOURCE/'summary.json').read_text())['systems'][KIND]
    table=[r for r in summary['table'] if r['name']=='lead_support_optimized']
    table.append(dict(name=NAME,states=17,rms_MW={ev:max(r['rms_MW'] for r in rows if r['event']==ev) for ev in ['server','grid','combined']},
              frequency_pass=fa['pass_count'],stable_points=sum(r['pole_max']<0 for r in fa['points']),
              worst_isolation=max(r['isolation'] for r in fa['points']),worst_grid_error=max(r['grid_error'] for r in fa['points']),
              noise_MW=max(r['noise_MW'] for r in fa['points']),hess_peak_MW=max(r['hess_peak_MW'] for r in rows),
              energy_excursion_kWh=max(r['energy_excursion_kWh'] for r in rows),grid_frequency_drop_Hz=max(r['frequency_drop_Hz'] for r in rows if r['event']=='grid')))
    dump(OUT/'summary.json',dict(table=table,rows=rows,frequency=fa,completed=len(records),
         max_clipping=max(r['clipping_max'] for r in rows),max_kcl=max(r['kcl_max'] for r in rows)))
    print(json.dumps(table,indent=2),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['design','assemble','finish'])
    p.add_argument('--site',type=int,choices=[0,1],default=0);a=p.parse_args()
    if a.action=='design':design(a.site)
    elif a.action=='assemble':assemble()
    else:finish()
