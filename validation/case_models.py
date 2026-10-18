"""Exported plant models, operating points, and feedback interconnections."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.linalg import solve
from data_utils import items

OUT = Path(__file__).resolve().parent / 'results' / 'zhao_grid_cases'
BASE = OUT / 'confirmed_settings_rerun'
CAPS = {'single_sm': (.35, 1.10), 'ieee9_two_dc': (.45, 1.)}
TRAIN = np.array([0,1,2,3,4,9,10,11,12])
START = {'single_sm': 0, 'ieee9_two_dc': 5}

def response(m, freq):
    A, B, C, D = [np.atleast_2d(m[k]) for k in ('A','B','C','D')]
    return np.array([C @ solve(2j*np.pi*f*np.eye(len(A))-A, B) + D for f in freq])

def interconnected(m, ctrl):
    n=int(m['nd']); ny=3*n; Ac,Bc,Cc=ctrl
    A,B,C,D=[m[k] for k in ('A','B','C','D')]
    Bu=B[:,n+1:]; Du=D[:ny,n+1:]
    Aa=np.block([[A,Bu@Cc],[Bc@C[:ny],Ac+Bc@Du@Cc]])
    Bb=np.vstack([B[:,:n+1], Bc@D[:ny,:n+1]])
    Ca=np.block([C,D[:,n+1:]@Cc]); Da=D[:,:n+1]
    return dict(A=Aa,B=Bb,C=Ca,D=Da)

def all_plants(kind):
    old = items(loadmat(OUT/'plant_models.mat', simplify_cells=True)['models'])
    held = items(loadmat(OUT/'bilateral_design'/'heldout_models.mat', simplify_cells=True)['models'])
    return old[START[kind]:START[kind]+5]+[m for m in held if m['kind'] == kind]

def plants(kind):
    corners=items(loadmat(OUT/'same_order_robust'/'corner_models.mat',simplify_cells=True)['models'])
    return all_plants(kind)+[m for m in corners if m['kind']==kind]

def load_case(kind):
    d = np.load(BASE / (kind + '_plants.npz'))
    return plants(kind), d['frequency'], d['H'], d['target']
