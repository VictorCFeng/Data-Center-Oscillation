"""Read completed nonlinear trajectory shards."""

import re

import numpy as np

from scipy.io import loadmat

from data_utils import items

def collect(directory,kind):
    records={};failed=[]
    paths=list(directory.glob(kind+'_nonlinear_shard*.mat'))
    for path in paths:
        if not re.fullmatch(kind+r'_nonlinear_shard\d+\.mat',path.name):continue
        d=loadmat(path,simplify_cells=True)
        if np.size(d['records']):
            for r in items(d['records']):
                assert r['key'] not in records,r['key'];records[r['key']]=r
        if np.size(d['failures']):failed+=items(d['failures'])
    return records,failed
