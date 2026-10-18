"""MATLAB struct conversion, JSON output, and the ProPlot figure backend."""
import json
import numpy as np

def items(obj):
    if isinstance(obj,dict):return [obj]
    return list(np.atleast_1d(obj))

def ctrl(d,n):
    A=np.atleast_2d(d['Ac']);return A,np.asarray(d['Bc']).reshape(len(A),3*n),np.asarray(d['Cc']).reshape(n,len(A))

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    # proplot 0.9.7 reads the colormap registry removed in matplotlib 3.6.
    if not hasattr(cm,'cmap_d'):cm.cmap_d=cm._colormaps._cmaps
    import proplot as pplt
    pplt.rc.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'lines.linewidth':1.5,'grid.alpha':.15})
    return pplt

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
