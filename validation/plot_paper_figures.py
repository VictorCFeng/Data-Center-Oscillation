"""Generate F01-F05 from the saved single-site and independent-network results."""
import json
import numpy as np

import figure_style as plot
from trajectory_io import collect

SOURCE = plot.OUT
LOCAL = SOURCE.parent / 'independent_site_design_local_target'

def read(path):
    return json.loads(path.read_text())

def main():
    names = plot.NAMES[:-1] + ['proposed']
    records, data, metrics, frequency_metrics = {}, {}, {}, {}
    local_summary = read(LOCAL / 'summary.json')
    for kind in plot.KINDS:
        all_records = {}
        for folder in [SOURCE, SOURCE.parent/'bilateral_power_only_verification']:
            all_records.update(collect(folder, kind)[0])
        if kind == 'ieee9_two_dc':
            all_records.update(collect(LOCAL, kind)[0])
        base_rows = read(SOURCE/f'{kind}_uniform_time_metrics.json')['rows']
        base_metrics = {r['name']:r for r in read(SOURCE/f'{kind}_uniform_frequency_metrics.json')}
        selected, rows, fa = {}, [], {}
        for label in names + ['lead_support', 'service_reference']:
            independent = kind == 'ieee9_two_dc' and label == 'proposed'
            actual = ('independent_local_q' if independent else 'centered_q_g3_u20') if label == 'proposed' else label
            folder = LOCAL if independent else SOURCE
            chosen = [r for r in all_records.values() if r['name'] == actual]
            for r in chosen:
                selected[r['key']] = dict(r, name=label)
            if label == 'service_reference':
                continue
            source_rows = local_summary['rows'] if independent else base_rows
            rows += [dict(r, name=label) for r in source_rows if r['name']==actual]
            source_metrics = read(LOCAL/f'{kind}_uniform_frequency_metrics.json')[0] if independent else base_metrics[actual]
            fa[label] = dict(source_metrics, name=label)
            if label in names:
                data[kind,label] = dict(np.load(folder/f'{kind}_{actual}_uniform_frequency.npz'))
        records[kind],metrics[kind],frequency_metrics[kind] = selected,rows,fa
        print(kind,'plotted trajectories',len(selected),flush=True)
    plot.figures(records,data,metrics,frequency_metrics,names=names,figdir=LOCAL/'paper_figures')

if __name__=='__main__':
    main()
