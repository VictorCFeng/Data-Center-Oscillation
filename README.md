# Reproducing the case studies

This package reproduces the numerical case studies and Figs. F01–F05 for *Data Center Oscillation Isolation and Grid Support: Fundamental Limits and Source-Selective Control*. It contains two systems: a data center connected to a finite synchronous-machine grid (`single_sm`), and two data centers with independently designed local controllers in a modified IEEE 9-bus network (`ieee9_two_dc`). Both cases use the data-center and grid-device equations released by Zhao and Zhao; the network configurations and controllers used here are documented adaptations, not unchanged upstream test cases.

## Requirements

- MATLAB with `ode15s` and the functions used by the included model files. The reported nonlinear runs used MATLAB R2023b.
- Python 3.11 with the versions in `requirements.txt` for controller calculations, frequency responses, and figures. Install them with `python -m pip install -r requirements.txt` in a dedicated environment.
- Times New Roman for the supplied figure style. Run Python commands from the package root, the directory containing this README. On a headless machine, set `MPLBACKEND=Agg`; limiting numerical-library threads may help with performance.

The nonlinear runs are computationally substantial. Frozen plant models, controller realizations, job manifests, trajectories, and final figures are included so that figures can be regenerated without repeating the integrations.

The package requires no LaTeX source, PDF, or TeX installation. The 10 Python modules contain the final controller design, frequency-response calculations, trajectory processing, and plotting functions; unrelated exploratory workflows are excluded.

## Package contents

| Path | Purpose |
| --- | --- |
| `external/Data-Center-Analytical-Dynamic-Model/` | Upstream Zhao–Zhao source used by the model adapter; see its README for the original paper and repository. The source revision is `d7260b49`. |
| `validation/zhao_grid_cases.m` | Builds the modified single-generator and IEEE 9-bus cases, operating points, and model exports. |
| `validation/export_same_order_robust.m` | Exports the four corner operating points 10–13 used for coefficient fitting. |
| `validation/bilateral_final_nonlinear.m` | Runs the nonlinear simulations specified by the job manifests. |
| `validation/controller_design.py`, `validation/independent_site_design.py` | Synthesizes the single-site and independent local controllers. |
| `validation/case_metrics.py`, `validation/plot_paper_figures.py` | Evaluates responses and produces the paper figures. |
| `validation/results/zhao_grid_cases/` | Frozen models, controllers, manifests, trajectories, metrics, and figure outputs. |

The literature comparator implements an **adaptation of the rule-based controller (RBC)** in You *et al.*, “Source side mitigation of AI datacenter power fluctuations with a hybrid energy storage system and residual differentiable predictive control,” arXiv:2606.04869 (2026). It does not reproduce that paper's learned predictive controller. The `RBC + support` variant adds a grid-support branch in the present study. The lead-compensation-plus-support comparator is an engineering baseline; its optimized variant uses a matched controller structure with tuned coefficients. These method distinctions are retained in the manifests and figure labels.

## Generate F01–F05 from the included results

From the package root, run:

```sh
python validation/case_metrics.py frequency single_sm
python validation/case_metrics.py frequency ieee9_two_dc
python validation/case_metrics.py time single_sm
python validation/case_metrics.py time ieee9_two_dc
python validation/independent_site_design.py finish
python validation/plot_paper_figures.py
```

The final PDF, PNG, and SVG files are written to `validation/results/zhao_grid_cases/independent_site_design_local_target/paper_figures/`. The first two commands regenerate the `*_uniform_frequency.npz` response files that the figure script reads.

## Repeat the nonlinear simulations

The MATLAB runner resumes jobs already present in its shard files. To integrate all trajectories anew, work in a separate copy of the package with the existing `*_nonlinear_shard*.mat` output files removed from that copy; keep the frozen plant, controller, and `*_jobs.json` input files. From the package root in MATLAB, run:

```matlab
addpath('validation');
bilateral_final_nonlinear('run','single_sm',1,1,'bilateral_final_comparison');
bilateral_final_nonlinear('run','ieee9_two_dc',1,1,'bilateral_final_comparison');
bilateral_final_nonlinear('run','single_sm',1,1,'bilateral_power_only_verification');
bilateral_final_nonlinear('run','ieee9_two_dc',1,1,'bilateral_power_only_verification');
bilateral_final_nonlinear('run','ieee9_two_dc',1,1,'independent_site_design_local_target');
```

The third and fourth arguments are shard index and shard count; `1,1` runs every job in each listed experiment. After MATLAB finishes, run the Python commands above to recalculate metrics and figures. The manifests specify 286 trajectories per system, for 572 in total. The network controllers are designed independently at each data center.

## Repeat controller synthesis

The included controller files are the fixed inputs for reproducing the figures. To solve for their coefficients again, use a separate copy because synthesis writes controller files and performance results:

```sh
python validation/controller_design.py single_sm
python validation/independent_site_design.py design --site 0
python validation/independent_site_design.py design --site 1
python validation/independent_site_design.py assemble
```

Optimizer versions and tolerances may change coefficient values slightly; compare closed-loop responses and constraints, not only file hashes. The reported results cover the specified operating points and disturbances, not every possible grid condition or storage state.
