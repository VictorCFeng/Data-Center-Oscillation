# Case-study reproduction

Code and results for *Data Center Oscillation Isolation and Grid Support: Fundamental Limits and Source-Selective Control* ([arXiv:2610.26812](https://arxiv.org/abs/2610.26812)). The cases are a single-generator system (`single_sm`) and a modified IEEE 9-bus network with two independently controlled data centers (`ieee9_two_dc`).

## Model and benchmark sources

The data-center and grid-device equations are from:

> X. Zhao and J. Zhao, “Dynamic Modeling of Data-Center Power Delivery for Power System Resonance Analysis,” arXiv:2604.06624, 2026. [Paper](https://arxiv.org/abs/2604.06624) · [GitHub repository](https://github.com/xingyuz-phd/Data-Center-Analytical-Dynamic-Model)

The included source is in `external/Data-Center-Analytical-Dynamic-Model/` (revision `d7260b49`). This study modifies the network configurations and adds storage controllers; these are not unchanged upstream cases.

The RBC benchmark adapts the rule-based controller in H. You, C. Lou, and J. Yang, “[Source Side Mitigation of AI Datacenter Power Fluctuations with a Hybrid Energy Storage System and Residual Differentiable Predictive Control](https://arxiv.org/abs/2606.04869),” 2026, not its learned predictive controller. `RBC + support` adds our frequency-support branch. `Lead + support` is an engineering baseline; its optimized variant retunes the same structure.

## Environment

- MATLAB R2023b: nonlinear simulations using `ode15s`, without Simulink.
- Python 3.11: controller optimization, frequency responses, and plotting; dependencies are listed in `requirements.txt`.
- Times New Roman: figure font.

Run commands from this directory. Models, controllers, and saved trajectories are included under `validation/results/zhao_grid_cases/`; no LaTeX files or installation are required.

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

PDF, PNG, and SVG outputs: `validation/results/zhao_grid_cases/independent_site_design_local_target/paper_figures/`.

## Repeat the nonlinear simulations

Completed trajectories are skipped. For a fresh run, use a separate package copy and remove only its `*_nonlinear_shard*.mat` outputs, retaining model, controller, and `*_jobs.json` inputs. Run in MATLAB:

```matlab
addpath('validation');
bilateral_final_nonlinear('run','single_sm',1,1,'bilateral_final_comparison');
bilateral_final_nonlinear('run','ieee9_two_dc',1,1,'bilateral_final_comparison');
bilateral_final_nonlinear('run','single_sm',1,1,'bilateral_power_only_verification');
bilateral_final_nonlinear('run','ieee9_two_dc',1,1,'bilateral_power_only_verification');
bilateral_final_nonlinear('run','ieee9_two_dc',1,1,'independent_site_design_local_target');
```

These commands run all 572 trajectories (286 per system). Then repeat the figure-generation commands above.

## Repeat controller synthesis

Use a separate package copy: synthesis overwrites controller outputs.

```sh
python validation/controller_design.py single_sm
python validation/independent_site_design.py design --site 0
python validation/independent_site_design.py design --site 1
python validation/independent_site_design.py assemble
```

After redesign, rerun the simulations with the new controllers before regenerating figures.
