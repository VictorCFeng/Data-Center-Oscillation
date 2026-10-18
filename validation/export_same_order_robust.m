function export_same_order_robust()
% Additional boundary operating points, using unchanged Zhao device equations.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'validation'));
base=fullfile(root,'validation','results','zhao_grid_cases');out=fullfile(base,'same_order_robust');
if ~exist(out,'dir'),mkdir(out);end
api=zhao_grid_cases('api');
load(fullfile(base,'bilateral_design','all_nonlinear_plants.mat'),'plants');
models=struct([]);points=[.8 .8;.8 1.2;1.2 .8;1.2 1.2];
for kk={'single_sm','ieee9_two_dc'}
    for i=1:4
        p=api.make_case(kk{1},points(i,1),points(i,2));[m,p]=api.linearize_case(p);
        m.kind=kk{1};m.op=9+i;m.load_scale=points(i,1);m.line_scale=points(i,2);
        assert(m.residual<1e-7&&max(real(eig(m.A)))<0,'Invalid corner equilibrium');
        p.validation_op=m.op;plants{end+1}=p;
        if isempty(models),models=m;else,models(end+1)=m;end
        fprintf('CORNER %s op%d load=%.1f line=%.1f residual=%.3g maxRe=%.5g\n',m.kind,m.op,m.load_scale,m.line_scale,m.residual,max(real(eig(m.A))));
    end
end
save(fullfile(out,'corner_models.mat'),'models','-v7');
save(fullfile(out,'all_nonlinear_plants.mat'),'plants','-v7');
end
