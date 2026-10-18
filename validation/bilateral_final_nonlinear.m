function bilateral_final_nonlinear(action,kind,shard,nshards,experiment)
% New bilateral tests; unchanged upstream Zhao device equations.
% Every controller uses realized clipped power in d=p+u and its predictor.
maxNumCompThreads(1);
if nargin<2,kind='';end
if nargin<3,shard=1;end
if nargin<4,nshards=1;end
if nargin<5,experiment='bilateral_design';end
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'validation'));
base=fullfile(root,'validation','results','zhao_grid_cases');out=fullfile(base,experiment);
api=zhao_grid_cases('api');
if strcmp(action,'export')
    orig=load(fullfile(base,'nonlinear_plants.mat'),'plants');plants=orig.plants;models=struct([]);
    points=[.9 .9;.9 1.1;1.1 .9;1.1 1.1];
    for kk={'single_sm','ieee9_two_dc'}
        for op=1:4
            p=api.make_case(kk{1},points(op,1),points(op,2));[m,p]=api.linearize_case(p);
            m.kind=kk{1};m.op=5+op;m.load_scale=points(op,1);m.line_scale=points(op,2);
            assert(m.residual<1e-7&&max(real(eig(m.A)))<0,'Held-out plant invalid');
            p.validation_op=5+op;plants{end+1}=p;
            if isempty(models),models=m;else,models(end+1)=m;end
        end
    end
    save(fullfile(out,'heldout_models.mat'),'models','-v7');save(fullfile(out,'all_nonlinear_plants.mat'),'plants','-v7');
    fprintf('BILATERAL exported 8 independently initialized held-out plants.\n');return;
end
% An experiment may supply additional equilibria without changing old runs.
plantfile=fullfile(out,'all_nonlinear_plants.mat');
if ~exist(plantfile,'file'),plantfile=fullfile(base,'bilateral_design','all_nonlinear_plants.mat');end
load(plantfile,'plants');
manifest=jsondecode(fileread(fullfile(out,[kind '_jobs.json'])));jobs=manifest.jobs;
jobs=jobs(shard:nshards:end);file=fullfile(out,sprintf('%s_nonlinear_shard%d.mat',kind,shard));
records=struct([]);failures=struct([]);if exist(file,'file'),load(file,'records','failures');end
% Resume after changing the worker count without repeating or losing jobs.
% The current partition is disjoint; old shard ownership is only provenance.
completed={};previous=dir(fullfile(out,[kind '_nonlinear_shard*.mat']));
for ii=1:numel(previous)
    if isempty(regexp(previous(ii).name,[kind '_nonlinear_shard[0-9]+\.mat$'],'once')),continue;end
    saved=load(fullfile(previous(ii).folder,previous(ii).name),'records');
    if ~isempty(saved.records),completed=[completed,{saved.records.key}];end
end
for jj=1:numel(jobs)
    job=jobs(jj);key=job.key;
    if any(strcmp(completed,key)),continue;end
    if job.op<=5
        start=1;if strcmp(kind,'ieee9_two_dc'),start=6;end;p=plants{start+job.op-1};
    else
        ix=find(cellfun(@(q)strcmp(q.kind,kind)&&isfield(q,'validation_op')&&q.validation_op==job.op,plants),1);
        p=plants{ix};
    end
    data=load(fullfile(out,job.controller_file));if iscell(data.designs),d=data.designs{1};else,d=data.designs(1);end
    for fld={'Ac','Bc','Cc','Gu'},d.(fld{1})=sparse(d.(fld{1}));end
    [J,~]=api.jac(@(z)api.plant_dae(z,zeros(p.nin,1),p),p.z0,1e-6);
    p.Jv=J(p.nx+1:end,p.nx+1:end);
    fprintf('BILATERAL %s shard%d [%d/%d] %s\n',kind,shard,jj,numel(jobs),key);
    try
        started=tic;r=simulate(p,d,job,api);fprintf('  Wall time %.3f s\n',toc(started));r.key=key;r.op=job.op;r.target=job.target;r.scene=job.scene;r.phase=job.phase;
        if isempty(records),records=r;else,records(end+1)=r;end
        if ~isempty(failures),failures=failures(~strcmp({failures.key},key));end
    catch err
        failure=struct('key',key,'message',err.message);
        if isempty(failures),failures=failure;else,failures(end+1)=failure;end
        fprintf('FAILED %s\n',err.message);
    end
    pending=[file '.pending.mat'];save(pending,'records','failures','-v7');movefile(pending,file,'f');
end
fprintf('BILATERAL FINISHED %s shard%d: %d complete, %d failed\n',kind,shard,numel(records),numel(failures));
end

function [w,g]=inputs(t,p,job)
w=zeros(p.nd,1);g=0;
if strcmp(job.scene,'overlap')
    if t>=2&&t<18
        env=min(1,(t-2)/.3)*min(1,(18-t)/.3);
        if ~strcmp(job.event,'grid')
            freq=2;if isfield(job,'frequency'),freq=job.frequency;end
            for i=1:p.nd,w(i)=.01*env*sin(2*pi*freq*(t-2)+(i-1)*pi/2);end
        end
        freq=2;if isfield(job,'frequency'),freq=job.frequency;end
        if ~strcmp(job.event,'server'),g=.01*env*sin(2*pi*freq*(t-2)+job.phase);end
    end
elseif strcmp(job.scene,'gridpulse')
    if t>=2&&t<2.2,g=.05;end
else
    if ~strcmp(job.event,'grid')&&t>=2&&t<26
        tau=t-2;env=min(1,tau/.2)*min(1,(26-t)/.2);
        for i=1:p.nd,w(i)=env*(.025*sin(2*pi*(2*tau+.20*tau^2)+(i-1)*pi/3)+.012*sin(2*pi*13*tau));end
    end
    if ~strcmp(job.event,'server')&&t>=5,g=.05;end
    if strcmp(job.scene,'stress'),w=5*w;g=3*g;end
end
end

function [dz,y,u,w,g,kcl,clip]=rhs(t,z,p,d,job,api)
xc=z(p.nx+1:end);raw=d.Cc*xc;u=max(-.2,min(.2,raw));clip=max(abs(raw-u));
[w,g]=inputs(t,p,job);v=api.algebraic_voltage(z(1:p.nx),u,g,p);
[F,ya]=api.plant_dae([z(1:p.nx);v],[w;g;u],p);y=ya-p.y0;
dx=d.Ac*xc+d.Bc*y(1:3*p.nd)+d.Gu*(u-raw);
dz=[F(1:p.nx);dx];kcl=norm(F(p.nx+1:end),inf);
end

function rec=simulate(p,d,job,api)
opts=odeset('RelTol',2e-7,'AbsTol',2e-9,'MaxStep',.01,'Stats','on');
edges=[0 2 5 26 30];if strcmp(job.scene,'overlap'),edges=[0 2 18 22];end
if strcmp(job.scene,'gridpulse'),edges=[0 2 2.2 10];end
x=[p.x0;zeros(size(d.Ac,1),1)];tt=[];xx=[];
for j=1:numel(edges)-1
    ts=(edges(j):.004:edges(j+1))';
    % Resolve the fast response immediately after the grid event.
    if isfield(job,'transient_dt')&&edges(j)==5
        ts=unique(round([ts;(5:job.transient_dt:5.25)'],10));
    end
    [t,z]=ode15s(@(t,z)rhs(t,z,p,d,job,api),ts,x,opts);
    assert(abs(t(end)-ts(end))<1e-9,'Incomplete trajectory');x=z(end,:)';
    if j>1,t=t(2:end);z=z(2:end,:);end
    tt=[tt;t];xx=[xx;z];
end
Y=zeros(numel(tt),numel(p.y0));U=zeros(numel(tt),p.nd);W=U;G=zeros(numel(tt),1);kr=0;cr=0;
for j=1:numel(tt)
    [~,y,u,w,g,k,c]=rhs(tt(j),xx(j,:)',p,d,job,api);
    Y(j,:)=y';U(j,:)=u';W(j,:)=w';G(j)=g;kr=max(kr,k);cr=max(cr,c);
end
assert(all(isfinite(Y(:)))&&all(isfinite(U(:))),'Nonfinite result');
rec=struct('kind',p.kind,'name',d.name,'event',job.event,'t',tt,'y',Y,'u',U,'w',W,'g',G,'kcl_max',kr,'clipping_max',cr);
fprintf('  Completed %.0fs, KCL %.2g, peak u %.4g MW, min f %.5g Hz\n',tt(end),kr,max(abs(U(:)))*100,min(Y(:,end-1)));
end
