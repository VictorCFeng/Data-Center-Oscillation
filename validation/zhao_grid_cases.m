function api=zhao_grid_cases(action)
% Reproducible finite-grid adapters for Zhao & Zhao, commit d7260b49.
% Original device equations are called without modification. Only topology,
% extra DC ports, a grid-event load and unity-PF HESS injections are added.
if nargin<1, action='export'; end
api=[];
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'external','Data-Center-Analytical-Dynamic-Model','src','case9'));
out=fullfile(root,'validation','results','zhao_grid_cases');
if ~exist(out,'dir'), mkdir(out); end
switch action
case 'api'
    % Device and network equations used by the nonlinear simulation runner.
    api=struct('plant_dae',@plant_dae,'jac',@jac, ...
        'make_case',@make_case,'linearize_case',@linearize_case,'algebraic_voltage',@algebraic_voltage);
case 'export'
    models=cell(0); plants=cell(0); k=0;
    points=[1 1; .8 1; 1.2 1; 1 .8; 1 1.2];
    for kind={'single_sm','ieee9_two_dc'}
        for op=1:size(points,1)
            k=k+1; p=make_case(kind{1},points(op,1),points(op,2));
            [m,p]=linearize_case(p);
            m.kind=kind{1}; m.op=op; m.load_scale=points(op,1); m.line_scale=points(op,2);
            models{k}=m; plants{k}=p;
            fprintf('%s op%d: nx=%d residual=%.3g maxRe=%.6g\n',kind{1},op,m.nx,m.residual,max(real(eig(m.A))));
            assert(m.residual<1e-7,'Equilibrium failed');
            assert(max(real(eig(m.A)))<0,'Baseline unstable: stabilize before IMC');
        end
    end
    models=[models{:}];
    save(fullfile(out,'plant_models.mat'),'models','-v7');
    save(fullfile(out,'nonlinear_plants.mat'),'plants','-v7');
otherwise
    error('Unknown action');
end
end

function p=make_case(kind,ls,zs)
p.kind=kind; p.baseMVA=100; p.fb=60;
if strcmp(kind,'single_sm')
    % SDCIB Thevenin R/X from SDCIB.m, now terminated by the upstream SM.
    p.nb=2; p.dc_bus=2; p.event_bus=1;
    branch=[1 2 .02 .19 0]; typ=[3;1]; Vm=[1.04;1];
    Pd=[0;.5*ls]; Qd=[0;0]; Pg=[0;0]; Qg=[0;0];
else
    % Unchanged IEEE 9-bus branch table from upstream case9.m.
    p.nb=9; p.dc_bus=[5 8]; p.event_bus=6;
    branch=[1 4 0 .0576 0;4 5 .017 .092 .158;5 6 .039 .17 .358;3 6 0 .0586 0;6 7 .0119 .1008 .209;7 8 .0085 .072 .149;8 2 0 .0625 0;8 9 .032 .161 .306;9 4 .01 .085 .176];
    typ=[3;2;1;1;1;1;1;1;1]; Vm=[1.04;1.025;1.025;ones(6,1)];
    Pd=[0;0;0;0;.625;.45;0;.5;0]; Qd=[0;0;0;0;.25;.15;0;0;0];
    Pd(p.dc_bus)=Pd(p.dc_bus)*ls;
    Pg=[0;.815;.425;zeros(6,1)]; Qg=[0;0;-.0547;zeros(6,1)];
end
branch(:,3:4)=branch(:,3:4)*zs;
Y=zeros(p.nb); for b=1:size(branch,1)
    i=branch(b,1); j=branch(b,2); yy=1/complex(branch(b,3),branch(b,4)); sh=1i*branch(b,5)/2;
    Y(i,i)=Y(i,i)+yy+sh; Y(j,j)=Y(j,j)+yy+sh; Y(i,j)=Y(i,j)-yy; Y(j,i)=Y(j,i)-yy;
end
pq=find(typ==1); ns=find(typ~=3); V=Vm;
opt=optimoptions('fsolve','Display','none','FunctionTolerance',1e-13,'StepTolerance',1e-13,'OptimalityTolerance',1e-13,'MaxIterations',300);
pf=@(xx) pf_residual(xx,Y,V,ns,pq,Pg-Pd,Qg-Qd);
[xx,rr,ef]=fsolve(pf,[zeros(numel(ns),1);Vm(pq)],opt);
assert(ef>0 && norm(rr,inf)<1e-8,'Power flow failed: flag %d, residual %g',ef,norm(rr,inf));
va=zeros(p.nb,1); va(ns)=xx(1:numel(ns)); Vm(pq)=xx(numel(ns)+1:end); V=Vm.*exp(1i*va);
S=V.*conj(Y*V); p.V0=V; p.nd=numel(p.dc_bus);
Pn=Pd; Pn(p.dc_bus)=0;
% Remaining original loads are constant impedance, as in upstream code.
% Q at Bus 5 remains a passive shunt, not fictitious DC reactive control.
p.Y=Y+diag(conj(Pn+1i*Qd)./abs(V).^2);
p.sm=sm_params(p.fb);
[xsm,p.sm,p.rsm]=sm_init_from_pf_fsolve(V(1),conj((S(1)+Pd(1)+1i*Qd(1))/V(1)),p.sm);
p.delta0=xsm(1); p.sm_idx=1:8; x=xsm(2:end);
if p.nb==9
    p.gfm=gfm_params(p.fb);
    [xgfm,p.rgfm]=gfm_init_from_pf_qss_newton(V(2),conj(S(2)/V(2)),p.gfm);
    p.gfl=gfl_params(p.fb);
    [xgfl,p.gfl,p.rgfl]=gfl_init_from_pf_qss_newton(V(3),conj((Pg(3)+1i*Qg(3))/V(3)),p.gfl);
    p.gfm_idx=numel(x)+(1:numel(xgfm)); x=[x;xgfm];
    p.gfl_idx=numel(x)+(1:numel(xgfl)); x=[x;xgfl];
end
p.dc=dc_params(p.fb); p.dc_idx=cell(1,p.nd); p.rdc=cell(1,p.nd);
for i=1:p.nd
    bus=p.dc_bus(i);
    [xd,rd]=datacenter_init_from_pf_qss(V(bus),conj(-Pd(bus)/V(bus)),p.dc);
    p.dc_idx{i}=numel(x)+(1:21); p.rdc{i}=rd; x=[x;xd];
end
p.nx=numel(x); p.x0=x; p.z0=[x;real(V);imag(V)]; p.nin=2*p.nd+1;
p.event_scale=abs(V(p.event_bus))^2;
[~,p.y0]=plant_dae(p.z0,zeros(p.nin,1),p);
end

function F=pf_residual(x,Y,V,ns,pq,P,Q)
va=angle(V); vm=abs(V); va(ns)=x(1:numel(ns)); vm(pq)=x(numel(ns)+1:end);
V=vm.*exp(1i*va); S=V.*conj(Y*V); F=[P(ns)-real(S(ns));Q(pq)-imag(S(pq))];
end

function [F,y]=plant_dae(z,inp,p)
x=z(1:p.nx); V=z(p.nx+(1:p.nb))+1i*z(p.nx+p.nb+(1:p.nb));
w=inp(1:p.nd); g=inp(p.nd+1); u=inp(p.nd+2:end);
dx=zeros(p.nx,1); I=zeros(p.nb,1);
xsm=[p.delta0;x(p.sm_idx)]; omega=xsm(2);
[ds,I(1),as]=sm_port_model(0,xsm,V(1),p.sm,p.rsm.pref);
dx(p.sm_idx)=ds(2:end); % Exact removal of fixed reference angle, as upstream.
if p.nb==9
    r=p.rgfm; [dx(p.gfm_idx),I(2)]=gfm_port_model(0,x(p.gfm_idx),V(2),p.gfm,r.pref0,r.qref0,r.ome_ref0,r.v_ref0,omega);
    r=p.rgfl; [dx(p.gfl_idx),I(3)]=gfl_port_model(0,x(p.gfl_idx),V(3),p.gfl,r.pref0,r.qref0,omega);
end
pp=zeros(p.nd,1); ff=pp;
for k=1:p.nd
    bus=p.dc_bus(k); r=p.rdc{k}; r.p_load=r.p_load+w(k); xd=x(p.dc_idx{k});
    [dx(p.dc_idx{k}),Id,aux]=datacenter_port_model(0,xd,V(bus),p.dc,r,omega);
    I(bus)=I(bus)+Id+u(k)/conj(V(bus));
    pp(k)=aux.p_pcc-u(k); ff(k)=p.fb*(p.dc.kp_pll*xd(3)+p.dc.ki_pll*xd(2));
end
I(p.event_bus)=I(p.event_bus)-g*V(p.event_bus)/p.event_scale;
res=I-p.Y*V;
F=[dx;real(res);imag(res)];
y=[pp;abs(V(p.dc_bus));ff;p.fb*(omega-1);as.pe];
end

function [m,p]=linearize_case(p)
z=p.z0; u=zeros(p.nin,1); [f,y]=plant_dae(z,u,p); n=p.nx;
[J,H]=jac(@(zz) plant_dae(zz,u,p),z,1e-6);
[B,E]=jac(@(uu) plant_dae(z,uu,p),u,1e-6);
Fz=J(n+1:end,n+1:end); Fx=J(n+1:end,1:n); Fu=B(n+1:end,:);
Vx=-Fz\Fx; Vu=-Fz\Fu;
m.A=J(1:n,1:n)+J(1:n,n+1:end)*Vx;
m.B=B(1:n,:)+J(1:n,n+1:end)*Vu;
m.C=H(:,1:n)+H(:,n+1:end)*Vx; m.D=E+H(:,n+1:end)*Vu;
m.nx=n; m.nd=p.nd; m.ny=3*p.nd; m.y0=y;
m.residual=norm(f,inf);
m.bus=p.dc_bus; m.load_cmd=cellfun(@(r)r.p_load,p.rdc);
m.poles=eig(m.A); m.algebraic_rcond=rcond(Fz);
p.linear=m;
end

function [J,H]=jac(fun,x,h)
[f,y]=fun(x); J=zeros(numel(f),numel(x)); H=zeros(numel(y),numel(x));
for k=1:numel(x)
    step=h*max(1,abs(x(k))); e=zeros(size(x));e(k)=step;
    [fp,yp]=fun(x+e); [fm,ym]=fun(x-e);
    J(:,k)=(fp-fm)/(2*step); H(:,k)=(yp-ym)/(2*step);
end
end

function v=algebraic_voltage(x,u,g,p)
% At fixed device states all upstream port currents are affine in V.
% Only the added constant-power HESS injection requires Newton iteration.
v0=[real(p.V0);imag(p.V0)];
[ff,~]=plant_dae([x;v0],zeros(p.nin,1),p);
r=ff(p.nx+1:end); J=p.Jv; e=p.event_bus;
J(e,e)=J(e,e)-g/p.event_scale; J(e+p.nb,e+p.nb)=J(e+p.nb,e+p.nb)-g/p.event_scale;
r(e)=r(e)-g*v0(e)/p.event_scale; r(e+p.nb)=r(e+p.nb)-g*v0(e+p.nb)/p.event_scale;
v=v0-J\r;
for it=1:8
    rr=r+J*(v-v0); JJ=J;
    for i=1:p.nd
        k=p.dc_bus(i); a=v(k); b=v(k+p.nb); v2=a*a+b*b;
        ids=[k,k+p.nb]; rr(ids)=rr(ids)+u(i)*[a;b]/v2;
        JJ(ids,ids)=JJ(ids,ids)+u(i)/v2^2*[b*b-a*a,-2*a*b;-2*a*b,a*a-b*b];
    end
    if norm(rr,inf)<1e-11, break; end
    v=v-JJ\rr;
end
assert(norm(rr,inf)<1e-8,'Network algebraic solve failed');
end

% Parameter blocks below reproduce case9.m LOCAL defaults, not the differing
% standalone default_datacenter_params.m in the same upstream directory.
function sm=sm_params(fb)
sm=struct('omega_b',2*pi*fb,'ws',1,'xd',.146,'xq',.0969,'xdp',.0608,'xqp',.0969,'Td0p',8.96,'Tq0p',.31,'H',3,'D',0,'ka',5,'Ta',.2,'Te',.314,'kf',.063,'Tf',.35,'ke',1,'ae',.0039,'be',1.555,'r',.15,'Tsv',.1,'Tch',.5);
end
function gfm=gfm_params(fb)
wb=2*pi*fb; wv=2*pi*80; wi=2*pi*200; cf=.074; lf=.08;
gfm=struct('base',struct('ome_sys',1,'ome_b',wb),'filt',struct('lf',lf,'rf',.003,'cf',cf,'lg',.2,'rg',.01),'droop',struct('Kp',.02,'ome_z',20,'Kq',.05,'ome_f',50),'virt',struct('rv',0,'lv',.2),'inner',struct('kpv',4*wv*cf/wb,'kiv',wv^2*cf/wb,'kpc',1.414*wi*lf/wb,'kic',wi^2*lf/wb),'dc',struct('Vdc',1));
end
function gfl=gfl_params(fb)
gfl=struct('base',struct('baseMVA',100,'fb_hz',fb,'ome_b',2*pi*fb,'ome_sys',1),'filt',struct('lf',.08,'rf',.003,'cf',.074,'lg',.1,'rg',.01),'pll',struct('kp',.05,'ki',1.42,'ome_lp',376.99),'outer',struct('Kp_p',.01,'Ki_p',.12,'Kp_q',.01,'Ki_q',.12,'ome_z',41.47,'ome_f',41.47),'inner',struct('kpc',.15,'kic',.267,'kffv',0,'lf',.08),'dc',struct('Vdc',1));
end
function dc=dc_params(fb)
dc=struct('fb',fb,'omega_b',2*pi*fb,'omega_s',1,'omega_lp',2*pi*100,'l_afe',.05,'r_afe',.003,'c_dc',2,'vdc_ups_ref',1,'l_vsi',.05,'r_vsi',.003,'c_vsi',.2,'vu_vsi_ref',1,'omega_vsi',1,'c_psu',2,'r_psu',.005,'vpsu_ref',1,'c_eq',.2,'veq_ref',.5,'kp_pll',.471,'ki_pll',41.89);
w=2*pi*5;dc.kp_dc_afe=2*w*dc.c_dc/dc.omega_b;dc.ki_dc_afe=w^2*dc.c_dc/dc.omega_b;
w=2*pi*200;dc.kp_c_afe=2*.707*w*dc.l_afe/dc.omega_b-dc.r_afe;dc.ki_c_afe=w^2*dc.l_afe/dc.omega_b;
w=2*pi*100;dc.kp_v_vsi=2*w*dc.c_vsi/dc.omega_b;dc.ki_v_vsi=w^2*dc.c_vsi/dc.omega_b;
w=2*pi*400;dc.kp_c_vsi=2*w*dc.l_vsi/dc.omega_b-dc.r_vsi;dc.ki_c_vsi=w^2*dc.l_vsi/dc.omega_b;
w=2*pi*10;dc.kp_v_psu=2*w*dc.c_psu/dc.omega_b;dc.ki_v_psu=w^2*dc.c_psu/dc.omega_b;
w=2*pi*100;dc.kp_v_eq=2*w*dc.c_eq/dc.omega_b;dc.ki_v_eq=w^2*dc.c_eq/dc.omega_b;
end
