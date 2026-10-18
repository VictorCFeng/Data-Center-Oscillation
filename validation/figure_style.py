"""Shared ProPlot figure layouts and styles."""
import numpy as np
from matplotlib.figure import Figure

from case_models import BASE, CAPS
from data_utils import plotting

OUT=BASE.parent/'bilateral_final_comparison'
FIG=OUT/'paper_figures'
KINDS=['single_sm','ieee9_two_dc']
NAMES=['uncontrolled','power_only','rbc_adapted','rbc_support_added','lead_support_optimized','centered_q_g3_u20']
LABELS=['No HESS','Power only','RBC (adapted)','RBC + support','Lead + support (opt.)','Proposed']
COLORS=['#777777','#AA4499','#E69F00','#D55E00','#009E73','#0072B2']
STYLES=[':',(0,(1,2)),'--','-.',(0,(5,2,1,2)),'-']

def figures(records, data, metrics, frequency_metrics, names=None, figdir=None):
    names=names or NAMES;figdir=figdir or FIG
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, LogFormatterMathtext, LogLocator
    from design_specs import spec
    import matplotlib as mpl
    pplt=plotting();figdir.mkdir(exist_ok=True)
    # Use Times New Roman consistently across all figure panels.
    from matplotlib import font_manager
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
        'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,
        'font.size':7,'axes.labelsize':7,
        'xtick.labelsize':6,'ytick.labelsize':6,'axes.linewidth':.55,
        'axes.labelpad':1.4,'xtick.major.pad':1.8,'ytick.major.pad':1.8,
        'grid.linewidth':.5,'grid.alpha':.20})
    markers=[None,'o','D','v','s','^']
    def layout(rows=2,grid_time=False):
        gap=.38 if grid_time else .20;ah=.73
        fig,axs=pplt.subplots(nrows=rows,ncols=2,figwidth=3.5,
            figheight=ah*rows+gap*(rows-1)+.86,share=False,tight=False,
            left='.47in',right='.10in',top='.29in',bottom='.57in',
            hspace=f'{gap}in',wspace='.35in')
        for i,ax in enumerate(axs):
            ax.format(grid=True,ticklabelsize=6,labelsize=7)
            ax.text(0,1.025,f'({chr(97+i)})',transform=ax.transAxes,fontsize=6.5,va='bottom')
            if i<2:ax.set_title(['Single data center','IEEE 9-bus'][i],fontsize=7,pad=6)
        return fig,axs
    def line(ax,x,y,i):
        # Sparse markers sample horizontal position, not arc length: rapidly
        # oscillating transients must not accumulate hundreds of markers.
        x=np.asarray(x);u=(np.arange(7)+.2+.09*i)/7
        positions=np.exp(np.log(x[0])+u*np.log(x[-1]/x[0])) if x[0]>0 and x[-1]/x[0]>100 else x[0]+u*(x[-1]-x[0])
        mark=np.unique(np.searchsorted(x,positions)).tolist()
        return ax.plot(x,y,color=COLORS[i],ls=STYLES[i],lw=.8 if i else .7,
            marker=markers[i],markevery=mark,ms=2.7,mew=.65,
            mfc='white',zorder=3+i*.1)
    def target(ax,x,y):
        ax.plot(x,y,color='#202020',ls='none',marker='+',markevery=(.07,.23),
            ms=3.4,mew=.8,zorder=5)
    def save(fig,axs,num,has_target=False,violations=False):
        handles=[Line2D([],[],color=COLORS[i],ls=STYLES[i],marker=markers[i],
            mfc='white',ms=2.7,mew=.65,lw=.8,label=LABELS[i]) for i in range(6)]
        rows=[handles[:3],handles[3:]]
        if has_target:rows[0].append(Line2D([],[],color='black',ls='none',marker='+',ms=3.5,label='Grid target'))
        if violations:rows[0].append(Line2D([],[],color='#333333',ls='none',marker='x',ms=3,label='Violation'))
        for row,y in zip(rows,[.16/fig.get_size_inches()[1],.008/fig.get_size_inches()[1]]):
            Figure.legend(fig,handles=row,loc='lower center',bbox_to_anchor=(.5,y),
                ncol=len(row),frameon=False,prop={'size':5.8},columnspacing=.8,
                handletextpad=.35,handlelength=1.9,borderpad=0)
        for ax in axs:
            if ax.get_yscale()=='log':
                ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
                ax.yaxis.set_major_formatter(LogFormatterMathtext())
            if num==4 and ax.get_xscale()=='log':
                ax.xaxis.set_major_formatter(LogFormatterMathtext())
        for ext in ['pdf','png','svg']:fig.savefig(figdir/f'F{num:02d}.{ext}',dpi=300)
        pplt.close(fig)
    def freq_axes(ax,xlim,ylabel='',bottom=False):
        ax.format(xscale='log',xlim=xlim,ylabel=ylabel)
        if bottom:ax.format(xlabel='Frequency (Hz)')
        else:ax.tick_params(labelbottom=False)
    fig,axs=layout()
    for j,kind in enumerate(KINDS):
        n=j+1
        for i,name in enumerate(names):
            z=data[kind,name];f=z['frequency'];sel=(f>=.1)&(f<=50)
            a=np.max(z['isolation'],axis=(0,2))
            line(axs[j],f[sel],20*np.log10(np.maximum(a[sel],1e-8)),i)
            gain=np.max(np.abs(z['Y'][:,:,-2,:n]),axis=(0,2))/100
            line(axs[j+2],f[sel],gain[sel],i)
        for ax in [axs[j],axs[j+2]]:ax.axvspan(1,20,color='#0072B2',alpha=.075,zorder=0)
        axs[j].plot([1,20],[20*np.log10(.65)]*2,color='#333333',ls=':',lw=.85)
        axs[j].axhline(0,color='#777777',lw=.5,ls=':')
        axs[j].format(ylim=(-80,12),yticks=[-80,-60,-40,-20,0])
        axs[j+2].format(yscale='log')
        freq_axes(axs[j],(.1,50),r'$A$ (dB)' if j==0 else '')
        freq_axes(axs[j+2],(.1,50),'Gain (Hz/MW)' if j==0 else '',True)
    save(fig,axs,1)
    fig,axs=layout()
    for j,kind in enumerate(KINDS):
        n=j+1
        for i,name in enumerate(names):
            z=data[kind,name];f=z['frequency'];sel=(f>=.005)&(f<=20)
            line(axs[j],f[sel],20*np.log10(np.maximum(z['grid_error'].max(axis=0)[sel],1e-8)),i)
            line(axs[j+2],f[sel],np.abs(z['Y'][12,sel,-2,n])/100,i)
        target(axs[j+2],f[sel],np.abs(z['target'][12,sel,-2])/100)
        for lo,hi in spec(kind)['service_bands_Hz']:
            for ax in [axs[j],axs[j+2]]:ax.axvspan(lo,hi,color='#0072B2',alpha=.065,zorder=0)
            axs[j].plot([lo,hi],[20*np.log10(CAPS[kind][0])]*2,color='#333333',ls=':',lw=.85)
        axs[j].format(ylim=(-100,10),yticks=[-100,-50,0]);axs[j+2].format(yscale='log')
        freq_axes(axs[j],(.005,20),r'$E_g$ (dB)' if j==0 else '')
        freq_axes(axs[j+2],(.005,20),'Gain (Hz/MW)' if j==0 else '',True)
    save(fig,axs,2,has_target=True)
    def get(kind,name,event):
        return next(r for r in records[kind].values() if r['name']==name and r['event']==event and r['op']==13)
    fig,axs=layout(3)
    for j,kind in enumerate(KINDS):
        series=[]
        for name in names:
            r=get(kind,name,'server');sel=(r['t']>=12)&(r['t']<=12.6)
            series.append((r['t'][sel],100*r['y'][sel,0]))
        for row,indices in enumerate([range(6),range(1,6),[1,4,5]]):
            ax=axs[2*row+j]
            # Keep identical time windows, progressively removing large traces.
            for i in indices:
                t,y=series[i];line(ax,t,y,i)
            limit=max(np.max(np.abs(series[i][1])) for i in indices)*1.13
            ax.format(xlim=(12,12.6),ylim=(-limit,limit),ylabel=r'$p_1$ (MW)' if j==0 else '',xticks=[12,12.2,12.4,12.6])
            ax.yaxis.set_major_locator(MaxNLocator(3))
            if row<2:
                ax.tick_params(labelbottom=False)
                next_indices=range(1,6) if row==0 else [1,4,5]
                zoom=max(np.max(np.abs(series[i][1])) for i in next_indices)*1.13
                ax.axhspan(-zoom,zoom,color='#0072B2',alpha=.065,zorder=0)
            else:ax.format(xlabel='Time (s)')
    save(fig,axs,3)
    fig,axs=layout(3,True)
    for j,kind in enumerate(KINDS):
        ref=get(kind,'service_reference','grid');t=ref['t']
        for i,name in enumerate(names):
            r=get(kind,name,'grid');sel=(t>=4.5)&(t<=30);early=(t>=5.000099)&(t<=30)
            line(axs[j],t[sel],100*r['y'][sel,0],i)
            error=100*np.abs(r['y'][early,0]-ref['y'][early,0])
            line(axs[j+2],t[early]-5,np.where(error>=1e-7,error,np.nan),i)
            line(axs[j+4],t[sel],r['y'][sel,-2],i)
        target(axs[j],t[sel],100*ref['y'][sel,0]);target(axs[j+4],t[sel],ref['y'][sel,-2])
        for row in [0,2]:
            ax=axs[j+2*row];ax.format(xlim=(4.5,30),xticks=[5,15,25],xlabel='Time (s)')
            ax.yaxis.set_major_locator(MaxNLocator(3))
        axs[j].format(ylabel=r'$p_1$ (MW)' if j==0 else '')
        axs[j+2].format(xscale='log',yscale='log',xlim=(1e-4,25),ylim=(1e-5,10),
            xticks=[1e-4,.01,1],yticks=[1e-5,.001,.1,10],xlabel=r'$t-5$ (s)',ylabel='Error (MW)' if j==0 else '')
        axs[j+4].format(ylabel=r'$\Delta f_{\rm SM}$ (Hz)' if j==0 else '')
    save(fig,axs,4,has_target=True)
    fig,axs=layout()
    for j,kind in enumerate(KINDS):
        rr=metrics[kind]
        fa=frequency_metrics[kind]
        for i,name in enumerate(names):
            a=sorted([r for r in rr if r['name']==name and r['event']=='combined'],key=lambda r:r['op'])
            x=np.array([r['op'] for r in a]);y=np.array([r['rms_MW'] for r in a]);line(axs[j],x,y,i)
            bad=np.array([not r['pass'] for r in fa[name]['points']])
            axs[j].plot(x[bad],y[bad],ls='none',marker='x',ms=3,mew=.7,color='#333333',zorder=6)
            budget=CAPS[kind][1]*np.sqrt(j+1)
            line(axs[j+2],x,[r['noise_MW']/budget for r in fa[name]['points']],i)
        for ax in [axs[j],axs[j+2]]:
            ax.axvspan(5.5,9.5,color='#0072B2',alpha=.075,zorder=0)
            ax.format(xlim=(.5,13.5),xticks=[1,5,9,13])
        axs[j].format(yscale='log',ylabel='Target RMS (MW)' if j==0 else '');axs[j].tick_params(labelbottom=False)
        axs[j+2].axhline(1,color='#333333',ls=':',lw=.85)
        axs[j+2].format(ylim=(-.05,1.13),yticks=[0,.5,1],ylabel='Meas. gain / limit' if j==0 else '',xlabel='Operating point')
    save(fig,axs,5,violations=True)
