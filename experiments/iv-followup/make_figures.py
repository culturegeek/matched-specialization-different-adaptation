"""Static scientific figures from seed-clustered CSV summaries."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
def read(p):return list(csv.DictReader(p.open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.bbox':'tight'})
levels=read(ROOT/'results_confirm/levels.csv')
effects=read(ROOT/'results_confirm/route_effects.csv')
old=read(ROOT/'existing_analysis/seed_predictors.csv')
new=read(ROOT/'results_confirm/seed_predictors.csv')

fig,axes=plt.subplots(1,2,figsize=(10,4.0))
for rows,label,color in [(old,'Original 40 seeds','#9882b4'),(new,'New 40 seeds','#187d7a')]:
    axes[0].scatter([float(r['q_ratio']) for r in rows],[float(r['response_ratio']) for r in rows],s=22,alpha=.75,label=label,color=color)
axes[0].plot([1.5,2.5],[1.5,2.5],'k--',lw=1,label='Response ratio = scaling ratio')
axes[0].set(xlabel='Acquisition / update: mean inverse scaling',ylabel='Acquisition / update: ten-step response')
axes[0].legend(fontsize=8)
modes=['own','m_other','v_other','v_equal_scale','scalar_v_own','v_other_equal_scale','scalar_v_equal','reset','sgd_base','sgd_calibrated']
labels=['Carried Adam','m swap','v swap','Scale equalized','Scalar v / own scale','v swap + equal scale','Scalar v + equal scale','Fresh Adam','SGD, lr .002','SGD, calibrated']
for i,m in enumerate(modes):
    r=next(r for r in effects if r['mode']==m and r['trajectory']=='continued_opposition' and r['step']=='10')
    v,lo,hi=[float(r[k]) for k in ['mean','low','high']]
    axes[1].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#174a6e',capsize=2)
axes[1].axvspan(-.02,.02,color='#eeeeee');axes[1].axvline(0,color='gray',lw=.7)
axes[1].set(yticks=range(len(labels)),yticklabels=labels,xlabel='Ten-step response: acquisition minus update')
axes[1].invert_yaxis();fig.tight_layout()
fig.savefig(OUT/'scale_and_prediction.pdf');fig.savefig(OUT/'scale_and_prediction.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(13.2,3.5))
for ax,tr,title in [(axes[0],'continued_opposition','Continued opposing conditions'),(axes[1],'pulse_then_neutral','Ten-step pulse, then balanced')]:
    for mode,ls in [('own','-'),('reset','--')]:
        for route,color in [('acquisition','#b55a30'),('update','#6750a4')]:
            rr=sorted([r for r in levels if r['trajectory']==tr and r['mode']==mode and r['route']==route],key=lambda r:int(r['step']))
            x=[int(r['step']) for r in rr];y=[float(r['mean']) for r in rr]
            ax.plot(x,y,color=color,ls=ls,label=route+' / '+('carried' if mode=='own' else 'reset'))
            ax.fill_between(x,[float(r['low']) for r in rr],[float(r['high']) for r in rr],color=color,alpha=.10)
    ax.axhline(0,color='gray',lw=.7);ax.axvline(10,color='gray',lw=.6,ls=':')
    ax.set(xscale='symlog',xlabel='Updates from pulse onset',ylabel='Allocation index (initial favored target positive)',title=title)
    ax.set_xlim(0,510)
    ax.set_xticks([0,1,10,100,500],labels=['0','1','10','100','500']);ax.legend(fontsize=6.7)
ax=axes[2]
for mode,color,label in [('own','#174a6e','Carried Adam'),('reset','#b55a30','Fresh Adam'),('sgd_calibrated','#397f65','Calibrated SGD')]:
    rr=sorted([r for r in effects if r['trajectory']=='continued_opposition' and r['mode']==mode],key=lambda r:int(r['step']))
    x=[int(r['step']) for r in rr];y=[float(r['mean']) for r in rr]
    ax.plot(x,y,color=color,marker='o',ms=3,lw=1.3,label=label)
    ax.fill_between(x,[float(r['low']) for r in rr],[float(r['high']) for r in rr],color=color,alpha=.10)
ax.axhspan(-.02,.02,color='#eeeeee',zorder=0);ax.axhline(0,color='gray',lw=.7)
ax.set(xscale='symlog',xlabel='Updates from pulse onset',ylabel='Route difference in specialization loss',title='When the route difference appears')
ax.set_xlim(0,500);ax.set_xticks([0,1,10,20,50,100,500],labels=['0','1','10','20','50','100','500']);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'time_windows.pdf');fig.savefig(OUT/'time_windows.png',dpi=170);plt.close(fig)
