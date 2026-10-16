from pathlib import Path
import argparse,json
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
r=json.loads((a.root/'results/result.json').read_text())
f=pd.concat([pd.read_csv(x) for x in sorted((a.root/'runs').glob('seed_*/formation_trace.csv'))])
f.groupby(['condition','step']).v_encoder.mean().to_csv(a.out/'projection_curve.csv')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fonts=list(Path('/usr/local/texlive').glob('*/texmf-dist/fonts/opentype/public/haranoaji/HaranoAjiGothic-Regular.otf'))
for font in fonts:font_manager.fontManager.addfont(str(font))
plt.rcParams['axes.unicode_minus']=False
for lang in ['en','ja']:
 if lang=='ja' and fonts:plt.rcParams['font.family']=font_manager.FontProperties(fname=fonts[0]).get_name()
 else:plt.rcParams['font.family']='DejaVu Sans'
 fig,ax=plt.subplots(1,2,figsize=(8.0,3.05),gridspec_kw={'width_ratios':[1,1]})
 labels=['Original acquisition','Update route','Projected acquisition'] if lang=='en' else ['元の取得経路','更新経路','射影後の取得経路']
 for c,col,l in zip(['acquisition','update','projected'],['#2878B5','#D55E00','#14856F'],labels):
  z=f[f.condition==c].groupby('step').v_encoder.mean();ax[0].plot(z.index,z.values,color=col,label=l,lw=1.7)
 ax[0].set(xlabel='Formation updates' if lang=='en' else '形成更新数',ylabel='Encoder second-moment sum' if lang=='en' else 'encoder二次モーメント総和',title='(a) Formation' if lang=='en' else '(a) 形成中の変化');ax[0].legend(fontsize=7.8,frameon=False);ax[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
 for j,(donor,col,label) in enumerate(zip(['acquisition','projected'],['#2878B5','#14856F'],['Original $v$','Projected-history $v$'] if lang=='en' else ['元の$v$','射影履歴の$v$'])):
  z=[r['response']['10']['response_'+rc+'_'+donor] for rc in ['acquisition','projected']];x=[i+(j-.5)*.18 for i in range(2)];y=[o['mean'] for o in z]
  ax[1].errorbar(x,y,yerr=[[o['mean']-o['low'] for o in z],[o['high']-o['mean'] for o in z]],fmt='o',color=col,capsize=3,label=label)
 ax[1].set_xticks([0,1],['Original\nrecipient','Projected\nrecipient'] if lang=='en' else ['元の学習器','射影後の学習器']);ax[1].set(xlim=(-.45,1.45),ylim=(.27,.39),ylabel='Specialization loss, 10 updates' if lang=='en' else '10更新での特化減少',title='(b) Only donor $v$ changes' if lang=='en' else '(b) 供与する$v$だけを変更');ax[1].legend(fontsize=7.8,frameon=False,loc='upper right')
 fig.tight_layout(w_pad=2.1);fig.savefig(a.out/('projection_'+lang+'.pdf'));fig.savefig(a.out/('projection_'+lang+'.png'),dpi=180);plt.close(fig)
