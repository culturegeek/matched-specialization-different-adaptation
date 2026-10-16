import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
arr=np.stack([np.load(f)['values'].reshape(2,2,500,4,28) for f in sorted(a.input.glob('seed_*/records.npz'))]);fields=list(np.load(next(a.input.glob('seed_*/records.npz')))['fields']);ix={f:i for i,f in enumerate(fields)}
font=next(Path('/System/Library/Fonts').glob('ヒラ*角*W3.ttc'),None)
if font:fontManager.addfont(str(font))
for lang in ['en','ja']:
 plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name() if lang=='ja' and font else 'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42 if lang=='en' else 3})
 fig,ax=plt.subplots(2,2,figsize=(8,5.8),layout='constrained');colors=['#2166ac','#d6604d'];labels=['Acquisition route','Update route'] if lang=='en' else ['取得経路','更新経路']
 for li,n in [(0,224),(1,66)]:
  for ri in [0,1]:
   v=arr[:,:,ri,:,li,ix['v_after']].mean(axis=(0,1))/n
   cancel=(-arr[:,:,ri,:,li,ix['v_cross']]/arr[:,:,ri,:,li,ix['v_self']]).mean(axis=(0,1))
   ax[0,li].plot(np.arange(1,501),v,c=colors[ri],label=labels[ri]);ax[1,li].plot(np.arange(1,501),cancel,c=colors[ri])
  ancestor=arr[:,:,0,:,li,ix['v_initial']].mean(axis=(0,1))/n
  ax[0,li].plot(np.arange(1,501),ancestor,'--',c='#888888',lw=1,label='Retained ancestor' if lang=='en' else '初期状態からの残存分')
  ax[0,li].set_yscale('log');ax[0,li].set_title(('(a)' if li==0 else '(b)')+(' Encoder layer ' if lang=='en' else ' 共有表現の第')+str(li+1)+('' if lang=='en' else '層'))
  ax[1,li].set_title(('(c)' if li==0 else '(d)')+(' Accumulated cancellation' if lang=='en' else ' 蓄積した相殺の割合'));ax[1,li].set_ylim(0,1)
  for aa in ax[:,li]:aa.set_xlabel('Formation updates' if lang=='en' else '形成中の更新回数');aa.grid(alpha=.15)
 ax[0,0].set_ylabel('Mean second moment per parameter' if lang=='en' else 'パラメータ当たりの二次モーメント')
 ax[1,0].set_ylabel('Negative cross term / head-square sum' if lang=='en' else '負の交差項／個別二乗項の合計')
 ax[0,1].legend(loc='upper left',fontsize=7)
 fig.savefig(a.out/f'formation_logging_{lang}.pdf');fig.savefig(a.out/f'formation_logging_{lang}.png',dpi=160);plt.close(fig)
