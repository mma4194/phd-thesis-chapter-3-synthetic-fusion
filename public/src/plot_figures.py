"""Regenerate Figures 2 and 3. Figure 3 uses recorded case-study results."""
from pathlib import Path
import argparse,csv,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',default='runs/figures')
p.add_argument('--controlled-dir',help='Optional fresh controlled-run directory')
a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
controlled_dir=Path(a.controlled_dir) if a.controlled_dir else root/'reference/controlled'
blue='#003FA8';dark='#000000'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'text.color':dark,'axes.labelcolor':dark,'xtick.color':dark,'ytick.color':dark,'axes.edgecolor':'#8C939C','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
# Controlled benchmark: intervals are per fixed parameter (not a pooled mixture).
rows=list(csv.DictReader(open(controlled_dir/'decision_by_setting.csv')))
fig,ax=plt.subplots(figsize=(6.7,2.6));fig.subplots_adjust(left=.105,right=.98,bottom=.22,top=.86)
plot_rows=[]
def wilson(k,n):
 z=1.95996398454;p=k/n;d=1+z*z/n;mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return mid-half,mid+half
for cval,label,color,marker,offset in [(.05,'Identical process (c = 0.05)',blue,'o',-.04),(.045,'Within tolerance (c = 0.045)','#5F6873','s',.04)]:
 rr=[next(x for x in rows if x['method']=='framework' and int(x['features'])==m and float(x['candidate_c'])==cval) for m in [2,8,32]]
 k=np.array([int(x['withheld']) for x in rr]);n=100; yy=k/100
 bounds=np.array([wilson(int(i),100) for i in k]);xx=np.arange(3)+offset
 ax.errorbar(xx,yy,yerr=np.vstack([yy-bounds[:,0],bounds[:,1]-yy]),color=color,marker=marker,capsize=3,lw=1.2,markersize=5,label=label)
 for i,v in enumerate(k):
  dx=(-5 if i==2 else 15 if i==0 else 5); dy=((14 if cval==.05 else -5) if i==0 else 8 if cval==.045 else -16)
  ax.annotate(f'{v}/100',(xx[i],yy[i]),xytext=(dx,dy),ha=('right' if i==2 else 'left'),textcoords='offset points',fontsize=8,color=dark)
  plot_rows.append({'candidate_c':cval,'features':[2,8,32][i],'withheld':int(v),'trials':100,'wilson_low':bounds[i,0],'wilson_high':bounds[i,1]})
ax.set_xticks(range(3),['2','8','32']);ax.set_xlabel('Required feature scope');ax.set_ylabel('Unnecessary withholding');ax.set_ylim(-.01,.40);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.grid(axis='y',color='#DDE2E8',lw=.5);ax.set_axisbelow(True)
ax.legend(loc='upper left',frameon=False,fontsize=8)
fig.suptitle('Acceptable controls: more checks increase withholding',x=.105,ha='left',fontsize=10,color=dark)
fig.savefig(out/'fig02_controlled_decisions.pdf');plt.close(fig)
with open(out/'fig02_data.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(plot_rows[0]));w.writeheader();w.writerows(plot_rows)
# Saved residential and Smart* values, read from the supplied results macros.
values=json.loads((root/'reference/paper_plot_values.json').read_text())
def macro(name):return float(values[name])
fig,axs=plt.subplots(1,2,figsize=(6.9,2.8),gridspec_kw={'width_ratios':[1,1.35]});fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.82,wspace=.38)
rr=[macro('QSixSeqRealUnionRate'),macro('QSixSeqRealHoldoutUnionRate'),macro('QSixSeqSyntheticUnionRate')]
ax=axs[0];ax.bar(range(3),rr,color=['#BCC4CE','#DFE3E8',blue],width=.62,edgecolor='#707986',linewidth=.5)
ax.set_xticks(range(3),['Full real','Real query\nhalf','Synthetic']);ax.get_xticklabels()[1].set_color(dark);ax.set_ylabel('Active-window rate');ax.set_ylim(0,.74);ax.set_title('(a) Windows with events',loc='left',pad=12,color=dark)
for i,v in enumerate(rr):ax.text(i,v+.024,f'{v:.3f}',ha='center',fontsize=8,color=dark)
ax.grid(axis='y',color='#E1E5EA',lw=.5);ax.set_axisbelow(True)
roles=['Continuous','Binary','Drivers','Observability'];counts=np.array([[macro('SmartStarContinuous'+g),macro('SmartStarBinary'+g),macro('SmartStarDriver'+g),macro('SmartStarObs'+g)] for g in ['Pass','Warning','Fatal']]).astype(int)
ax=axs[1];left=np.zeros(4)
for vals,label,color in zip(counts,['Pass','Warning','Fatal'],['#EDF0F4','#D8DEE7','#C1CDDE']):
 ax.barh(range(4),vals,left=left,height=.62,label=label,color=color,edgecolor='white',linewidth=.6)
 for i,(v,l) in enumerate(zip(vals,left)):
  if v>=9:ax.text(l+v/2,i,str(v),ha='center',va='center',fontsize=8,color=dark)
 left+=vals
ax.set_yticks(range(4),roles);ax.invert_yaxis();ax.set_xlabel('Audited features');ax.set_xlim(0,110);ax.set_title('(b) Smart* role grades',loc='left',pad=12)
for i,v in enumerate(left):ax.text(v+2,i,f'n={int(v)}',va='center',fontsize=7.4,color=dark)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=3,frameon=False,fontsize=8,handlelength=1.2,columnspacing=1)
fig.savefig(out/'fig03_empirical_quality.pdf');plt.close(fig)
with open(out/'fig03_data.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['panel','group','metric','value','source'])
 for name,value,key in zip(['Full real','Real holdout','Synthetic'],rr,['QSixSeqRealUnionRate','QSixSeqRealHoldoutUnionRate','QSixSeqSyntheticUnionRate']):w.writerow(['a',name,'active_union_rate',value,'results.tex:'+key])
 for i,role in enumerate(roles):
  for j,g in enumerate(['Pass','Warning','Fatal']):w.writerow(['b',role,g,counts[j,i],'results.tex:SmartStar'+('Obs' if role=='Observability' else role.rstrip('s') if role=='Drivers' else role)+g])
