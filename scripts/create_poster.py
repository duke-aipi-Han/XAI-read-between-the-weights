"""Create the 36 by 24 inch research poster from the frozen study results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / 'results/repair_v2_study.json').read_text())
s = {x['session_id']: x for x in r['sessions']}
plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'ps.fonttype':42})
fig = plt.figure(figsize=(36,24), facecolor='white')
navy='#173448'; blue='#287b8c'; coral='#cb653f'; light='#e9f3f4'; ink='#233642'; gray='#52626a'; grid='#dce5e8'
def text(x,y,t,size=25,color=ink,weight='normal',**kw):
    fig.text(x,y,t,fontsize=size,color=color,weight=weight,ha='left',va='top',**kw)
def heading(x,y,t):
    text(x,y,t,34,navy,'bold')
    fig.add_artist(plt.Line2D([x,x+.277],[y-.021,y-.021],transform=fig.transFigure,color=blue,lw=3))
def note(x,y,t,size=20): text(x,y,t,size,gray,linespacing=1.34)

# Header occupies a clean full-width band; all main content has 0.5-inch margins.
fig.add_artist(Rectangle((0,.873),1,.127,transform=fig.transFigure,facecolor=navy,edgecolor='none'))
text(.035,.977,'READ BETWEEN THE WEIGHTS',73,'white','bold')
text(.036,.922,'Can an LLM inspect a neural network and propose a useful weight edit?',32,'#cce9eb')
# Three aligned columns.
l=.035; m=.354; q=.681
heading(l,.845,'1  THE QUESTION')
text(l,.807,'Can we repair a model by inspecting\nand editing its weights directly?',26,ink,'bold',linespacing=1.25)
note(l,.752,'Astra and Qwen act as reviewers: they inspect\na separate, small neural network and each\npropose one targeted change.',22)
heading(l,.674,'THE SMALL NETWORK')
text(l,.639,'Same sign = 1    Different signs = 0',23,ink,'bold')
note(l,.612,'The first two inputs define an XNOR task.\nTwo more inputs are distractions.',21)
# Compact XNOR visual.
ax=fig.add_axes([l+.002,.397,.126,.174]); ax.set_xlim(-1,1); ax.set_ylim(-1,1)
for x,y,label,c in [(-1,0,'0','#f2d6c9'),(0,0,'1','#b9dfe1'),(-1,-1,'1','#b9dfe1'),(0,-1,'0','#f2d6c9')]:
    ax.add_patch(Rectangle((x,y),1,1,facecolor=c,edgecolor='white',linewidth=3));ax.text(x+.5,y+.5,label,ha='center',va='center',fontsize=28,fontweight='bold',color=ink)
ax.axhline(0,color=navy,lw=1);ax.axvline(0,color=navy,lw=1);ax.set_xticks([-0.5,.5],['−','+']);ax.set_yticks([-.5,.5],['−','+']);ax.tick_params(labelsize=19,length=0);ax.set_xlabel('input 1',fontsize=20);ax.set_ylabel('input 2',fontsize=20)
for z in ax.spines.values():z.set_visible(False)
text(l+.155,.555,'4 → 8 → 8 → 1',26,blue,'bold')
note(l+.155,.52,'Small network: 121 parameters\n\nA deliberately weakened\ntrained checkpoint.',20)
heading(l,.354,'THE STARTING WEAKNESS')
note(l,.319,'Accuracy on 4,096 held-out examples:',22)
text(l,.279,'92.33%',47,blue,'bold')
text(l+.135,.270,'overall',22,gray)
note(l,.217,'But when both inputs are negative,\naccuracy falls to 78.2%.',23)
text(l,.164,'REVIEWER MODELS',19,blue,'bold')
note(l,.143,'GPT-6.2 Astra (OpenAI): Nov 2026; size not disclosed.',17)
note(l,.122,'Qwen3.8-27B (Alibaba): Aug 2026; 27B language parameters.',17)
# Center column
heading(m,.845,'2  HOW WE TESTED')
note(m,.807,'Each reviewer received the same weights and\nrepair task in three independent conditions.',22)
text(m,.733,'WEIGHTS ONLY',23,blue,'bold'); note(m+.136,.733,'No model queries.',22)
text(m,.692,'FIXED PROBES',23,blue,'bold'); note(m+.136,.692,'32 preset inputs.',22)
text(m,.651,'ADAPTIVE PROBES',23,blue,'bold'); note(m+.136,.651,'32 inputs in 4 rounds.',22)
text(m,.604,'A probe = an input and the network’s response.',21,gray)
heading(m,.555,'ALL SIX EDITS HELPED')
note(m,.517,'Change in overall accuracy, percentage points',21)
ids=['v2-astra-weights','v2-astra-fixed','v2-astra-adaptive','v2-qwen-weights','v2-qwen-fixed','v2-qwen-adaptive']
vals=[100*s[i]['accuracy_change'] for i in ids]
labels=['Astra  /  weights','Astra  /  fixed','Astra  /  adaptive','Qwen  /  weights','Qwen  /  fixed','Qwen  /  adaptive']
ax=fig.add_axes([m+.075,.208,.186,.262]);y=np.arange(6)
ax.barh(y,vals,color=[blue]*3+[coral]*3,height=.65);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,4.8);ax.set_xticks([0,1,2,3,4]);ax.tick_params(axis='both',labelsize=18,length=0,pad=7);ax.grid(axis='x',color=grid,lw=1.2);ax.set_axisbelow(True)
for j,v in enumerate(vals):ax.text(v+.07,j,f'+{v:.2f}',va='center',fontsize=19,fontweight='bold',color=ink)
for z in ax.spines.values():z.set_visible(False)
note(m,.166,'Starting accuracy: 92.33%. One session per\nreviewer and condition; all edits were frozen\nbefore evaluation on the same held-out set.',19)
# Right column
heading(q,.845,'3  WHAT CHANGED?')
note(q,.807,'Count of examples that changed after each edit\non the same 4,096 held-out inputs.',22)
transitions=json.loads((ROOT/'results/repair_v2_error_transitions.json').read_text())['sessions']
by_id={row['session_id']:row for row in transitions}
ax=fig.add_axes([q+.096,.483,.177,.253]); y=np.arange(len(ids))
ax.axhspan(.52,1.48,color='#edf5f5',zorder=0)
good=ax.barh(y-.18,[by_id[i]['improved'] for i in ids],.34,color=blue,label='Errors corrected')
bad=ax.barh(y+.18,[by_id[i]['regressed'] for i in ids],.34,color=coral,label='New errors')
for bar in [*good,*bad]:
    ax.text(bar.get_width()+2,bar.get_y()+bar.get_height()/2,str(int(bar.get_width())),va='center',fontsize=15,color=ink)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,205);ax.set_xticks([0,50,100,150,200]);ax.tick_params(axis='both',labelsize=15,length=0,pad=5);ax.grid(axis='x',color=grid);ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=15,ncol=2,loc='upper left',bbox_to_anchor=(-.42,1.07))
for z in ax.spines.values():z.set_visible(False)
text(q,.445,'ASTRA / FIXED PROBES',20,blue,'bold')
note(q,.417,'180 corrected; 33 new errors. Its edit raised\nQ-- accuracy from 78.2% to 95.8%, while\nQ-+ fell from 91.2% to 88.0%.',20)
heading(q,.333,'WHAT WE LEARNED')
note(q,.296,'All six edits raised accuracy in this case.\nOne checkpoint and one session per condition\ncannot establish a general model ranking or\nprove that probing caused the differences.',20)
text(q,.202,'POSSIBLE FUTURE: RECURSIVE SELF-IMPROVEMENT',18,blue,'bold')
note(q,.176,'Direct weight editing may eventually help\nself-improvement loops, where a model helps\nidentify and test changes to its own weights.\nThis study edited a separate, smaller model.',19)
# Footer is separate from the body with primary context and reproducibility references.
fig.add_artist(plt.Line2D([.035,.965],[.099,.099],transform=fig.transFigure,color=grid,lw=2))
text(.035,.080,'CONTEXT',18,blue,'bold'); note(.105,.080,'Direct weight changes have a history: pruning (LeCun et al., 1989) and targeted model editing (Meng et al., 2022).',17)
text(.035,.051,'STUDY',18,blue,'bold'); note(.105,.051,'Frozen harder-checkpoint evaluation  |  6 independent sessions  |  4,096 held-out examples  |  Self-rewriting: Fan et al. (2026), arxiv.org/abs/2611.21885',17)
output=ROOT/'deliverables/Read_Between_the_Weights_Poster_36x24.pdf'
fig.savefig(output,format='pdf',facecolor='white',metadata={'Title':'Read Between the Weights - Research Poster','Subject':'LLM-guided inspection and editing of a small neural network'})
print(output)
