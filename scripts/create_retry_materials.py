"""Create comparison handout and poster from frozen study output."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
root=Path('.'); data=json.loads((root/'deliverables/Read_Between_the_Weights_Retry_Results.json').read_text()); rows=data['sessions']
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
navy='#173448'; teal='#287b8c'; coral='#cb653f'; ink='#233642'; muted='#52626a'; pale='#e9f3f4'

def chart(ax):
    colors={'GPT-6.2 Astra':teal,'Qwen3.8-27B':coral,'Claude Fable 5.2 (retry)':'#7763a7','Gemini 4 Neon (retry)':'#be8a28'}
    names={'weights_only':'weights','fixed_probes':'fixed','adaptive_probes':'adaptive'}
    for i,r in enumerate(rows):
        y=i+i//3*.45
        if r['change_pp'] is None: ax.text(.05,y,'incomplete',va='center',fontsize=10,color=muted,style='italic')
        else:
            v=r['change_pp'];ax.barh(y,v,height=.68,color=colors[r['reviewer']]);ax.text(v+(.07 if v>=0 else -.07),y,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=10,color=ink,weight='bold')
    ax.set_yticks([i+i//3*.45 for i in range(len(rows))],[f"{r['reviewer']} / {names[r['condition']]}" for r in rows],fontsize=9)
    ax.set_ylim(len(rows)-1+(len(rows)-1)//3*.45+.9,-.8);ax.set_xlim(-.7,5.1);ax.set_xlabel('Accuracy change, percentage points',fontsize=10,color=ink);ax.set_xticks([-0.5,0,1,2,3,4,5]);ax.tick_params(axis='both',length=0,pad=5,labelsize=9);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    for p in ax.spines.values():p.set_visible(False)

def save(fig,path):
    fig.savefig(path,format='pdf',facecolor='white',metadata={'Title':'Read Between the Weights - Four-reviewer comparison'})
    fig.savefig(path.with_suffix('.png'),dpi=85,facecolor='white')
    plt.close(fig)

fig=plt.figure(figsize=(8.5,11),facecolor='white')
fig.add_artist(Rectangle((0,.88),1,.12,transform=fig.transFigure,color=navy))
fig.text(.06,.957,'READ BETWEEN THE WEIGHTS',fontsize=21,color='white',weight='bold',va='top')
fig.text(.06,.911,'Four-reviewer repair retry',fontsize=12,color='#cce9eb',va='top')
fig.text(.06,.850,'Can a reviewer suggest one useful weight edit?',fontsize=14,weight='bold',color=navy)
fig.text(.06,.817,'One 4-8-8-1 network  |  4,096 previously examined examples  |  Exploratory',fontsize=9.3,color=muted)
ax=fig.add_axes([.30,.362,.62,.414]);chart(ax)
fig.text(.06,.318,'What the comparison shows',fontsize=13,weight='bold',color=navy)
fig.text(.06,.291,'New Fable runs used a 16,384-token output cap.',fontsize=10.5,color=ink)
fig.text(.06,.263,'Gemini used paid Vertex Priority and a revised prompt.',fontsize=10.5,color=ink)
fig.text(.06,.235,'Incomplete sessions are explicitly labeled in the chart.',fontsize=10.2,color=ink)
fig.text(.06,.190,'Conditions: weights only; 32 fixed probes; or 32 adaptive probes in four rounds.\nOnly original logits were returned. Missing responses are not zero effects.',fontsize=9.4,color=ink,linespacing=1.5)
fig.text(.06,.115,'Limitations: one checkpoint and one session per condition; differing providers and reasoning\nsystems; reused held-out set. This cannot establish a general model ranking or probe causality.',fontsize=8.9,color=muted,linespacing=1.5)
fig.text(.06,.051,'Source: frozen repair v2 and additional-reviewer cohorts  |  23 Nov 2026',fontsize=8,color=muted)
save(fig,root/'deliverables/Read_Between_the_Weights_Retry_Handout.pdf')

fig=plt.figure(figsize=(36,24),facecolor='white')
fig.add_artist(Rectangle((0,.87),1,.13,transform=fig.transFigure,color=navy))
fig.text(.035,.969,'READ BETWEEN THE WEIGHTS',fontsize=72,color='white',weight='bold',va='top')
fig.text(.035,.914,'Can LLM reviewers repair a small neural network by inspecting its weights?',fontsize=30,color='#cce9eb',va='top')
fig.text(.04,.824,'THE STUDY',fontsize=33,color=navy,weight='bold')
fig.text(.04,.785,'One small network; one proposed weight edit',fontsize=25,color=ink,weight='bold')
fig.text(.04,.747,'4 inputs -> 8 hidden -> 8 hidden -> 1 logit\n121 parameters; one deliberately weakened checkpoint\n4,096 held-out examples; starting accuracy 92.33%',fontsize=22,color=ink,linespacing=1.5,va='top')
fig.text(.04,.606,'THREE EVIDENCE CONDITIONS',fontsize=28,color=navy,weight='bold')
fig.text(.04,.568,'Weights only: network parameters\nFixed: four predetermined batches of eight probes\nAdaptive: four reviewer-selected batches of eight',fontsize=21,color=ink,linespacing=1.55,va='top')
fig.text(.04,.422,'INTERPRETATION',fontsize=28,color=navy,weight='bold')
fig.text(.04,.385,'Original Astra and Qwen sessions used a 4,096-token cap.\nFable retry used a 16,384-token cap throughout.\nGemini retry used paid Vertex Priority and a revised prompt.\nEach row represents one independent session.\nIncomplete sessions do not count as zero effects.',fontsize=21,color=ink,linespacing=1.48,va='top')
fig.text(.04,.205,'Same checkpoint and held-out examples across sessions',fontsize=24,color=teal,weight='bold')
fig.text(.04,.166,'Differences in provider and output cap limit comparisons.',fontsize=20,color=muted)
fig.text(.50,.824,'ACCURACY CHANGE AFTER ONE EDIT',fontsize=32,color=navy,weight='bold')
ax=fig.add_axes([.61,.248,.31,.53]);chart(ax)
for label in ax.get_yticklabels(): label.set_fontsize(18)
ax.xaxis.label.set_size(18);ax.tick_params(axis='x',labelsize=17)
for label in ax.texts: label.set_fontsize(18)
fig.text(.50,.178,'One session per model and condition; no uncertainty interval.',fontsize=20,color=muted)
fig.add_artist(plt.Line2D([.035,.965],[.12,.12],transform=fig.transFigure,color='#dce5e8',lw=2))
fig.text(.04,.090,'LIMITS',fontsize=20,color=teal,weight='bold')
fig.text(.12,.090,'Exploratory reuse of an examined held-out set; different providers and reasoning systems. Results do not establish a general model ranking.',fontsize=19,color=muted)
save(fig,root/'deliverables/Read_Between_the_Weights_Retry_Poster_36x24.pdf')
print('Created comparison handout and poster.')
