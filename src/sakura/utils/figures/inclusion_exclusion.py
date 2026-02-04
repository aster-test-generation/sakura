_E='High'
_D='Medium'
_C='Low'
_B=False
_A=True
import matplotlib.pyplot as plt,numpy as np
data_incl={'One paragraph desc':[_A,_A,_A],'Imperative and instructional tone':[_A,_A,_A],'Code quoting':[_B,_B,_B],'Variable names':[_B,_B,_B],'Testing framework':[_A,_A,_A],'Mention exact \n identifier names':[_A,_A,_B],'Mention exact \n external resources':[_A,_A,_B],'Exact test value or range':[_A,_B,_B],'Provide example\n test values':[_B,_A,_B],'Mention all test annotation':[_A,_B,_B],'Mention behavior\n changing test annotation':[_A,_A,_B],'Test intent':[_A,_A,_A],'Detailed description\n of focal methods':[_A,_A,_B],'High level description\n of the user story':[_B,_B,_A],'Exact assertion details':[_A,_A,_B],'High level assertion details':[_B,_B,_A],'Mention exception classes':[_A,_A,_B],'Only mention if \nexception occurs or not.':[_B,_B,_A],'Detailed description\n of setup methods':[_A,_A,_B],'High-level description\n of setup methods':[_B,_B,_A],'Specific details \nof the teardown methods':[_A,_A,_B],'Specific mention \nof the helper methods':[_A,_A,_B],'Include description \nof helper methods \nas part of the test methods':[_B,_B,_A]}
properties=list(data_incl.keys())
num_vars=len(properties)
angles=np.linspace(0,2*np.pi,num_vars,endpoint=_B)
colors={_C:'#1f77b4',_D:'#ff7f0e',_E:'#2ca02c'}
rings={_C:1,_D:2,_E:3}
fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=_A))
for r in rings.values():ax.plot(np.linspace(0,2*np.pi,200),[r]*200,linestyle='dotted',color='gray',alpha=.5)
ax.set_ylim(0,3.5)
for(level,r)in rings.items():
	for(i,prop)in enumerate(properties):
		if data_incl[prop][r-1]:ax.bar(angles[i],.8,width=2*np.pi/num_vars,bottom=r-.4,color=colors[level],alpha=.7,edgecolor='k',linewidth=.3,label=level if i==0 else'')
ax.set_xticks(angles)
ax.set_xticklabels(properties,fontsize=6)
ax.set_yticks([1,2,3])
ax.set_yticklabels([_C,_D,_E],fontsize=9)
ax.legend(loc='upper right',bbox_to_anchor=(1.2,1.1))
plt.tight_layout()
plt.savefig('abstraction_inclusion_exclusion.pdf',dpi=300,bbox_inches='tight')