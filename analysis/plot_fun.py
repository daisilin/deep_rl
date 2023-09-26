import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os
import sys
import seaborn as sns

def plot_one_feat_vs_iter(params_all_nn,feat,lowess=True):
    fig = plt.figure(figsize=(6,4))
    axs=[]
#     for ii,mcts in enumerate([25,50,80,100]):
    for ii,mcts in enumerate([100]):
        ax=fig.add_subplot(1,1,ii+1)
        ax.set_title(f'mcts = {mcts}')
        for cpuct in [2]:
            try:
                mask = (params_all_nn['mcts']==mcts)&(params_all_nn['cpuct']==cpuct)&(params_all_nn['iter']>=5)
                ax=sns.regplot(x='iter',y=feat,data=params_all_nn.loc[mask],ax=ax,label=str(cpuct),lowess=lowess)
                ax.legend(bbox_to_anchor=[1,1.05])
                axs.append(ax)
            except:
                pass
    plt.suptitle(feat)
    plt.tight_layout()


def plot_regression_coef(result,fig=None,ax=None):
    if ax is None:
        fig,ax=plt.subplots()
   
    xs = range(len(result.params))
    ax.bar(xs,result.params)
    ticklabels = []
    for i in result.params.index:
        s = i
        ticklabels.append(s)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticklabels,rotation=90)

    for ii,p in enumerate(result.pvalues):
        if (p<=0.05)&(p>=0.01):
            ax.text(xs[ii],result.params.iloc[ii]*1.1,'*',horizontalalignment='center')
        elif (p<0.01)&(p>=0.001):
            ax.text(xs[ii],result.params.iloc[ii]*1.1,'**',horizontalalignment='center')
        elif (p<0.001):
            ax.text(xs[ii],result.params.iloc[ii]*1.1,'***',horizontalalignment='center')

    return fig,ax
