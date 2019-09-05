#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:40:57 2017
The script used to produce figures in Zhang, Rennie, Hawkins, Bathelt &Astle (2019).
@author: mz
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


###############Figure 1: Component planes and number of PPs in nodes###############
sns.set(style="ticks")
sns.set_style("whitegrid")
sns.set_style({'axes.grid': False,
                   'font.family': [u'serif'],
                   'font.sans-serif': [u'Computer Modern Unicode'],})
cmap = sns.diverging_palette(240, 10, as_cmap=True) # Generate a custom diverging colormap
names=['Forward Digit','Dot Matrix','Backward Digit','Mr.X']
xnames=['Forward \nDigit','Dot \nMatrix','Backward \nDigit','Mr.X']
#plot component maps
fig, axs = plt.subplots(2,2, figsize=(6, 7),sharex=True, sharey=True)
axs = axs.ravel()
cbar_ax = fig.add_axes([0.1, -0.03, 0.85, .03])

for i, ax in enumerate(axs): 
    w=weights[:,i].reshape((8,8))
    im = sns.heatmap(w[::-1], ax=ax, cmap='Blues', cbar= i== 0, 
                     cbar_kws={"orientation": "horizontal","ticks":[70,90,110,130]} if i==0 else None,
                    cbar_ax=None if i else cbar_ax, square=True,linewidth=0.1,
                    xticklabels=np.arange(1,9),yticklabels=np.arange(1,9)[::-1])
    ax.set_title(names[i],size=8)
#mappable = im.get_children()[0]
#plt.colorbar(mappable, ax = [],orientation = 'horizontal')
plt.suptitle('(a)SOM component planes',fontsize=12,x=0.5,y=1.05)
fig.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/component_plane.png', dpi=600)

plt.show()

#hitmap
node, count = np.unique(bmu1,return_counts=True)
fig, ax = plt.subplots(1,1,figsize=(5,5))
sns.heatmap(count.reshape((8,8))[::-1], annot=True, annot_kws={"size": 8}, ax=ax,
            cmap='Blues', square=True,linewidth=0.1, cbar=False,cbar_kws={"orientation": "horizontal"}, 
            xticklabels=np.arange(1,9),yticklabels=np.arange(1,9)[::-1])
plt.suptitle('(b)Number of participants\n in each node', fontsize=12 )

plt.show()


###############Figure 2: plot RDMs###############
cov_pre = np.corrcoef(WeightsPre.T); cov_pre[di]=1
di=np.diag_indices(cov_pre.shape[0],ndim=2)
cov_post = np.corrcoef(WeightsPost.T); cov_post[di]=1
cov_dif = cov_post- cov_pre; cov_dif[di]=0

####if want to custimize annotation on heatmap (e.g. statistical siginficance)
strings = strings = np.asarray([['', '', '', ''],
                                ['', '', '', ''],
                                ['*', '', '', ''],
                                ['', '', '*','']])
labels = (np.asarray([" {1:.2f}{0}".format(string, value)
                      for value, string in zip(cov_dif.flatten(),
                                               strings.flatten())])).reshape(4, 4)

fig, axs = plt.subplots(1,3,figsize=(14,4), sharex=True, sharey=True)
axs=axs.ravel()
cmap_rdm = sns.diverging_palette(240, 10, as_cmap=True) # Generate a custom diverging colormap
g = sns.heatmap(cov_pre,annot=True,annot_kws={"size": 8}, vmin=0,vmax=1, ax=axs[0], 
            square=True,cmap='Reds', cbar_kws={"shrink": .6}, 
            xticklabels=xnames,yticklabels=names)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g= sns.heatmap(cov_post,annot=True,annot_kws={"size": 8}, vmin=0,vmax=1, ax=axs[1], 
            square=True,cmap='Reds', cbar_kws={"shrink": .6},
            xticklabels=xnames,yticklabels=names)
g.set_xticklabels(g.get_xticklabels(), rotation=45)

g=sns.heatmap(cov_dif,annot = labels, fmt = '',annot_kws={"size": 8}, vmin=-0.5,vmax=0.5, ax=axs[2], 
            square=True,cmap=cmap_rdm, cbar_kws={"shrink": .6},
            xticklabels=xnames, yticklabels=names)
g.set_xticklabels(g.get_xticklabels(), rotation=45)

axs[0].set_title('(a) Pre training \ncorelation matrix (SOM weights)',size=9)
axs[1].set_title('(b) Post training \ncorelation matrix (SOM weights)',size=9)
axs[2].set_title('(c) Post-Pre',size=9)
axs[0].tick_params('x', pad=-15)
axs[1].tick_params('x', pad=-15)
axs[2].tick_params('x', pad=-15)


suptitle = fig.suptitle('Change in task relationships over time (SOM weights)', fontsize=12, x=0.5, y=1.05)
fig.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/rdm(som).png', dpi=600,\
            bbox_inches='tight',bbox_extra_artists=[suptitle])
plt.show()




##################################################################
###############Figure 3: Plot the k-mean figure panel#############
##################################################################
#read in matlab matrices
clabels=pd.DataFrame(data=CalmAceRawPPcluster, columns=\
        ['FD','DM','BD','MrX','c_calm'])

clabels_training=pd.DataFrame(data=CalmAceRawPPclusterSOM_Pre_Post_IQ, columns=\
        ['FD_pre','DM_pre','BD_pre','MrX_pre','c_pre',\
         'FD_post','DM_post','BD_post','MrX_post','c_post', 'IQ'])
improvement = pd.DataFrame(data=improvements, columns=['FD_dif','DM_dif','BD_dif','MrX_dif'])
clabels2 = pd.concat([clabels_training,improvement], axis=1)
clabel_pre = clabels2[['FD_pre','DM_pre','BD_pre','MrX_pre','c_pre']]
clabel_post = clabels2[['FD_post','DM_post','BD_post','MrX_post','c_post']]
mover1 = clabels2.loc[(clabels2.c_pre!=1)&(clabels2.c_post==1)][['FD_dif','DM_dif','BD_dif','MrX_dif','IQ']]
mover2 = clabels2.loc[(clabels2.c_pre!=2)&(clabels2.c_post==2)][['FD_dif','DM_dif','BD_dif','MrX_dif','IQ']]
mover3 = clabels2.loc[(clabels2.c_pre!=3)&(clabels2.c_post==3)][['FD_dif','DM_dif','BD_dif','MrX_dif','IQ']]
stayer4 = clabels2.loc[(clabels2.c_pre==4)&(clabels2.c_post==4)][['FD_dif','DM_dif','BD_dif','MrX_dif','IQ']]
movers = pd.concat([mover1,mover2,mover3,stayer4], keys=['mover1', 'mover2', 'mover3','stayer4']).reset_index()
movers=movers.rename({'level_0':'group'}, axis=1)
ci_list=[]; m_list=[]; size=[] #calculate mean and confidence interval and N for plotting
ci_list.append(clabels.groupby(['c_calm']).sem().reset_index()[['FD','DM','BD','MrX']].values*1.96)
m_list.append(clabels.groupby(['c_calm']).mean().reset_index()[['FD','DM','BD','MrX']].values)
size.append(clabels.groupby(['c_calm']).size().values)
ci_list.append(clabels2.groupby(['c_pre']).sem().reset_index()[['FD_pre','DM_pre','BD_pre','MrX_pre']].values*1.96)
m_list.append(clabels2.groupby(['c_pre']).mean().reset_index()[['FD_pre','DM_pre','BD_pre','MrX_pre']].values)
size.append(clabels2.groupby(['c_pre']).size().values)
ci_list.append(clabels2.groupby(['c_post']).sem().reset_index()[['FD_post','DM_post','BD_post','MrX_post']].values*1.96)
m_list.append(clabels2.groupby(['c_post']).mean().reset_index()[['FD_post','DM_post','BD_post','MrX_post']].values)
size.append(clabels2.groupby(['c_post']).size().values)
ci_list.append(movers.groupby(['group']).sem().reset_index()[['FD_dif','DM_dif','BD_dif','MrX_dif']].values*1.96)
m_list.append(movers.groupby(['group']).mean().reset_index()[['FD_dif','DM_dif','BD_dif','MrX_dif']].values)
size.append(movers.groupby(['group']).size().values)

from matplotlib import colors
cmap = colors.ListedColormap(['turquoise', 'gold', 'firebrick', 'limegreen'])
colours = ['turquoise', 'gold', 'firebrick', 'limegreen', 'darkorange','darkorchid', 'blue']
fig_title = ['(c) CALM/ACE sample','(d) Pre-training','(e) Post-training','(f) Interest group improvements']

fig, axs = plt.subplots(3,2,figsize=(10,14))
axs=axs.ravel()
## plot cluster map into figure panel
ax=sns.heatmap(SortedNodeClust, annot=True, annot_kws={"size": 8}, ax=axs[0],
            cmap=cmap, square=True,linewidth=0.1, cbar=True, cbar_kws={"shrink": .82},
            xticklabels=np.arange(1,9),yticklabels=np.arange(1,9)[::-1])
axs[0].set_title('(a) Cluster map',size=9)
# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1, 2, 3,4])
colorbar.set_ticklabels(['c1', 'c2', 'c3','c4'])

#plot IQ into figure panel
#dropna because several movers don't have IQ data
sns.boxplot(x='group',y='IQ', data=movers.dropna(), ax=axs[1], whis=[5,95], palette=colours[:4])
axs[1].set_xticklabels(['mover to c1','mover to c2','mover to c3','stayer in c4'])
axs[1].set_title('(b) Interest group IQ',size=9)
axs[1].set_xlabel('')
axs[1].set(ylabel = "Pre-training IQ")
axs[1].set_ylim([70,140])

# statistical annotation
x1, x2, x3, x4 = 0, 1, 2, 3   # bar 0,1,2,3
y1 =  movers.loc[movers.group=='mover1'].IQ.mean()+14 
y2, y3, y4, col = y1+2, y1+4, movers.loc[movers.group=='mover3'].IQ.mean()+18,'k'
axs[1].plot([x1+0.1, x2-0.1], [y1, y1], lw=1.0, c=col)
axs[1].plot([x1+0.1, x3-0.1], [y2, y2], lw=1.0, c=col)
axs[1].plot([x1+0.1, x4-0.1], [y3, y3], lw=1.0, c=col)
axs[1].plot([x3+0.1, x4-0.1], [y4, y4], lw=1.0, c=col)
axs[1].text((x1+x3)*.5, y3+2, "*", ha='center', color=col)
axs[1].text((x3+x4)*.5, y4+2, "*", ha='center', color=col)

#############subgroups################
for j, ax in enumerate(axs[2:]): 
    for i in range(n_clusters): #mlist=[calm, pre, post, mover]; ci = 1.96*std/square root(n)
        #change the linestyle of mover plot to differentiate from other plots
        if j==3:
            ax.errorbar((np.array(range(4))+0.05*i),m_list[j][i,:], yerr=ci_list[j][i,:], 
                    color=colours[i], linestyle='-.' )
        else:
            ax.errorbar((np.array(range(4))+0.05*i),m_list[j][i,:], yerr=ci_list[j][i,:], 
                    color=colours[i])
    if j==0 or j==1: 
        plt.setp(ax.get_xticklabels(),visible=False)

    if j==3: 
        ax.set_ylim([-20,40])
        ax.set_xticklabels(names, rotation=30)
        ax.legend([r"$\bf{" + 'mover' + "}$"+ " to c1(N=%d) "% size[j][0], r"$\bf{" + 'mover' + "}$"+ " to c2(N=%d)"% size[j][1],\
                   r"$\bf{" + 'mover' + "}$"+ " to c3(N=%d)"% size[j][2],r"$\bf{" + 'stayer' + "}$"+ " in c4(N=%d)"% size[j][3]], \
                    frameon=False, loc=4)
    if j==2: 
        ax.set_xticklabels(names, rotation=30)
    if j!=3:    
        ax.set_ylim([60,130])
        ax.legend(['c1(N=%d) '% size[j][0],'c2(N=%d)'% size[j][1],\
                   'c3(N=%d)'% size[j][2],'c4(N=%d)'% size[j][3]], frameon=False, loc=4)
#    ax.set_facecolor('white')
    ax.set_xlim([-0.5,4-0.5])
    ax.set_xticks(range(4))
    if j%2==0: ax.set_ylabel('Stantard scores')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax.set_title(fig_title[j],size=9)
plt.suptitle('Results of K-mean clustering and group profiles',fontsize=12)
#plt.show()
plt.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/kmean.png', dpi=300, bbox_inches = "tight")
    
################Supplementary figure: 
################Demonstration of the impact of SOM training parameters on model performance
#read in Matlab matrix as dataframe
param_mat=pd.DataFrame(data=MengyaMatrix, columns=\
        ['FD','DM','BD','MrX','topo_error','quan_error',\
         'map_size','initial_neighbourhood size','rough_training','fine_training',\
         'prediction_error (z_score)','quantization_error (z_score)', 'composite_score'])#the order of tasks needs checking
#make snapshots of 3d figure, can be used to produce .gif
#not important
#for angle in range(70,210,20):
#    # Make the plot
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.scatter(temp.map_size,temp.rough_train,temp.composite, s=50, c=temp.Init_n)
#    
#    # Set the angle of the camera
#    ax.view_init(10,angle)
#    # Save it
#    filename='/Users/mz/Desktop/PhD/SOM_paper/Figure_CALM_ACE/animation/'+str(angle)+'.png'
#    plt.savefig(filename, dpi=96)

######Composite score as function of parameter change#####

g=sns.factorplot(x="rough_training", y='quantization_error (z_score)', 
                             data=param_mat.loc[(param_mat.map_size.isin([6,8,10]))&
                                (param_mat.rough_training.isin([2,4,6,8,10,12,20,30]))&
                                (param_mat['initial_neighbourhood size'].isin([2,3,4]))],
                              hue="map_size", col='initial_neighbourhood size', row="fine_training",kind="point",
                              dodge=True, height=4, aspect=1.4, linestyles=["-", "--",':'])
g.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/quantization_error.png' , dpi=200, bbox_inches = "tight")
    

################Supplementary figure: 
############Silhouette plot for SOM weights and raw data##########
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.cm as cm
    n_clusters = 4
    # Calculate average silhouette score 
    # 
    silhouette_avg = np.mean(SilValsClust[:,0])
    silhouette_avg_raw = np.mean(SilValsClustRaw[:,0]) #for cluster result on raw data

    # Create a subplot with 1 row and 2 columns
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,5))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1]); ax2.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(SilValsClust) + (n_clusters + 1) * 10])
    ax2.set_ylim([0, len(SilValsClustRaw) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            SilValsClust[:,0][SilValsClust[:,1] == i+1]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=colours[i], edgecolor=colours[i], alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette plot for K=4 on SOM weights")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color='darkorange', linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    #2nd plot for the raw data
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            SilValsClustRaw[:,0][SilValsClustRaw[:,1] == i+1]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax2.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=colours[i], edgecolor=colours[i], alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax2.set_title("Silhouette plot for K=4 on raw CALM/ACE data")
    ax2.set_xlabel("Silhouette coefficient values")
    ax2.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax2.axvline(x=silhouette_avg_raw, color='darkorange', linestyle="--")

    ax2.set_yticks([])  # Clear the yaxis labels / ticks
    ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    #save figure
    plt.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/silhouette.png', dpi=200, bbox_inches = "tight")
    plt.show()

#######Mean silhouette value with different K############
silvalue=SILSOL.T[:,0]
ax=sns.pointplot(x=np.arange(2,9),y=silvalue)
ax.set_title("Mean silhouette values for k-cluster solutions")
ax.set_xlabel("Number of cluster (k)")
ax.set_ylabel("mean silhouette value")
plt.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/mean_silhouette.png', dpi=200,bbox_inches = "tight")

#######Group profiles with Pre and Post training k=4############
clabels_post=pd.DataFrame(data=ClustMatrixPost, columns=\
        ['FD_post','DM_post','BD_Post','MrX_post','c_post'])
clabels_pre=pd.DataFrame(data=ClustMatrixPre, columns=\
        ['FD_pre','DM_pre','BD_pre','MrX_pre','c_pre'])
ci_list_training=[]; m_list_training=[]; size_training=[]
ci_list_training.append(clabels_pre.groupby(['c_pre']).sem().reset_index()[['FD_pre','DM_pre','BD_pre','MrX_pre']].values*1.96)
m_list_training.append(clabels_pre.groupby(['c_pre']).mean().reset_index()[['FD_pre','DM_pre','BD_pre','MrX_pre']].values)
size_training.append(clabels_pre.groupby(['c_pre']).size().values)
ci_list_training.append(clabels_post.groupby(['c_post']).sem().reset_index()[['FD_post','DM_post','BD_Post','MrX_post']].values*1.96)
m_list_training.append(clabels_post.groupby(['c_post']).mean().reset_index()[['FD_post','DM_post','BD_Post','MrX_post']].values)
size_training.append(clabels_post.groupby(['c_post']).size().values)
fig_title_training = ['(a) Pre-training','(b) Post-training']
fig, axs = plt.subplots(1,2,figsize=(10,6))
axs=axs.ravel()
for j, ax in enumerate(axs): 
    for i in range(n_clusters): #mlist=[calm, pre, post, mover]; ci = 1.96*std/square root(n)
        ax.errorbar((np.array(range(4))+0.05*i),m_list_training[j][i,:], yerr=ci_list_training[j][i,:], 
                    color=colours[i])
        ax.set_xticklabels(names, rotation=30)   
        ax.set_ylim([60,140])
        ax.legend(['c1 (N=%d) '% size_training[j][0],'c2 (N=%d)'% size_training[j][1],\
                   'c3 (N=%d)'% size_training[j][2],'c4 (N=%d)'% size_training[j][3]], frameon=False, loc=4)
        ax.set_xlim([-0.5,4-0.5])
        ax.set_xticks(range(4))
        if j==0: ax.set_ylabel('Stantard scores')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_title(fig_title_training[j],size=9)

plt.suptitle('Group profiles with k=4 on Pre- and Post-training SOMs',fontsize=12)
plt.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/kmean_pre&postSOM.png', dpi=300, bbox_inches = "tight")
plt.show()

#######Group profiles with different K############

clabels_difk=pd.DataFrame(data=CalmAceClusterMatrix, columns=\
        ['FD','DM','BD','MrX','k2','k3','k4','k5'])
ci_list_difk=[]; m_list_difk=[]; size_difk=[]
ci_list_difk.append(clabels_difk.groupby(['k2']).sem().reset_index()[['FD','DM','BD','MrX']].values*1.96)
m_list_difk.append(clabels_difk.groupby(['k2']).mean().reset_index()[['FD','DM','BD','MrX']].values)
size_difk.append(clabels_difk.groupby(['k2']).size().values)
ci_list_difk.append(clabels_difk.groupby(['k3']).sem().reset_index()[['FD','DM','BD','MrX']].values*1.96)
m_list_difk.append(clabels_difk.groupby(['k3']).mean().reset_index()[['FD','DM','BD','MrX']].values)
size_difk.append(clabels_difk.groupby(['k3']).size().values)
ci_list_difk.append(clabels_difk.groupby(['k4']).sem().reset_index()[['FD','DM','BD','MrX']].values*1.96)
m_list_difk.append(clabels_difk.groupby(['k4']).mean().reset_index()[['FD','DM','BD','MrX']].values)
size_difk.append(clabels_difk.groupby(['k4']).size().values)
ci_list_difk.append(clabels_difk.groupby(['k5']).sem().reset_index()[['FD','DM','BD','MrX']].values*1.96)
m_list_difk.append(clabels_difk.groupby(['k5']).mean().reset_index()[['FD','DM','BD','MrX']].values)
size_difk.append(clabels_difk.groupby(['k5']).size().values)
fig_title_difk = ['(a) K=2','(b) K=3','(c) K=4','(d) K=5']
k_clusters = np.arange(2,6)
fig, axs = plt.subplots(2,2,figsize=(10,10))
axs=axs.ravel()
for j, ax in enumerate(axs): 
    for i in range(k_clusters[j]): #mlist=[calm, pre, post, mover]; ci = 1.96*std/square root(n)
        ax.errorbar((np.array(range(4))+0.05*i),m_list_difk[j][i,:], yerr=ci_list_difk[j][i,:], 
                    color=colours[i])
        ax.set_xticklabels(names, rotation=30)   
        ax.set_ylim([60,130])
        ax.set_xlim([-0.5,4-0.5])
        ax.set_xticks(range(4))
        if j==0 or 2: ax.set_ylabel('Stantard scores')
        if j==0 or j==1: 
            plt.setp(ax.get_xticklabels(),visible=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_title(fig_title_difk[j],size=9)
        if j==0:
            ax.legend(['c1 (N=%d) '% size_difk[j][0],'c2 (N=%d)'% size_difk[j][1]],frameon=False, loc=4)
        if j==1:
            ax.legend(['c1 (N=%d) '% size_difk[j][0],'c2 (N=%d)'% size_difk[j][1],\
                   'c3 (N=%d)'% size_difk[j][2]], frameon=False, loc=4)
        if j==2:
            ax.legend(['c1 (N=%d) '% size_difk[j][0],'c2 (N=%d)'% size_difk[j][1],\
                   'c3 (N=%d)'% size_difk[j][2],'c4 (N=%d)'% size_difk[j][3]], frameon=False, loc=4)
        if j==3:
            ax.legend(['c1 (N=%d) '% size_difk[j][0],'c2 (N=%d)'% size_difk[j][1],\
                   'c3 (N=%d)'% size_difk[j][2],'c4 (N=%d)'% size_difk[j][3], 'c5 (N=%d)'% size_difk[j][4]],\
                frameon=False, loc=4)

plt.suptitle('Group profiles with different Ks on CALM/ACE SOM',fontsize=12)
plt.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/kmean_differentK.png', dpi=300,bbox_inches = "tight")

plt.show()

#######COGITO panel (swarmplot, SOM participant distribution)############
xl = pd.read_excel('/Users/mz/Desktop/PhD/SOM_paper/cogito_data/Cogito6task.xls',sheetname=0,header=0)
names_cogito = ['Animal Span','N-Back', 'Spatial Updating',\
                'Reading Span', 'Counting Span','Rotation Span']
df=xl.melt()
for i,x in enumerate(xl.columns):
    df.variable.values[df.variable.values==x]=i
df = df.assign (task1= (df.variable.values/2+1))
df = df.assign (task2= (df.variable.values%2+1))
df['task1'] = df['task1'].map({1: names_cogito[0], 2: names_cogito[1], 3:names_cogito[2], \
                                4:names_cogito[3], 5: names_cogito[4], 6:names_cogito[5]})
df['task2'] = df['task2'].map({1: 'Pre-training', 2: 'Post-training'}) #How to recode dataFrame column values
bmu1=CogitoMatrixMengya[:,6]; bmu2=CogitoMatrixMengya[:,13]
node, count = np.unique(bmu1,return_counts=True)
node, count_post = np.unique(bmu2,return_counts=True)
count_post=list(count_post)
count_post.insert(0,2); count_post=np.array(count_post)
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(211)
ax=sns.swarmplot(x='task1',y='value',hue='task2',data=df, dodge=True, palette='Set2',\
                 hue_order=['Pre-training','Post-training'],ax=ax); 
ax.set_xlabel('6 COGITO tasks'); 
ax.set_ylabel('task performance (accuracy)');
l = ax.legend(); 
ax.set_title('(a) Distribution of task scores in pre and post COGITO data',fontsize=11);
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax1 = fig.add_subplot(223)
sns.heatmap(count.reshape((6,6))[::-1], annot=True, annot_kws={"size": 8}, ax=ax1,
            cmap='Blues', square=True,linewidth=0.1, cbar=False,cbar_kws={"orientation": "horizontal"}, 
            xticklabels=np.arange(1,7),yticklabels=np.arange(1,7)[::-1])
ax1.set_title('(b) Number of participants \nin each node (pre-training)', fontsize=11 );
ax2 = fig.add_subplot(224)
sns.heatmap(count_post.reshape((6,6))[::-1], annot=True, annot_kws={"size": 8}, ax=ax2,
            cmap='Blues', square=True,linewidth=0.1, cbar=False,cbar_kws={"orientation": "horizontal"}, 
            xticklabels=np.arange(1,7),yticklabels=np.arange(1,7)[::-1])
ax2.set_title('(c) Number of participants \nin each node (pre-training)', fontsize=11 ); 
fig.tight_layout()

plt.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/COGITO_panel.png', dpi=300,bbox_inches = "tight")
plt.show()

#######COGITO (pre and post component planes)############
names_cogito = ['Animal Span','N-Back', 'Spatial Updating',\
                'Reading Span', 'Counting Span','Rotation Span']
fig, axs = plt.subplots(2,3, figsize=(6 ,4),sharex=True, sharey=True)
axs = axs.ravel()
cbar_ax = fig.add_axes([0.1, -0.03, 0.85, .03])

for i, ax in enumerate(axs): 
    w=weightspreTrain[:,i].reshape((6,6))
    im = sns.heatmap(w[::-1], ax=ax, cmap='Blues', cbar= i== 0, 
                     cbar_kws={"orientation": "horizontal","ticks":[0.2,0.4,0.6,0.8,1.0]} if i==0 else None,
                    vmin=weightspreTrain.min(),vmax=weightspreTrain.max(),
                    cbar_ax=None if i else cbar_ax, square=True,linewidth=0.1,
                    xticklabels=np.arange(1,7),yticklabels=np.arange(1,7)[::-1])
    ax.set_title(names_cogito[i],size=10)
plt.suptitle('(a) COGITO pre-training SOM component planes',fontsize=12,x=0.5,y=1.05)
fig.tight_layout()
plt.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/COGITOpre_componentplane.png', dpi=400, bbox_inches = "tight")

plt.show()

fig, axs = plt.subplots(2,3, figsize=(6, 4))
axs = axs.ravel()
cbar_ax = fig.add_axes([0.1, -0.03, 0.85, .03])

for i, ax in enumerate(axs): 
    w=weightspostTrain[:,i].reshape((6,6))
    im = sns.heatmap(w[::-1], ax=ax, cmap='Blues', cbar= i== 0, 
                     cbar_kws={"orientation": "horizontal","ticks":[0.2,0.4,0.6,0.8,1.0]} if i==0 else None,
                    vmin=weightspreTrain.min(),vmax=weightspreTrain.max(),
                    cbar_ax=None if i else cbar_ax, square=True,linewidth=0.1,
                    xticklabels=np.arange(1,7),yticklabels=np.arange(1,7)[::-1])
    ax.set_title(names_cogito[i],size=10)
plt.suptitle('(b) COGITO post-training SOM component planes',fontsize=12,x=0.5,y=1.05)
fig.tight_layout()
plt.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/COGITOpost_componentplane.png', dpi=400,bbox_inches = "tight")

plt.show()

###############Control data : RDMs and profiles###############
###############Significance testing for change in relationships###############
RDM_pairwiser_test(WeightsControlsPre, WeightsControlPost, (1,3))
RDM_pairwiser_test(clabels_control.iloc[:,0:4].values, clabels_control.iloc[:,5:9].values, (0,1))


############################################################
cov_pre = np.corrcoef(WeightsControlsPre.T); 
di=np.diag_indices(cov_pre.shape[0],ndim=2)
cov_pre[di]=1
cov_post = np.corrcoef(WeightsControlPost.T); 
cov_post[di]=1
cov_dif = cov_post- cov_pre; cov_dif[di]=0

fig, axs = plt.subplots(1,3,figsize=(14,4), sharex=True, sharey=True)
axs=axs.ravel()
cmap_rdm = sns.diverging_palette(240, 10, as_cmap=True) # Generate a custom diverging colormap
g = sns.heatmap(cov_pre,annot=True,annot_kws={"size": 8}, vmin=0,vmax=1, ax=axs[0], 
            square=True,cmap='Reds', cbar_kws={"shrink": .6}, 
            xticklabels=xnames,yticklabels=names)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g= sns.heatmap(cov_post,annot=True,annot_kws={"size": 8}, vmin=0,vmax=1, ax=axs[1], 
            square=True,cmap='Reds', cbar_kws={"shrink": .6},
            xticklabels=xnames,yticklabels=names)
g.set_xticklabels(g.get_xticklabels(), rotation=45)

g=sns.heatmap(cov_dif,annot = labels, fmt = '',annot_kws={"size": 8}, vmin=-0.5,vmax=0.5, ax=axs[2], 
            square=True,cmap=cmap_rdm, cbar_kws={"shrink": .6},
            xticklabels=xnames, yticklabels=names)
g.set_xticklabels(g.get_xticklabels(), rotation=45)

axs[0].set_title('(a) Pre training \ncorelation matrix (SOM weights)',size=9)
axs[1].set_title('(b) Post training \ncorelation matrix (SOM weights)',size=9)
axs[2].set_title('(c) Post-Pre',size=9)
axs[0].tick_params('x', pad=-15)
axs[1].tick_params('x', pad=-15)
axs[2].tick_params('x', pad=-15)


suptitle = fig.suptitle('Control group change in task relationships over time (SOM weights)', fontsize=12, x=0.5, y=1.05)
fig.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/rdm(control_som).png', dpi=300,\
            bbox_inches='tight',bbox_extra_artists=[suptitle])
plt.show()

#########if want to custimize annotation on heatmap (e.g. statistical siginficance)
strings = strings = np.asarray([['', '', '', ''],
                                ['*', '', '', ''],
                                ['', '*', '', ''],
                                ['', '*', '','']])
labels = (np.asarray([" {1:.2f}{0}".format(string, value)
                      for value, string in zip(cov_dif.flatten(),
                                               strings.flatten())])).reshape(4, 4)
    
#####################Control data RDMs raw data#######################################
cov_pre = np.corrcoef(clabels_control.iloc[:,0:4].values.T); 
di=np.diag_indices(cov_pre.shape[0],ndim=2)
cov_pre[di]=1
cov_post = np.corrcoef(clabels_control.iloc[:,5:9].values.T); 
cov_post[di]=1
cov_dif = cov_post- cov_pre; cov_dif[di]=0
strings_raw = strings_raw = np.asarray([['', '', '', ''],
                                ['*', '', '', ''],
                                ['', '', '', ''],
                                ['', '', '','']])
labels_raw = (np.asarray([" {1:.2f}{0}".format(string, value)
                      for value, string in zip(cov_dif.flatten(),
                                               strings_raw.flatten())])).reshape(4, 4)

fig, axs = plt.subplots(1,3,figsize=(14,4), sharex=True, sharey=True)
axs=axs.ravel()
cmap_rdm = sns.diverging_palette(240, 10, as_cmap=True) # Generate a custom diverging colormap
g = sns.heatmap(cov_pre,annot=True,annot_kws={"size": 8}, vmin=0,vmax=1, ax=axs[0], 
            square=True,cmap='Reds', cbar_kws={"shrink": .6}, 
            xticklabels=xnames,yticklabels=names)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g= sns.heatmap(cov_post,annot=True,annot_kws={"size": 8}, vmin=0,vmax=1, ax=axs[1], 
            square=True,cmap='Reds', cbar_kws={"shrink": .6},
            xticklabels=xnames,yticklabels=names)
g.set_xticklabels(g.get_xticklabels(), rotation=45)

g=sns.heatmap(cov_dif,annot = labels_raw, fmt = '',annot_kws={"size": 8}, vmin=-0.5,vmax=0.5, ax=axs[2], 
            square=True,cmap=cmap_rdm, cbar_kws={"shrink": .6},
            xticklabels=xnames, yticklabels=names)
g.set_xticklabels(g.get_xticklabels(), rotation=45)

axs[0].set_title('(a) Pre training \ncorelation matrix (raw data)',size=9)
axs[1].set_title('(b) Post training \ncorelation matrix (raw data)',size=9)
axs[2].set_title('(c) Post-Pre',size=9)
axs[0].tick_params('x', pad=-15)
axs[1].tick_params('x', pad=-15)
axs[2].tick_params('x', pad=-15)


suptitle = fig.suptitle('Control group change in task relationships over time (raw data)', fontsize=12, x=0.5, y=1.05)
fig.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/rdm(control_raw).png', dpi=300,\
            bbox_inches='tight',bbox_extra_artists=[suptitle])
plt.show()


##################################################################
###############Control data: pre and post and gain profiles#############
#read in matlab matrices
clabels_control=pd.DataFrame(data=ControlMatrix, columns=\
        ['FD_pre','DM_pre','BD_pre','MrX_pre','c_pre',\
         'FD_post','DM_post','BD_post','MrX_post','c_post'], dtype='int32')
improvement = pd.DataFrame(data= np.subtract(clabels_control.iloc[:,5:9].values,clabels_control.iloc[:,0:4].values) , \
                columns=['FD_dif','DM_dif','BD_dif','MrX_dif'])
clabels2 = pd.concat([clabels_control,improvement], axis=1)
clabel_pre = clabels2[['FD_pre','DM_pre','BD_pre','MrX_pre','c_pre']]
clabel_post = clabels2[['FD_post','DM_post','BD_post','MrX_post','c_post']]
mover1 = clabels2.loc[(clabels2.c_pre!=1)&(clabels2.c_post==1)][['FD_dif','DM_dif','BD_dif','MrX_dif']]
mover2 = clabels2.loc[(clabels2.c_pre!=2)&(clabels2.c_post==2)][['FD_dif','DM_dif','BD_dif','MrX_dif']]
mover3 = clabels2.loc[(clabels2.c_pre!=3)&(clabels2.c_post==3)][['FD_dif','DM_dif','BD_dif','MrX_dif']]
stayer4 = clabels2.loc[(clabels2.c_pre==4)&(clabels2.c_post==4)][['FD_dif','DM_dif','BD_dif','MrX_dif']]
movers = pd.concat([mover1,mover2,mover3,stayer4], keys=['mover1', 'mover2', 'mover3','stayer4']).reset_index()
movers=movers.rename({'level_0':'group'}, axis=1)
ci_list=[]; m_list=[]; size=[] #calculate mean and confidence interval and N for plotting
ci_list.append(clabels2.groupby(['c_pre']).sem().reset_index()[['FD_pre','DM_pre','BD_pre','MrX_pre']].values*1.96)
m_list.append(clabels2.groupby(['c_pre']).mean().reset_index()[['FD_pre','DM_pre','BD_pre','MrX_pre']].values)
size.append(clabels2.groupby(['c_pre']).size().values)
ci_list.append(clabels2.groupby(['c_post']).sem().reset_index()[['FD_post','DM_post','BD_post','MrX_post']].values*1.96)
m_list.append(clabels2.groupby(['c_post']).mean().reset_index()[['FD_post','DM_post','BD_post','MrX_post']].values)
size.append(clabels2.groupby(['c_post']).size().values)
ci_list.append(movers.groupby(['group']).sem().reset_index()[['FD_dif','DM_dif','BD_dif','MrX_dif']].values*1.96)
m_list.append(movers.groupby(['group']).mean().reset_index()[['FD_dif','DM_dif','BD_dif','MrX_dif']].values)
size.append(movers.groupby(['group']).size().values)

colours = ['turquoise', 'gold', 'firebrick', 'limegreen', 'darkorange','darkorchid', 'blue']
fig_title = ['(a) Pre-training','(b) Post-training','(c) Interest group improvements']
fig, axs = plt.subplots(1,3,figsize=(14,4))
axs=axs.ravel()
#############subgroups
for j, ax in enumerate(axs): 
    for i in range(n_clusters): #mlist=[calm, pre, post, mover]; ci = 1.96*std/square root(n)
        ax.errorbar((np.array(range(4))+0.05*i),m_list[j][i,:], yerr=ci_list[j][i,:], 
                    color=colours[i] )
    if j==2: 
        ax.set_ylim([-20,40])
        ax.set_xticklabels(names, rotation=30)
        ax.legend(['mover to c1(N=%d) '% size[j][0],'mover to c2(N=%d)'% size[j][1],\
                   'mover to c3(N=%d)'% size[j][2],'stayer in c4(N=%d)'% size[j][3]], frameon=False, loc=4)

    ax.set_xticklabels(names, rotation=30)
    if j!=2:    
        ax.set_ylim([60,130])
        ax.legend(['c1(N=%d) '% size[j][0],'c2(N=%d)'% size[j][1],\
                   'c3(N=%d)'% size[j][2],'c4(N=%d)'% size[j][3]], frameon=False, loc=4)
#    ax.set_facecolor('white')
    ax.set_xlim([-0.5,4-0.5])
    ax.set_xticks(range(4))
    if j==0: ax.set_ylabel('Stantard scores')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax.set_title(fig_title[j],size=9)
plt.suptitle('Control group pre and post-training profiles',fontsize=12)

plt.savefig('/Users/mz/Desktop/PhD/SOM_paper/paper_review/kmean_control.png', dpi=300, bbox_inches = "tight")
plt.show()
##############Statistical analyses (mainly ANOVA and post-hoc tests)##############
##############one-way ANOVA test for between group difference in gain scores (training vs. control)
from scipy.stats import ttest_ind
ttest_ind(xl[xl['group']==1].DR_dif, xl[xl['group']==2].DR_dif, equal_var=False)
ttest_ind(xl[xl['group']==1].DM_dif, xl[xl['group']==2].DM_dif, equal_var=False)
ttest_ind(xl[xl['group']==1].BDR_dif, xl[xl['group']==2].BDR_dif, equal_var=False)
ttest_ind(xl[xl['group']==1].MrX_dif, xl[xl['group']==2].MrX_dif, equal_var=False)

##############one-way ANOVA test for group IQ difference
import statsmodels.api as sm
from statsmodels.formula.api import ols
results = ols('IQ ~ C(group)', data=movers.dropna()).fit()
results.summary()
aov_table = sm.stats.anova_lm(results, typ=2)
#############post-hoc pairwise comparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
re = pairwise_tukeyhsd(movers.dropna().IQ,movers.dropna().group)
print re
#print p-vlaues for group comparison
psturng(np.abs(re.meandiffs / re.std_pairs), len(re.groupsunique), re.df_total)
##############two-way ANOVA test for improvement profiles
a=movers.drop(['level_1','IQ'],axis=1)
a=pd.melt(a, id_vars=['group'], var_name='task', value_name='scores')
formula = 'scores~ C(group) + C(task) + C(group):C(task)'
formula1 = 'MrX_dif~ C(group)'
model = ols(formula1, movers.dropna()).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print aov_table
m=[]
for x in movers.columns[2:6]:
    re = pairwise_tukeyhsd(movers.dropna()[x],movers.dropna().group)
    print re
    p=psturng(np.abs(re.meandiffs / re.std_pairs), len(re.groupsunique), re.df_total)
    m.append(p)
##############one-way ANOVA test for CALM profiles
formula = 'MrX~ C(c_calm)'
model = ols(formula, clabels.dropna()).fit()
aov_table_calm = sm.stats.anova_lm(model, typ=2)
print aov_table_calm
for x in clabels.columns[0:4]:
    re_calm = pairwise_tukeyhsd(clabels.dropna()[x],clabels.dropna().c_calm)
    print re_calm
##############one-way ANOVA test for Pre profiles
for x in clabel_pre.columns[0:4]:
    re_pre = pairwise_tukeyhsd(clabel_pre.dropna()[x],clabel_pre.dropna().c_pre)
    p = psturng(np.abs(re_pre.meandiffs / re_pre.std_pairs), len(re_pre.groupsunique), re_pre.df_total)
    print re_pre,'\n', p
##############one-way ANOVA test for Post profiles
for x in clabel_post.columns[0:4]:
    re_post = pairwise_tukeyhsd(clabel_post.dropna()[x],clabel_post.dropna().c_post)
    p = psturng(np.abs(re_post.meandiffs / re_post.std_pairs), len(re_post.groupsunique), re_post.df_total)
    print re_pre,'\n', p
    
##############MANOVA for testing whether IQ modulate multivariate gain profiles
from statsmodels.multivariate.manova import MANOVA
manova_model = MANOVA(movers.dropna().iloc[:,2:6], movers.dropna().IQ)
manova_model.my_test()