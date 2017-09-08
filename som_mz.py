# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:49:15 2017

@author: mz
"""

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import sompy
import pandas as pd
from sompy.visualization.bmuhits import BmuHitsView

#read in awma data
xl = pd.read_excel('/Users/mz/Downloads/Awma_data.xlsx',sheetname=0,header=0)
names = list(xl.columns)
data = np.array(xl.values)
data = np.array(data,dtype='float')
mask=~np.any(np.isnan(data),axis=1)#get rid of missing data #or df.dropna()
data=data[mask]
#read in training data
xl2 = pd.read_excel('/Users/mz/Downloads/training data_4 tasks.xlsx',sheetname=0,header=0)#145 subjects
xl2 = pd.read_excel('/Users/mz/Downloads/training data_4 tasks+IQ.xlsx',sheetname=0,header=0) #124 subjects
names2 = list(xl2.columns)
training_data = np.array(xl2.values)
training_data = np.array(training_data,dtype='float')
mask=~np.any(np.isnan(training_data),axis=1)#get rid of missing data #or df.dropna()
training_data=training_data[mask]
data_pre0 = training_data[:,:4]
data_post0 = training_data[:,4:8]
data_pre = training_data[:,:4]
data_post = training_data[:,4:8]
name_pre = names2[:4]
name_post = names2[4:8]
name_general = ['DR','DM','BDR','MrX'] #A general name list for tasks
data_iq = training_data[:,8]

#for saving subset for predicting
m=np.random.randint(0,7,data.shape[0])
m.shape+=(1,)
mask1=np.any(m,axis=1)
mask2=~np.any(m,axis=1)
data_train = data[mask1]
data_new = data[mask2]
print (data_train.shape[0],data_new.shape[0])





#for initializing sm1 and sm2 with same pca
data_combined = np.append(data_pre, data_post,axis=0)
sm_combined.codebook.pca_linear_initialization(sm_combined._data)
pca=sm_combined.codebook.matrix
sm.codebook.matrix=pca
sm1.codebook.matrix=pca
sm2.codebook.matrix=pca
#training som
mapsize = [10,10]
sm = sompy.sompy.SOMFactory().build(data, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=names)
sm.train(n_job=1, verbose='info', train_rough_len=10, train_finetune_len=5)

sm1 = sompy.sompy.SOMFactory().build(data_pre, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=name_pre)
sm1.train(n_job=1, verbose='info', train_rough_len=10, train_finetune_len=5)

sm2 = sompy.sompy.SOMFactory().build(data_post, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=name_pre)
sm2.train(n_job=1, verbose='info', train_rough_len=10, train_finetune_len=5)

sm_combined = sompy.sompy.SOMFactory().build(data_combined, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=names)
sm_temporary = sompy.sompy.SOMFactory().build(data_pre, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=names)

#train multiple maps/middle stages
i=0; trainlen = 8
smlist = []
while i<trainlen:
    smlist.append(sompy.sompy.SOMFactory().build(data, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=names))
    smlist[i].train(n_job=1, verbose='info', train_rough_len=i+1, train_finetune_len=0)
    quantization_error = np.mean(smlist[i]._bmu[1])
    
    print ('map is trained with %d iteration' %(i+1))
    print ("Quantization error = %s" % (quantization_error))
    i+=1

    
topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))


    
i=0; trainlen = 8
while i<trainlen:
    i=7
    v = sompy.mapview.View2DPacked(20, 20, 'test',text_size=8)  
    v.show(smlist[i], what='codebook', cmap=None, col_sz=4)
    v.save('/Users/mz/Desktop/feature_map%d' %(i+1))
    i+=1
    
'''component map'''
v = sompy.mapview.View2DPacked(20, 20, 'test',text_size=8)  
v.show(sm2, what='codebook', cmap=None, col_sz=2) #which_dim='all' default
# v.save('/Users/mz/Desktop/feature_map')

#alternative way
view2D  = sompy.visualization.mapview.View2D(20,20,"rand data",text_size=10)
view2D.show(sm2, col_sz=2, which_dim="all", desnormalize=False)


#clustering feature map
from sklearn.cluster import KMeans #Sum of distances of samples to their closest cluster center.

n_cluster = 2; kmeans=[]; fig = plt.figure()
ticks = np.arange(0,10,1)
for i in range(data.shape[1]):
    kmeans.append( KMeans(n_cluster))
    X = sm2._normalizer.denormalize_by(sm2.data_raw[:,i], sm2.codebook.matrix[:,i])
    X.shape+=(1,)
    Z = kmeans[i].fit_predict(X)
    Z = Z.reshape(mapsize)
    ax=fig.add_subplot(2,2,i+1)
    ax.set_xticks(ticks,minor=True)
    ax.set_yticks(ticks,minor=True)
    ax.set_xlim(0,9)
    ax.set_ylim(0,9)
    ax.set_title(names[i])
    ax.imshow(Z[::-1], interpolation='nearest',
               aspect='auto', origin='lower',
               cmap=plt.cm.get_cmap('RdYlBu_r'),alpha = 0.8)

    
fig.subplots_adjust(left=0.2,right=0.8,bottom=0.1,top=0.9)
fig.show()


'''hits map'''
vhts  = sompy.visualization.bmuhits.BmuHitsView(2,2,"Hits Map",text_size=10)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=10, cmap="Greys", logaritmic=False)
#vhts.save('C:/Users/mz01/Desktop/BlahBlah/hitmap')

'''k-mean map'''
sm.cluster(3)
cluster_label= sm.cluster_labels
hits  = sompy.visualization.hitmap.HitMapView(10,10,"Clustering",text_size=8)
hits.show(sm)
hits.save('/Users/mz/Desktop/kmean')

#elbow method
from sklearn.cluster import KMeans
Ks = range(3, 10)
km = [KMeans(n_clusters=i) for i in Ks]
my_matrix = sm._normalizer.denormalize_by(sm.data_raw,sm.codebook.matrix)
[km[i].fit(my_matrix) for i in range(len(km))]
score = [km[i].inertia_ for i in range(len(km))]
plt.plot(Ks, score)

'''u-matrix'''
u = sompy.visualization.umatrix.UMatrixView(20, 20, 'umatrix', show_axis=True, text_size=8, show_text=True)
#This is the Umat value
UMAT  = u.build_u_matrix(sm, distance=1, row_normalized=False)
UMAT = u.show(sm, distance2=1, row_normalized=False, show_data=False, labels=False, contooor=False, blob=False)


"""""""""""""""""""""""""""""""""""""""""""""
added-on stuff
"""""""""""""""""""""""""""""""""""""""""""""
mat = sm.codebook.matrix #to get node weights
fmap = {}; dist = {}
for i in range(mat.shape[1]):
    fmap[i+1] = mat[:, i].reshape(mapsize[0], mapsize[1]) #weights of feature N

for i in range(len(fmap)): #get the Euclidean distance between features
    for j in range(len(fmap)-1-i):
        dist[((i+1)*10+i+j+2)] = np.linalg.norm (fmap[i+1]-fmap[i+j+2])

proj = sm.project_data(sm.data_raw)#to get the node each data is allocated

from collections import Counter
Counter(proj).items() #raw hitmap 
np.where(proj==12) #to see what data points is in one node

"""""""""""""""""""""""""""""""""""
how extreme data affect prediction accuracy
"""""""""""""""""""""""""""""""""""
#for testing the effect of extreme data on prediction performance
def data_extremity (dataset, label, target, som, num, step, plot=False):
    import math
    if plot==True:
        import seaborn as sns
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        sns.set_style({'axes.grid': True,
                       'font.family': [u'serif'],
                       'font.sans-serif': [u'Computer Modern Unicode'],})
    extremity_list = np.zeros([math.ceil((dataset.shape[0]-num)/step)+1, len(target)])
    for j in range(len(target)):
        data_sorted=dataset[dataset[:,target[j]].argsort()] #sort according to the target column
        i=0; ceiling = (math.ceil((dataset.shape[0]-num)/step))+1
        while i < ceiling:
            if i== (math.ceil((dataset.shape[0]-num)/step)):
                data_new1=data_sorted[-num:,:]
            else:
                data_new1=data_sorted[i*step:num+i*step,:]
            ind = np.arange(0, dataset.shape[1])
            indX = ind[ind != target[j]]
            real1 = data_new1[:,target[j]]
            target_feature = np.array([target[j]]) 
            given_feature = indX
            new_data1 = data_new1[:,indX]
            predicted_value1 = som.predict_by(new_data1,target[j], k=10) #gives out predicted value of given feature (assumed to be the last column of raw data)
            predicted_value1.shape+=(1,)
            real1.shape+=(1,)
            
            dif=np.mean(np.abs(predicted_value1-real1))
            extremity_list[i,j]=dif
            i=i+1

            #plot
        if plot == True:
            fig, ax1 = plt.subplots()

            ax1.scatter((np.arange(data_sorted.shape[0])),data_sorted[:,target[j]],marker='.',color='blue')
#            ax1.axvline(i*step,color='r',ls=':')
#            ax1.axvline(num+i*step,color='r',ls=':')
            ax1.set_xlabel('subjects')
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylabel('task scores', color='blue')
            ax1.tick_params('y', colors='blue')
            ax1.spines['left'].set_color('blue');
            ax2 = ax1.twinx()
#            ax2.axhline(dif,xmin=(i*step+100)/600,color='g')
            ax2.plot([(2*k*step+num)/2 for k in range(ceiling)], list(extremity_list[:,j]),'-',color='firebrick')
            ax2.plot([(2*k*step+num)/2 for k in range(ceiling)], list(extremity_list[:,j]),'o',color='firebrick')
            ax2.set_ylabel('mean absolute difference', color='firebrick')
            ax2.tick_params('y', colors='firebrick')
            ax2.set_yticks(np.arange(4,18, 2.0))
            ax2.spines['right'].set_color('firebrick');ax2.spines['left'].set_color('blue')
            ax1.set_title(label[target[j]])
            ax2.grid(None)
#            ax2.annotate('%.1f' % dif, xy=(145,dif+0.5), textcoords='data',color='g')
            ax1.spines['top'].set_visible(False);ax2.spines['top'].set_visible(False)
            plt.tight_layout()
            plt.show()
            #fig.savefig('/Users/mz/Desktop/%s%d.png' %(label[target[j]],i))
    return extremity_list

extremity_list_awma= data_extremity(data,names,[0,1,2,3],sm,100,50, plot=True)

"""""""""
prediction
"""""""""
#a=np.random.normal(100,15,(10,3)) #generade random data for testing
#a=a.round()

def simple_prediction(data, som, K):
    target = [0,1,2,3];dif=[]; prediction=[]
    for j in range(len(target)):
        ind = np.arange(0, data.shape[1])
        indX = ind[ind != target[j]]
        real = data[:,target[j]]
        new_data= data[:,indX]
        target_feature = np.array([target[j]]) 
        given_feature = indX
        predicted_value = som.predict_by(new_data,target[j], k=K) #gives out predicted value of given feature (assumed to be the last column of raw data)
        
        if predicted_value.ndim ==1:   #incase of predicting one column
            predicted_value.shape+=(1,)
            
        if real.ndim==1:
            real.shape+=(1,)
        dif.append(((predicted_value-real)))
        prediction.append(predicted_value)
        print(np.mean(np.abs(predicted_value-real)))
    return dif, prediction
dif0, prediction_list=simple_prediction(data_new,sm,13)
dif1, prediction_list1=simple_prediction(data_pre,sm,13)
dif2, prediction_list2=simple_prediction(data_post,sm,13)

"""
prediction of >=2 features using sklearn knn
"""
from sklearn import neighbors
target_feature = np.array([2]) #column numbers of features to be predicted [n x p]
given_feature = np.array([0,1,3]) #column numbers of features to be used in prediction [n x q]
x_train = sm.codebook.matrix[:, given_feature]
y_train = sm.codebook.matrix[:, target_feature] 

clf = neighbors.KNeighborsRegressor(10, weights='distance')
clf.fit(x_train, y_train)
real = data_new[:,target_feature]
new_data = data_new[:,given_feature] #assign to-be-predicted data here [n x q]
normalized_new = sm._normalizer.normalize_by(sm.data_raw[:, given_feature], new_data)
predicted = clf.predict(normalized_new)
predicted_value = sm._normalizer.denormalize_by(sm.data_raw[:, target_feature], predicted)

acc=(1-np.abs((predicted_value-real)/real))*100
np.mean(acc)



#need to wrap this into a function
'''evaluation of prediction error'''

#iteration = 4 #total number of prediction
DFs = []
accuracy = [] 
K=10
key = ['task','median_acc', 'mean_acc','std_acc','p_mean','p_std','real_m','real_std','diff']
workbook = dict((k, []) for k in key)
target = [0,1,2,3]

for j in range(len(target)):
    ind = np.arange(0, data.shape[1])
    indX = ind[ind != target[j]]
    real = data_new[:,target[j]]
    target_feature = np.array([target[j]]) 
    given_feature = indX
    new_data= data_new[:,indX]
    workbook['real_std'].append(np.std(real))
    workbook['real_m'].append(np.mean(real))
    predicted_value = sm.predict_by(new_data,target[j], k=K)
    if predicted_value.ndim ==1:   #incase of predicting one column
        predicted_value.shape+=(1,)
    if real.ndim==1:
        real.shape+=(1,)
#    workbook['k'].append(K[j])
    workbook['p_mean'].append(np.mean(predicted_value))
    workbook['p_std'].append(np.std(predicted_value))
    for i in range(1):
        acc=np.zeros(real.shape)
        acc[:,i]=(1-np.abs((predicted_value[:,i]-real[:,i])/real[:,i]))*100
        accuracy.append(acc)
#        print ('predicted task:' + name_pre[target_feature[i]])
#        print ('median accuracy', np.median(accuracy[-1][:,i]))
#        print ('mean accuracy', np.mean(accuracy[-1][:,i]))
#        print ('std accuracy', np.std(accuracy[-1][:,i]))
#        print ('min accuracy', np.min(accuracy[-1][:,i]))
#        print ('max accuracy', np.max(accuracy[-1][:,i]))
        DFs.append(pd.DataFrame({('True Value '+names[target_feature[i]]): real[:,i], 'Predicted Value':predicted_value[:,i]}))
        workbook['task'].append(names[target_feature[i]])
        workbook['median_acc'].append(np.median(accuracy[-1][:,i]))
        workbook['mean_acc'].append(np.mean(accuracy[-1][:,i]))
        workbook['std_acc'].append(np.std(accuracy[-1][:,i]))
        workbook['diff'].append(np.mean(np.abs(predicted_value-real)))
#        fig = plt.figure(); 
#        
#        DFs[i].plot(DFs[i].index,DFs[i].columns[:],
#                       label=names[target_feature[i]],colormap='jet',x_compat=True,style='.-'); 
#        plt.legend(loc='best',bbox_to_anchor = (1.0, 1.0),fontsize = 'medium')
#        plt.ylabel('values')
#        font = {'size'   : 12}
#        plt.rc('font', **font)
#        fig.set_size_inches(10,10)

#write results to a csv file
csv=pd.DataFrame.from_dict(workbook)
csv.to_csv("/Users/mz/Desktop/prediction.csv")



"""permutation of difference between predicted and real value"""
#should be done with k that produces the best prediction results
#1.train som with data_train 2.bootstrap subset from data_new per puermutation 
#3.generate prediction 4.calculate difference
from scipy import stats

repetition=100 # Number of times for resampling subjects
repetition2 = 1000 #Number of times for shuffling group membership (real & predicted value)
dif_list=[]
target = [0,1,2,3];

m1= np.zeros([1,50])#creating a mask that resample 30 subjects from data_new each time
m2 = np.ones([1,data_new.shape[0]-50])
m3 = np.append(m1,m2)
m3.shape+=(1,)
m3 = ~np.any(m3,axis=1) 

for n in range(len(target)):
    dif=[];  
    ind = np.arange(0, data.shape[1])
    indX = ind[ind != target[n]]
    target_feature = np.array([target[n]]) 
    given_feature = indX
    for i in range(repetition):    
        data_subset = data_new[np.random.permutation(m3)] #bootstrapping
        real = data_subset[:,target[n]]
        new_data= data_subset[:,indX]   
    #    x_train = sm.codebook.matrix[:, given_feature]
    #    y_train = sm.codebook.matrix[:, target_feature] 
    #    forest = RandomForestRegressor(n_estimators = 5000,random_state = 0, n_jobs=-1)
    #    forest.fit(x_train,y_train)
    #    normalized_new = sm._normalizer.normalize_by(sm.data_raw[:, given_feature], new_data)
    #    predicted = forest.predict(normalized_new)
    #    t1  = sm._normalizer.denormalize_by(sm.data_raw[:, target_feature], predicted)
        #t1 = sm.predict_by(new_data,target, k=9) #wt='uniform' or 'distance'(default)
        t1 = sm.predict_by(new_data,target[n], k=5) 
        t2 = real
        #perm1 = np.random.permutation(t1)
        #perm2 = np.random.permutation(t2)
        #dif.append(np.abs(np.mean(perm1)-np.mean(perm2))) #would be the same if t1 isn't changed
        
        for j in range(repetition2):
            dif.append(np.mean(np.abs(np.random.permutation(t1)-t2))) #element by element substraction then average
    
    
    dif.sort()#dif needs to be a list, norm.pdf doesn't work well with np array
    dif_list.append(dif)

def plot_distribution(dif_list, dif_list2=None):
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_style({'axes.grid': False,
                       'font.family': [u'serif'],
                       'font.sans-serif': [u'Computer Modern Unicode'],})
    fig=plt.figure()
    
    for i in range(len(dif_list)):
        dif_list[i].sort()
        difmean = np.mean(dif_list[i])
        difstd=np.std(dif_list[i])
        pdf=stats.norm.pdf(dif_list[i],difmean,difstd) #probability density function
        fig.add_subplot(2, 2, i+1)
        plt.plot(dif_list[i],pdf,'-')
        plt.title(names[i])
        if dif_list2!=None:
            dif_list2[i].sort()
            difmean2 = np.mean(dif_list2[i])
            difstd2=np.std(dif_list2[i])
            pdf2=stats.norm.pdf(dif_list2[i],difmean2,difstd2)
            plt.plot(dif_list2[i],pdf,'-')
        plt.hist(dif_list[i],bins=20,normed=True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle('Difference distribution')
    plt.show()

plot_distribution(dif_list)
#test mean prediction difference against the dif distribution
p=[]
for i in range(4):
    delta = np.mean(np.abs(dif2[i])) #np.mean(np.abs(predicted_value-real))
    diffCount = len(np.where(np.asarray(dif_list[i]) >=delta)[0])
    p.append(1.0 - (float(diffCount)/float(len(dif_list[i]))))
print (p)

    
#permutation 2-sample t-test  
numSamples = 1000    
delta = t1.mean() - t2.mean()
estimates = np.array(map(lambda x: run_permutation_test(t1,t2,t1.size),range(numSamples)))
diffCount = len(np.where(estimates <=delta)[0])
p = 1.0 - (float(diffCount)/float(numSamples))
print (p)

#variance explained
from sklearn.metrics import explained_variance_score, mean_absolute_error
explained_variance_score(data_post,data_pre,multioutput='raw_values')
mean_absolute_error(real,predicted_value,multioutput='raw_values')

#a little ttest
from scipy import stats
from math import sqrt
[t,p] = stats.ttest_ind(predicted_value,real)
print ('t-score: ', t[0] , '; p_value: ' , p[0])

i=0
while i < len(DFs):
    t1 = DFs[i][DFs[i].columns[0]]
    t2 = DFs[i][DFs[i].columns[1]]
    [t,p] = stats.ttest_ind(t1,t2)
    print ('t-score: ', t , '; p_value: ' , p)
    #effect size (cohen's d)
    d= (np.mean(t1)-np.mean(t2))/sqrt((np.std(t1)**2+np.std(t2)**2)/2)
    print ('effect size: ', d)
    i +=1


"""
#Representation dissimilarity matrix of feature maps
"""
def RDM(som,labels=['DR','DM','BDR','MrX'],show=True):
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_style({'axes.grid': False,
                   'font.family': [u'serif'],
                   'font.sans-serif': [u'Computer Modern Unicode'],})
    cov_matrix= 1-np.corrcoef(som.codebook.matrix.T) #Pearson's r. each row is a variable, colomn an observation
    di=np.diag_indices(cov_matrix.shape[0],ndim=2)
    cov_matrix[di]=0
    if show==True:
        ax=plt.subplot()
#        cmap = sns.diverging_palette(240, 10, as_cmap=True) # Generate a custom diverging colormap
        sns.heatmap(cov_matrix,annot=True,vmin=0,vmax=1, square=True,cmap='Reds')
#        plt.pcolor(cov_matrix[::-1],vmin=0,vmax=1,cmap='RdYlBu_r')
#        plt.colorbar()
        ax.set_xticks(np.arange(0.5,0.5+cov_matrix.shape[0],1))
        ax.set_yticks(np.arange(0.5,0.5+cov_matrix.shape[0],1))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels[::-1])
        plt.title('Representation dissimilarity matrix')
        plt.show()
    
    return cov_matrix
    
cov=RDM(sm,show=True)
cov1=RDM(sm1,labels=name_pre,show=True)
cov2=RDM(sm2,labels=name_pre,show=True)

ax=plt.subplot()
cmap = sns.diverging_palette(240, 10, as_cmap=True) # Generate a custom diverging colormap
sns.heatmap(cov_top,annot=True,vmin=0,vmax=1, square=True,cmap='Reds')
ax.set_xticks([0.5, 1.5, 2.5,3.5])
ax.set_yticks([0.5, 1.5, 2.5,3.5])
ax.set_xticklabels(name_general)
ax.set_yticklabels(name_general[::-1])
plt.title('Top-down RDM - complex span')
plt.show()

"""
cakcukate RDM using mutual information
"""
# number of bins affects results 
def calc_MI(x, y, bins):
    from sklearn.metrics import mutual_info_score
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

"""
comparing RDMs
"""

def RDM_compare(m1,m2,repetition):
    from scipy.stats import spearmanr
    rdm1 = m1[np.triu_indices(m1.shape[0])]; rdm1.shape+=(1,)
    rdm2 = m2[np.triu_indices(m2.shape[0])]; rdm2.shape+=(1,)
    R = spearmanr(rdm1,rdm2)[0]
    r_list=[]
    for i in range(repetition):
        idx = np.random.permutation(rdm1)
        r =spearmanr(idx,rdm2)[0]
        r_list.append(r)
    diffCount = len(np.where(np.asarray(r_list) <= R)[0])
    p = (1.0 - (float(diffCount)/float(len(r_list))))
    return p, R
        
RDM_compare(cov1,cov2,1000)

"""
To test whether two relatedness values are significantly different 
"""
def RDM_relatedness_test(m1, m2, m_constant):
    from scipy.stats import spearmanr
    p1, R1 = RDM_compare(m_constant,m1,1000)
    p2, R2 = RDM_compare(m_constant,m2,1000)
    rdm1 = m1[np.triu_indices(m1.shape[0])]; rdm1.shape+=(1,)
    rdm2 = m2[np.triu_indices(m2.shape[0])]; rdm2.shape+=(1,)
    rdm = np.append(rdm1,rdm2,axis=0)
    m_constant=m_constant[np.triu_indices(m_constant.shape[0])]; m_constant.shape+=(1,)
    IND = np.arange(0,rdm.shape[0])
    r_list=[]
    for i in range(10000):
        ind=np.random.permutation(IND)
        temp1 = rdm[ind[0:10]]
        temp2 = rdm[ind[10:]]
        r1 =spearmanr(m_constant,temp1)[0]
        r2 =spearmanr(m_constant,temp2)[0]
        r_list.append(r2-r1)
    diffCount = len(np.where(np.asarray(r_list) <= (R2-R1))[0])
    p = (1.0 - (float(diffCount)/float(len(r_list))))
    return p, R1, R2
    
RDM_relatedness_test(cov1,cov2,cov_top)   

"""
plot spring graph
"""

def RDM_springplot(m1,m2,m3=None):
    import networkx as nx
    G=nx.from_numpy_matrix(m1)   # this would be the dissimilarity matrix from your analysis
    #fixed the first node at (0.5,0.5)
    position=nx.spring_layout(G,fixed=[1],pos={0:[0.5,0.5],1:[0,0],2:[0,1] ,3:[1,1]})
    nx.draw_networkx_nodes(G, position, node_size=200,label='pre',alpha=0.4)
    nx.draw_networkx_labels(G,position,labels={0:'DR',1:'DM',2:'BDR' ,3:'MrX'})
    nx.draw_networkx_edges(G, position, width=0.5,alpha=0.5,edge_shape='-')

    
    G2=nx.from_numpy_matrix(m2)   
    position2=nx.spring_layout(G2,fixed=[1],pos={0:[0.5,0.5],1:[0,0],2:[0,1] ,3:[1,1]})
    nx.draw_networkx_nodes(G2, position2, node_color='g',node_shape='^',alpha=0.4,node_size=200,label='post')
    nx.draw_networkx_labels(G2,position2,labels={0:'DR',1:'DM',2:'BDR' ,3:'MrX'})
    nx.draw_networkx_edges(G2, position2, width=0.5,alpha=0.5)
    if m3!=None:
        G3=nx.from_numpy_matrix(m3)   
        position3=nx.spring_layout(G3,fixed=[0],pos={0:[0.5,0.5],1:[0,0],2:[0,1] ,3:[1,1]})
        nx.draw_networkx_nodes(G3, position3, node_color='b',node_shape=(5,1),node_size=200,alpha=0.4)
        nx.draw_networkx_labels(G3,position3,labels={0:'DR',1:'DM',2:'BDR' ,3:'MrX'})
    plt.axis('off')
    plt.legend()
RDM_springplot(1-cov1,1-cov2) #1-RDM

"""
top-down RDMs
"""
#stimuli type
cov_top = np.array([[0, 1, 0, 1],[1, 0, 1, 0],[0, 1, 0, 1],[1, 0, 1, 0]]) #0.65 0.88
# serial recall/wm component
cov_top = np.array([[0, 0, 1, 1],[0, 0, 1, 1],[1, 1, 0, 0],[1, 1, 0, 0]]) #0.58  0.51
#complex span
cov_top = np.array([[0, 0, 0, 1],[0, 0, 0, 1],[0, 0, 0, 1],[1, 1, 1, 0]]) #0.27(not sig) 0.43(almost sig 0.08)

RDM_compare(cov_top,cov1,1000) 
RDM_compare(cov_top,cov2,1000)



"""
get individual movement on map clusters
"""
sm.cluster(3)
sm.cluster_labels[np.where(sm.cluster_labels==1)]=3 #rename cluster
sm.cluster_labels[np.where(sm.cluster_labels==0)]=4
sm.cluster_labels[np.where(sm.cluster_labels==4)]=1
hits  = sompy.visualization.hitmap.HitMapView(10,10,"Clustering",text_size=8)
hits.show(sm)
bmu_pre= sm.project_data(data_pre) #cluster new data to the trained map
bmu_post = sm.project_data(data_post)
coor_pre =  sm.cluster_labels[bmu_pre]; coor_pre.shape+=(1,)
coor_post = sm.cluster_labels[bmu_post];coor_post.shape+=(1,)
coor_comb = np.append(coor_pre,coor_post,axis=1)
coor_calm = sm.cluster_labels[sm.project_data(data)]; coor_calm.shape+=(1,)

def arrayCount(arr,target):#row by row comparison with a given tuple (x,y)
    from scipy.stats import sem
    count=0
    for i in range(arr.shape[0]):
        if tuple(arr[i,:])==target:
            count+=1
    if count<=2:
        print('no data in this category %s' %(target,))
        return count
    else:
        a = np.where((arr == target).all(axis=1))[0]
        m_pre = np.mean(data_pre[a,:],axis=0)
        m_post=  np.mean(data_post[a,:],axis=0)
        sem_pre = 1.96* sem(data_pre[a,:],axis=0) #95% confidence interval
        sem_post = 1.96* sem(data_post[a,:],axis=0)
        m_change = np.mean(data_post[a,:]-data_pre[a,:],axis=0)
        sem_change = 1.96* sem(data_post[a,:]-data_pre[a,:],axis=0)
        label = np.array(target)
        print('%s in this category %s' %(count,(target,)))
        return count, np.round(m_pre,decimals=0),np.round(m_post,decimals=0),np.round(sem_pre,decimals=0),np.round(sem_post,decimals=0), label,m_change, sem_change

import itertools
group = list(itertools.product([0,1,2],repeat=1)) #or repeat = 1
count=[];m_pre=[];m_post=[];std_pre=[];std_post=[];labels=[];m_change=[];std_change=[]
for i,combination in enumerate(group):
    try:
        a,b,c,d,e,f,g,h = arrayCount(coor_pre,combination) #coor_pre, coor_post
        count.append(a);m_pre.append(b);m_post.append(c);std_pre.append(d);std_post.append(e); labels.append(f);m_change.append(g);std_change.append(h)
    except Exception:
        pass

#to get subjects who move to the same cluster
b0=np.logical_or(coor_comb == (1,0),  coor_comb==(2,0))
b1=np.logical_or(coor_comb == (2,1),  coor_comb==(0,1))
b2=np.logical_or(coor_comb == (1,2),  coor_comb==(2,2))
a0=np.where((b0==[True,True]).all(axis=1))[0]
a1=np.where((b1==[True,True]).all(axis=1))[0]
a2=np.where((b2==[True,True]).all(axis=1))[0]
iq0=data_iq[a0]
iq1=data_iq[a1]
iq2=data_iq[a2]        
 
"""
line plot
"""

def make_SuePlot(mean, error, binarized_results=None, labels=None, legend=None, name='Sue_Plot',ylim=True):
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_style({'axes.grid': False,
                   'font.family': [u'serif'],
                   'font.sans-serif': [u'Computer Modern Unicode'],})
    colours = ['turquoise', 'gold', 'firebrick', 'limegreen', 'darkorange', 'blue','darkorchid']
#    colours = [ 'firebrick', 'limegreen', 'darkorange', 'blue','darkorchid']
#    colours = [ 'darkorange', 'blue','darkorchid']
#    plt.figure(figsize=(width, width*0.8), dpi=600,frameon=False)
    plt.figure()
    for i in range(len(mean)):
        plt.errorbar((np.array(range(len(mean[i])))+0.04*i),mean[i], yerr=error[i], color=colours[i])
 
    plt.xlim([-0.5,len(mean[0])-0.5])
    if ylim==True: plt.ylim([60,140])
    else: plt.ylim([-10,40])
    plt.xticks(range(0,len(mean[0])))
    if legend!=None:plt.legend([list(community) for community in legend ], frameon=True, loc=1)
    else: plt.legend([ 'C' + str(community+1) for community in range(len(mean)) ], frameon=True, loc=1)
    plt.ylabel('stantard_scores')
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticklabels(labels, rotation=45);
    plt.title(name)
    if binarized_results!=None:
        plt.figure()
        cmap = sns.light_palette("red", as_cmap=True) 
        combinations = list(itertools.combinations(legend, 2))
        new_style = {'grid': False}
        plt.rc('axes', **new_style)
        plt.imshow(binarized_results,
                  interpolation = 'none',
                  cmap=cmap,
                  aspect='equal')
        plt.yticks(range(0,len(combinations)))
        plt.xticks(range(0,len(mean[0])))
        plt.ylabel('contrast results')
        plt.ylim([-0.5, len(combinations)-0.5])
        plt.xlim([-0.5,len(mean[0])-0.5])
        ax=plt.gca()
        ax.set_xticklabels(labels, rotation=45);
        ax.set_yticklabels([str(combination[0]) + ' v ' + str(combination[1]) for combination in combinations], rotation=0);
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #    plt.tight_layout()
    #    plt.savefig(outfolder + 'Sue_plot' + name + '.png', dpi=300)

make_SuePlot(m_pre,std_pre, labels=name_pre,legend=labels, name='Pre-training')
make_SuePlot(m_post,std_post,labels=name_post,legend=labels, name='Post-training')
make_SuePlot(m_pre[0:2],std_pre[0:2],labels=name_pre,legend=labels[0:2], name='Pre cluster 0')
make_SuePlot(m_post[0:2],std_post[0:2],labels=name_post,legend=labels[0:2], name='Post cluster 0')
make_SuePlot(m_pre[2:4],std_pre[2:4],labels=name_pre,legend=labels[2:4], name='Pre cluster 1')
make_SuePlot(m_post[2:4],std_post[2:4],labels=name_post,legend=labels[2:4], name='Post cluster 1')
make_SuePlot(m_pre[4:],std_pre[4:],labels=name_pre,legend=labels[4:], name='Pre cluster 2')
make_SuePlot(m_post[4:],std_post[4:],labels=name_post,legend=labels[4:], name='Post cluster 2')
make_SuePlot(m_change[4:],std_change[4:],labels=name_general,legend=labels[4:], name='change profile', ylim=False)
make_SuePlot(m_change[0:2],std_change[0:2],labels=name_general,legend=labels[0:2], name='change profile', ylim=False)
make_SuePlot(m_change[2:4],std_change[2:4],labels=name_general,legend=labels[2:4], name='change profile', ylim=False)
make_SuePlot([m_change[0],m_change[3]],[std_change[0],std_change[3]],labels=name_general,legend=[labels[0],labels[3]], name='change profile', ylim=False)
m_change = np.mean(data_post[a0,:]-data_pre[a0,:],axis=0)
sem_change = 1.96* sem(data_post[a0,:]-data_pre[a0,:],axis=0)
m_change1 = np.mean(data_post[a1,:]-data_pre[a1,:],axis=0)
sem_change1 = 1.96* sem(data_post[a1,:]-data_pre[a1,:],axis=0)
m_change2 = np.mean(data_post[a2,:]-data_pre[a2,:],axis=0)
sem_change2 = 1.96* sem(data_post[a2,:]-data_pre[a2,:],axis=0)
m_change3=[]
m_change3.append(m_change)
m_change3.append(m_change1)
m_change3.append(m_change2)
sem_change3=[]
sem_change3.append(sem_change)
sem_change3.append(sem_change1)
sem_change3.append(sem_change2)

make_SuePlot(m_change3,sem_change3,labels=name_general,legend=labels, name='Post - Pre',ylim=False)


"""
linear regression prediction and associated test
"""
#generate regression coefficients
def multiple_reg (data, target):
    from sklearn import linear_model
    ind = np.arange(0, data.shape[1])
    indX = ind[ind != target]
    real = data[:,target]
    new_data= data[:,indX]
    reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    reg.fit (new_data,real)
    
    return np.insert(reg.coef_,0,reg.intercept_)

coef0 = multiple_reg(data,0)
coef1 = multiple_reg(data,1)
coef2= multiple_reg(data,2)
coef3 = multiple_reg(data,3)

    
#using linear multiple regression for prediction
#as a standard baseline to compare SOM prediction with
def linear_reg(data, target, cof):
    ind = np.arange(0, data.shape[1])
    indX = ind[ind != target]
    real = data[:,target]
    new_data= data[:,indX]
    predicted_value = cof[0]+cof[1]*new_data[:,0]+cof[2]*new_data[:,1]+cof[3]*new_data[:,2]
    real.shape+=(1,)
    predicted_value.shape+=(1,)  
    return ((predicted_value-real))
    #return np.mean(np.abs(predicted_value-real))
    
a1=linear_reg(data_new,0,multiple_reg(data_train,0)) #
a2=linear_reg(data_new,1,multiple_reg(data_train,1)) #
a3=linear_reg(data_new,2,multiple_reg(data_train,2)) #
a4=linear_reg(data_new,3,multiple_reg(data_train,3)) #

b1=linear_reg(data_pre,0,multiple_reg(data_train,0)) #
b2=linear_reg(data_pre,1,multiple_reg(data_train,1)) #
b3=linear_reg(data_pre,2,multiple_reg(data_train,2)) #
b4=linear_reg(data_pre,3,multiple_reg(data_train,3)) #

c1=linear_reg(data_post,0,multiple_reg(data_train,0)) #
c2=linear_reg(data_post,1,multiple_reg(data_train,1)) #
c3=linear_reg(data_post,2,multiple_reg(data_train,2)) #
c4=linear_reg(data_post,3,multiple_reg(data_train,3)) #
#
#41.5536548444
#15.979410228
#12.1954273635
#13.9056029368
#36.7106177169
#16.3599106406
#13.7732524753
#14.8711739322
#43.7914047881
#17.772701233
#14.6832604645
#16.9965798733

#t-test for prediction from regression and KNN
# (knn-true) vs. (regression-true)
from scipy.stats import ttest_ind
dif0, prediction_list=simple_prediction(data_new,sm,15)
dif1, prediction_list1=simple_prediction(data_pre,sm,15)
dif2, prediction_list2=simple_prediction(data_post,sm,15)

#12.1074427299
#11.9011974267
#8.40681301092
#11.9180914323
#11.3392037077
#12.2089200541
#10.2953609721
#12.2925984405
#13.5795607753
#18.8688969863
#12.7305104102
#14.1648976041


#permutation of two-sample t-test
#bootstrapping to create a distribution for each sample and returns a p value 
def perm_(a, b, repetition=10000,equl=True ): #equl=True if both sample sizes are equal
    dif_list=[]; mask1 = np.arange(a.shape[0]); mask2 = np.arange(b.shape[0])
    for i in range(repetition):
        temp = np.random.choice(mask1,a.shape)
        if equl==True: temp1=temp
        else: temp1=np.random.choice(mask2,b.shape)
        a_temp = np.mean(np.abs(a[temp]))
        b_temp = np.mean(np.abs(b[temp1]))
        dif_list.append(a_temp-b_temp)
    diffCount = len(np.where(np.asarray(dif_list) >=0)[0])
    p = (1.0 - (float(diffCount)/float(len(dif_list))))
    print (p)

perm_(a1, dif0[0])
perm_(a2, dif0[1])
perm_(a3, dif0[2])
perm_(a4, dif0[3])

perm_(b1, dif1[0])
perm_(b2, dif1[1])
perm_(b3, dif1[2])
perm_(b4, dif1[3])
perm_(c1, dif2[0])
perm_(c2, dif2[1])
perm_(c3, dif2[2])
perm_(c4, dif2[3])

perm_(iq1,iq0,equl=False)
 #one method is consistently better than the other by a very small margin
perm_(np.array(d_som[0]),np.array(d_reg[0])) #1
perm_(np.array(d_som[1]),np.array(d_reg[1])) #0
perm_(np.array(d_som[2]),np.array(d_reg[2])) #1
perm_(np.array(d_som[3]),np.array(d_reg[3])) #1

#0.0
#0.0021999999999999797
#0.8009999999999999
#0.006099999999999994

#0.0
#0.08079999999999998
#0.9997
#0.748

#0.0
#1.0
#1.0
#0.02859999999999996

#giving a dataset, return two permutated distribution of prediction accuracy, one is by som, the other by regression 
def difference_dis(data, proportion=15, rep=1000, method='som' ):
    repetition=rep # Number of times for resampling subjects
    dif_list1=[]; dif_list2=[]
    target = [0,1,2,3];
    N=int(data.shape[0]*proportion/100)
    m1= np.zeros([1,N])#creating a mask that resample 30 subjects from data_new each time
    m2 = np.ones([1,data.shape[0]-N])
    m3 = np.append(m1,m2)
    m3.shape+=(1,)
    m3 = ~np.any(m3,axis=1) 
    for n in range(len(target)):
        dif1=[];  dif2=[]
        ind = np.arange(0, data.shape[1])
        indX = ind[ind != target[n]]
        target_feature = np.array([target[n]]) 
        given_feature = indX
        for i in range(repetition):    
            M = np.random.permutation(m3)
            data_subset = data[M] #bootstrapping
            data_train=data[~M]
            real = data_subset[:,target[n]]
            new_data= data_subset[:,indX]   
            t1 = sm.predict_by(new_data,target[n], k=10) 
            t2 = real
            dif1.append(np.mean(np.abs(t1-t2))) 
            t_reg= linear_reg(data_subset,n,multiple_reg(data_train,n)) 
            dif2.append(np.mean(np.abs(t_reg)))
        dif_list1.append(dif1); dif_list2.append(dif2)

    if method =='both': return dif_list1, dif_list2
    if method =='som': return dif_list1
    if method=='regression':return dif_list2 
    
d_som, d_reg = difference_dis(data, proportion=15, rep=1000, method='both' )
d_som = difference_dis(data, proportion=15, rep=1000 )
plot_distribution(d_reg)
plot_distribution(d_som, d_reg)

