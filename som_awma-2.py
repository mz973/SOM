# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:49:15 2017

@author: mz
"""

import numpy as np
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
xl2 = pd.read_excel('/Users/mz/Downloads/training data_4 tasks.xlsx',sheetname=0,header=0)
names2 = list(xl2.columns)
training_data = np.array(xl2.values)
training_data = np.array(training_data,dtype='float')
mask=~np.any(np.isnan(training_data),axis=1)#get rid of missing data #or df.dropna()
training_data=training_data[mask]
data_pre = training_data[:,:4]
data_post = training_data[:,4:]
name_pre = names2[:4]
name_post = names2[4:]
name_general = ['DR','DM','BDR','MrX'] #A general name list for tasks

#for saving subset for predicting
m=np.random.randint(0,8,data.shape[0])
m.shape+=(1,)
mask1=np.any(m,axis=1)
mask2=~np.any(m,axis=1)
data_train = data[mask1]
data_new = data[mask2]
print (data_train.shape[0],data_new.shape[0])


#for testing the effect of extreme data on prediction performance
def data_extremity (dataset, label, target, som, num, step):
    import math
    extremity_list = np.zeros([math.ceil((dataset.shape[0]-num)/step)+1, len(target)])
    for j in range(len(target)):
        data_sorted=dataset[dataset[:,target[j]].argsort()] #sort according to the target column
        i=0; 
        while i < (math.ceil((dataset.shape[0]-num)/step))+1:
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
            predicted_value1 = som.predict_by(new_data1,target[j], k=9) #gives out predicted value of given feature (assumed to be the last column of raw data)
            predicted_value1.shape+=(1,)
            real1.shape+=(1,)
            
            dif=np.mean(np.abs(predicted_value1-real1))
            extremity_list[i,j]=dif
            #plot
            fig, ax1 = plt.subplots()
            ax1.scatter((np.arange(data_sorted.shape[0])),data_sorted[:,target[j]],marker='.')
            ax1.axvline(i*step,color='r',ls=':')
            ax1.axvline(num+i*step,color='r',ls=':')
            ax1.set_xlabel('subjects')
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylabel('raw_score', color='r')
            ax1.tick_params('y', colors='r')
            ax2 = ax1.twinx()
            ax2.axhline(dif,xmin=(i*step+20)/180,color='g')
            ax2.set_ylabel('mean_difference', color='g')
            ax2.tick_params('y', colors='g')
            ax2.set_yticks(np.arange(6,17, 2.0))
            ax1.set_title(label[target[j]])
            ax2.annotate('%.1f' % dif, xy=(145,dif+0.5), textcoords='data',color='g')
            i=i+1
            
            plt.tight_layout()
            plt.show()
            #fig.savefig('/Users/mz/Desktop/%s%d.png' %(label[target[j]],i))
    return extremity_list

extremity_list= data_extremity(data_post,name_post,[0,1,2,3],sm,100,15)




#training som
mapsize = [13,13]
sm = sompy.sompy.SOMFactory().build(data, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=names)
sm.train(n_job=1, verbose='info', train_rough_len=10, train_finetune_len=5)

sm1 = sompy.sompy.SOMFactory().build(data_pre, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=names)
sm1.train(n_job=1, verbose='info', train_rough_len=10, train_finetune_len=5)

sm2 = sompy.sompy.SOMFactory().build(data_post, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=names)
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
v.show(sm1, what='codebook', cmap=None, col_sz=4) #which_dim='all' default
# v.save('/Users/mz/Desktop/feature_map')

#alternative way
view2D  = sompy.visualization.mapview.View2D(20,20,"rand data",text_size=10)
view2D.show(sm, col_sz=2, which_dim="all", desnormalize=False)


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
i=2
sm.cluster(3)
cluster_label= sm.cluster_labels
hits  = sompy.visualization.hitmap.HitMapView(10,10,"Clustering",text_size=8)
hits.show(sm)
hits.save('/Users/mz/Desktop/kmean')


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

distance_matrix=sm.calculate_map_dist()
bmu_pre= sm.project_data(data_pre) #cluster new data to the trained map
bmu_post = sm.project_data(data_post)
coor_pre =  sm.cluster_labels[bmu_pre]; coor_pre.shape+=(1,)
coor_post = sm.cluster_labels[bmu_post];coor_post.shape+=(1,)
coor_comb = np.append(coor_pre,coor_post,axis=1)
migration = np.where((coor_pre-coor_post)!=0)
Counter(coor_pre).items()
Counter(coor_post).items() 


"""
prediction
"""
#a=np.random.normal(100,15,(10,3)) #generade random data for testing
#a=a.round()

def simple_prediction(data, som, K):
    target = [0,1,2,3];dif=[]
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
        dif.append(np.mean(np.abs(predicted_value-real)))
        print(np.mean(np.abs(predicted_value-real)))
    return dif
simple_prediction(data_new,sm,10)
dif1=simple_prediction(data_pre,sm,10)
dif2=simple_prediction(data_post,sm,10)

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



"""
prediction using random forest regression
Seem to generate comparable results to knn, but with benefit of ranking features
"""
from sklearn.ensemble import RandomForestRegressor
target = [0,1,2,3]

for j in range(len(target)):
    ind = np.arange(0, data.shape[1])
    indX = ind[ind != target[j]]
    real = data_new[:,target[j]]
    new_data= data_new[:,indX]
    target_feature = np.array([target[j]]) 
    given_feature = indX
    feat_labels = [names[i] for i in given_feature]
    x_train = sm.codebook.matrix[:, given_feature]
    y_train = sm.codebook.matrix[:, target_feature] 
    forest = RandomForestRegressor(n_estimators = 5000,random_state = 0, n_jobs=-1)
    forest.fit(x_train,y_train) #Note, random forest doesn't need standardised data, I've just added so I can run knn on later
    #prediction
    normalized_new = sm._normalizer.normalize_by(sm.data_raw[:, given_feature], new_data)
    predicted = forest.predict(normalized_new)
    predicted_value = sm._normalizer.denormalize_by(sm.data_raw[:, target_feature], predicted)
    if predicted_value.ndim ==1:   #incase of predicting one column
        predicted_value.shape+=(1,)
    if real.ndim==1:
        real.shape+=(1,)
    
        
    print(np.mean(np.abs(predicted_value-real)))
    
    #ranking features
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    print ('predicting: %s' %names[target[j]])
    for f in range(x_train.shape[1]):
        print('%2d) %-*s %f' % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

#plot decision tree    
from sklearn.tree import export_graphviz
import pydot
export_graphviz(forest.estimators_[0],out_file='/Users/mz/Desktop/tree.dot',feature_names=feat_labels,filled=True,rounded=True)
(graph,) = pydot.graph_from_dot_file('/Users/mz/Desktop/tree.dot')
graph.write_png('/Users/mz/Desktop/tree1.png')
        
        
# Plot the results
plt.figure()
s = 50
plt.scatter(np.arange(0,predicted_value.shape[0]),predicted_value[:,0] , c="navy", s=s, label="data")
plt.scatter(np.arange(0,predicted_value.shape[0]), real[:, 0], c="cornflowerblue", s=s, label="prediction")
plt.xlabel("")
plt.ylabel("scores")
plt.title("Random Forest Regression")
plt.legend()
plt.show()

   
#Plot feature ranking
plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), importances[indices], color= 'lightblue', align= 'center')
plt.xticks(range(x_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()


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



"""
evaluation of prediction using RANDOM FOREST
"""
DFs = []
accuracy = [] 
key = ['task','median_acc', 'mean_acc','std_acc','p_mean','p_std','real_m','real_std','diff']
workbook = dict((k, []) for k in key)
target = [0,1,2,3]

for j in range(len(target)):
    ind = np.arange(0, data.shape[1])
    indX = ind[ind != target[j]]
    real = data_post[:,target[j]]
    target_feature = np.array([target[j]]) 
    given_feature = indX
    new_data= data_post[:,indX]
    workbook['real_std'].append(np.std(real))
    workbook['real_m'].append(np.mean(real))
    feat_labels = [name_post[i] for i in given_feature]
    x_train = sm.codebook.matrix[:, given_feature]
    y_train = sm.codebook.matrix[:, target_feature] 
    forest = RandomForestRegressor(n_estimators = 5000,random_state = 0, n_jobs=-1)
    forest.fit(x_train,y_train) #Note, random forest doesn't need standardised data, I've just added so I can run knn on later
    #prediction
    normalized_new = sm._normalizer.normalize_by(sm.data_raw[:, given_feature], new_data)
    predicted = forest.predict(normalized_new)
    predicted_value = sm._normalizer.denormalize_by(sm.data_raw[:, target_feature], predicted)
    
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
        DFs.append(pd.DataFrame({('True Value '+name_post[target_feature[i]]): real[:,i], 'Predicted Value':predicted_value[:,i]}))
        workbook['task'].append(name_post[target_feature[i]])
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
csv.to_csv("/Users/mz/Desktop/prediction_post.csv")




"""permutation of difference between predicted and real value"""
#should be done with k that produces the best prediction results
#1.train som with data_train 2.bootstrap subset from data_new per puermutation 
#3.generate prediction 4.calculate difference
from scipy import stats

repetition=100 # Number of times for resampling subjects
repetition2 = 1000 #Number of times for shuffling group membership (real & predicted value)
dif_list=[]
target = [0,1,2,3];

m1= np.zeros([1,53])#creating a mask that resample 30 subjects from data_new each time
m2 = np.ones([1,data_new.shape[0]-53])
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
        t1 = sm.predict_by(new_data,target[n], k=10) 
        t2 = real
        #perm1 = np.random.permutation(t1)
        #perm2 = np.random.permutation(t2)
        #dif.append(np.abs(np.mean(perm1)-np.mean(perm2))) #would be the same if t1 isn't changed
        
        for j in range(repetition2):
            dif.append(np.mean(np.abs(np.random.permutation(t1)-t2))) #element by element substraction then average
    
    
    dif.sort()#dif needs to be a list, norm.pdf doesn't work well with np array
    dif_list.append(dif)

import seaborn as sns
sns.set_style("whitegrid")
sns.set_style({'axes.grid': False,
                   'font.family': [u'serif'],
                   'font.sans-serif': [u'Computer Modern Unicode'],})
fig=plt.figure()
for i in range(len(dif_list)):
    difmean = np.mean(dif_list[i])
    difstd=np.std(dif_list[i])
    pdf=stats.norm.pdf(dif_list[i],difmean,difstd) #probability density function
    fig.add_subplot(2, 2, i+1)
    plt.plot(dif_list[i],pdf,'-')
    plt.title(names[i])
    plt.hist(dif_list[i],bins=20,normed=True)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.suptitle('Difference distribution')
plt.show()

#test mean prediction difference against the dif distribution
p=[]
for i in range(4):
    delta = dif2[i] #np.mean(np.abs(predicted_value-real))
    diffCount = len(np.where(np.asarray(dif_list[i]) >=delta)[0])
    p.append(1.0 - (float(diffCount)/float(len(dif_list[i]))))


def run_permutation_test(x,y,size):
    pooled = np.hstack([x,y])
    np.random.shuffle(pooled) #(mix up group membership)
    pooled1 = pooled[:size]
    pooled2 = pooled[-size:]
    return (pooled1.mean() - pooled2.mean())#get test stats
    
#permutation 2-sample t-test  
numSamples = 1000    
delta = t1.mean() - t2.mean()
estimates = np.array(map(lambda x: run_permutation_test(t1,t2,t1.size),range(numSamples)))
diffCount = len(np.where(estimates <=delta)[0])
p = 1.0 - (float(diffCount)/float(numSamples))
print (p)

#variance explained
from sklearn.metrics import explained_variance_score, mean_absolute_error
explained_variance_score(real,predicted_value,multioutput='raw_values')
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
    
    if show==True:
        ax=plt.subplot()
        cmap = sns.diverging_palette(240, 10, as_cmap=True) # Generate a custom diverging colormap
        sns.heatmap(cov_matrix,vmin=0,vmax=1, square=True,cmap=cmap)
#        plt.pcolor(cov_matrix[::-1],vmin=0,vmax=1,cmap='RdYlBu_r')
#        plt.colorbar()
        ax.set_xticks([0.5, 1.5, 2.5,3.5])
        ax.set_yticks([0.5, 1.5, 2.5,3.5])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels[::-1])
        plt.title('Representation dissimilarity matrix')
        plt.show()
    
    return cov_matrix
    
cov=RDM(sm2,show=True)

"""
comparing RDMs
"""

def RDM_compare(m1,m2,repetition):
    from scipy.stats import spearmanr
    rdm1 = m1[np.triu_indices(m1.shape[0])]; rdm1.shape+=(1,)
    rdm2 = m2[np.triu_indices(m2.shape[0])]; rdm2.shape+=(1,)
    
    r_list=[]
    for i in range(repetition):
        idx = np.random.choice(rdm1.shape[0], rdm1.shape[0])
        temp1 = rdm1[idx,:]
        temp2 = rdm2[idx,:]
        r =spearmanr(temp1,temp2)[0]
        r_list.append(r)
    return np.mean(r_list)
        
RDM_compare(cov,cov2,100)

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
get individual movement on map clusters
"""
def arrayCount(arr,target):#row by row comparison with a given tuple (x,y)
    count=0
    for i in range(arr.shape[0]):
        if tuple(arr[i,:])==target:
            count+=1
    if count==0:
        print('no data in this category %s' %(target,))
        return count
    else:
        a = np.where((arr == target).all(axis=1))[0]
        m_pre = np.mean(data_pre[a,:],axis=0)
        m_post=  np.mean(data_post[a,:],axis=0)
        std_pre = np.std(data_pre[a,:],axis=0)
        std_post = np.std(data_post[a,:],axis=0)
        label = np.array(target)
        return count, np.round(m_pre,decimals=0),np.round(m_post,decimals=0),np.round(std_pre,decimals=0),np.round(std_post,decimals=0), label

import itertools
group = list(itertools.product([0,1,2],repeat=2)) #or repeat = 1
count=[];m_pre=[];m_post=[];std_pre=[];std_post=[];labels=[]
for i,combination in enumerate(group):
    try:
        a,b,c,d,e,f = arrayCount(coor_comb,combination)
        count.append(a);m_pre.append(b);m_post.append(c);std_pre.append(d);std_post.append(e); labels.append(f)
    except Exception:
        pass
            
 
"""
line plot
"""

def make_SuePlot(mean, error, labels=None, legend=None, name='Sue_Plot'):
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_style({'axes.grid': False,
                   'font.family': [u'serif'],
                   'font.sans-serif': [u'Computer Modern Unicode'],})
#   colours = ['turquoise', 'gold', 'firebrick', 'limegreen', 'darkorange', 'blue','darkorchid']
#    colours = [ 'firebrick', 'limegreen', 'darkorange', 'blue','darkorchid']
    colours = [ 'darkorange', 'blue','darkorchid']

    #plt.figure(figsize=(width, width*0.8), dpi=600,frameon=False)
 
    
    for i in range(len(mean)):
        plt.errorbar((np.array(range(len(mean[i])))+0.03*i),mean[i], yerr=error[i], color=colours[i])
 
    plt.xlim([-0.5,len(mean[0])-0.5])
    plt.ylim([60,140])
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
#    plt.subplot(2,1,2)
#    combinations = list(itertools.combinations(np.unique(community_affiliation), 2))
# 
#    new_style = {'grid': False}
#    matplotlib.rc('axes', **new_style)
#    plt.imshow(binarized_results,
#              interpolation = 'none',
#              cmap=LinearSegmentedColormap.from_list('mycmap', [(0, 'lightgray'), (1, 'orangered')]))
#    plt.yticks(np.arange(0,len(combinations)))
#    plt.xticks(np.arange(0, len(measures)))
#    plt.ylabel('contrast results')
# 
#    ax = plt.gca()
#    ax.set_yticklabels([str(combination[0]) + ' v ' + str(combination[1]) for combination in combinations], rotation=0);
#    ax.set_xticklabels(labels, rotation=90);
#    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#    plt.savefig(outfolder + 'Sue_plot' + name + '.png', dpi=300)

make_SuePlot(m_pre,std_pre,labels=name_pre,legend=labels, name='Pre cluster')
make_SuePlot(m_post,std_post,labels=name_post,legend=labels, name='Post cluster')
make_SuePlot(m_pre[0:2],std_pre[0:2],labels=name_pre,legend=labels[0:2], name='Pre cluster 0')
make_SuePlot(m_post[0:2],std_post[0:2],labels=name_post,legend=labels[0:2], name='Post cluster 0')
make_SuePlot(m_pre[2:4],std_pre[2:4],labels=name_pre,legend=labels[2:4], name='Pre cluster 1')
make_SuePlot(m_post[2:4],std_post[2:4],labels=name_post,legend=labels[2:4], name='Post cluster 1')
make_SuePlot(m_pre[4:],std_pre[4:],labels=name_pre,legend=labels[4:], name='Pre cluster 2')
make_SuePlot(m_post[4:],std_post[4:],labels=name_post,legend=labels[4:], name='Post cluster 2')

"""Keep track of the position of test sample in relation to the whole distribution"""
from scipy import stats


data_train_copy=data_train.copy()
for i in range(data_train.shape[1]):
    plt.subplot(2,2,i+1)
    data_train_copy[:,i].sort()
    M = np.mean(data_train_copy[:,i])
    STD = np.std(data_train_copy[:,i])
    PDF = stats.norm.pdf(data_train_copy[:,i],M,STD)
    #plt.scatter((np.arange(data_sorted.shape[0])),data_sorted[:,0])
    plt.plot(data_train_copy[:,i],PDF,'-')
    plt.hist(data_new[:,i],bins=10,normed=True)

#using linear multiple regression for prediction
#as a standard baseline to compare SOM prediction with
def linear_reg(data, target, cof):
    ind = np.arange(0, data.shape[1])
    indX = ind[ind != target]
    real = data[:,target]
    new_data= data[:,indX]
    predicted_value = cof[0]+cof[1]*new_data[:,0]+cof[2]*new_data[:,1]+cof[2]*new_data[:,2]
    print (np.mean(np.abs(predicted_value-real)))

    return predicted_value
a=linear_reg(data_pre,0,[24.302,0.518,0.153,0.073]) #13.383
a=linear_reg(data_pre,1,[32.924,0.226,0.143,0.264]) #15.637
a=linear_reg(data_pre,2,[32.553,0.168,0.161,0.295]) #12.86
a=linear_reg(data_pew,3,[42.389,0.072,0.290,0.238]) #13.632
a=linear_reg(data_new,0,[24.302,0.518,0.153,0.073]) #12.876
a=linear_reg(data_new,1,[32.924,0.226,0.143,0.264]) #14.629
a=linear_reg(data_new,2,[32.553,0.168,0.161,0.295]) #14.545
a=linear_reg(data_new,3,[42.389,0.072,0.290,0.238]) #10.550
a=linear_reg(data_post,0,[24.302,0.518,0.153,0.073]) #16.916
a=linear_reg(data_post,1,[32.924,0.226,0.143,0.264]) #27.126
a=linear_reg(data_post,2,[32.553,0.168,0.161,0.295]) #18.403
a=linear_reg(data_post,3,[42.389,0.072,0.290,0.238]) #15.433









