#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# In[2]:


dataset = sio.loadmat('q4_dataset')


# In[3]:


sorted(dataset.keys())


# In[4]:


data_globals = dataset['__globals__']
data_header = dataset['__header__']
data_version = dataset['__version__']
data_class_labels = dataset['class_labels']
data_images = dataset['images']
data_inception_features = dataset['inception_features']

data_globals = np.array(data_globals)
data_header = np.array(data_header)
data_version = np.array(data_version)
data_class_labels = np.array(data_class_labels)
data_images = np.array(data_images)
data_inception_features = np.array(data_inception_features)


# In[5]:


print('The size of data_globals            : ' +str(np.shape(data_globals)))
print('The size of data_header             : ' +str(np.shape(data_header)))
print('The size of data_version            : ' +str(np.shape(data_version)))
print('The size of data_class_labels       : ' +str(np.shape(data_class_labels)))
print('The size of data_images             : ' +str(np.shape(data_images)))
print('The size of data_inception_features : ' +str(np.shape(data_inception_features)))


# In[6]:


def parameter_tuning_SVM(data, labels, k, C, gammas=None, kernel = 'linear'):
    
    start_time = time.time()
        
    best_params = {'best_c': None, 'best_gamma': None}
    best_score = 0
    best_c = None
    best_gamma = None
    all_scores = list()
    fold_scores = list()
    
    np.random.seed(8)
    sampleSize = np.shape(data)[0]
    featureSize = np.shape(data)[1]
    
    randomIndexes = np.random.permutation(sampleSize)
    data = data[randomIndexes]
    labels = labels[randomIndexes]

    fold_size = int(sampleSize / k)
    
    # Implement a 5-fold cross validation
    if (kernel == 'linear'):
        for c in C:
            
            classifier = OneVsRestClassifier(SVC(kernel='linear', C=c))
            fold_scores = list()
            
            for j in range(k):

                test_index_start = fold_size*j
                valid_index_start = fold_size*(j+1)
                train_index_start = fold_size*(j+2)

                test_indeces = np.arange(test_index_start, valid_index_start) % sampleSize
                valid_indeces = np.arange(valid_index_start, train_index_start) % sampleSize
                train_indeces = np.arange(train_index_start, sampleSize + test_index_start) % sampleSize

                test_data = data[test_indeces]
                test_labels = labels[test_indeces]

                valid_data = data[valid_indeces]
                valid_labels = labels[valid_indeces]

                train_data = data[train_indeces]
                train_labels = labels[train_indeces]

                model = classifier.fit(train_data, train_labels)
                score = classifier.score(valid_data, valid_labels)
                fold_scores.append(score)
             
            mean_fold_score = np.mean(fold_scores)
            all_scores.append(mean_fold_score)
            
            if(mean_fold_score > best_score):
                best_score = mean_fold_score
                best_c = c

        best_params['best_c'] = best_c
            
            
    else:
        
        for c in C:
            
            for gam in gammas:
                
                classifier = OneVsRestClassifier(SVC(kernel='rbf', C=c, gamma= gam))
                fold_scores = list()
                
                for j in range(k):

                    test_index_start = fold_size*j
                    valid_index_start = fold_size*(j+1)
                    train_index_start = fold_size*(j+2)

                    test_indeces = np.arange(test_index_start, valid_index_start) % sampleSize
                    valid_indeces = np.arange(valid_index_start, train_index_start) % sampleSize
                    train_indeces = np.arange(train_index_start, sampleSize + test_index_start) % sampleSize

                    test_data = data[test_indeces]
                    test_labels = labels[test_indeces]

                    valid_data = data[valid_indeces]
                    valid_labels = labels[valid_indeces]

                    train_data = data[train_indeces]
                    train_labels = labels[train_indeces]


                    model = classifier.fit(train_data, train_labels)
                    score = classifier.score(valid_data, valid_labels)
                    fold_scores.append(score)
                
                mean_fold_score = np.mean(fold_scores)
                all_scores.append(mean_fold_score)

                if(mean_fold_score > best_score):
                    best_score = mean_fold_score
                    best_c = c
                    best_gamma = gam

        best_params['best_c'] = best_c
        best_params['best_gamma'] = best_gamma

    end_time = time.time()
    time_elapsed = end_time - start_time
        
    return best_params, all_scores, time_elapsed


# In[7]:


# SVM (linear) parameter tuning

C_linear = [10**-6, 10**-4, 10**-2, 1, 10**1, 10**10]

best_params_linear, scores_linear, time_linear = parameter_tuning_SVM(data_inception_features, data_class_labels, 5, 
                                                                      C_linear, gammas = None, kernel = 'linear')


# In[8]:


print('Best parameters for SVM (linear) : ' +str(best_params_linear))
print('\n')
print('The scores for SVM (linear) : ' +str(scores_linear))
print('\n')
print('Time elapsed on training SVM (linear) : ' +str(time_linear) +' seconds')


# In[9]:


figureNum = 0
x = ['10^-6', '10^-4', '10^-2', '1', '10^1', '10^10']
plt.figure(figureNum)
plt.bar(x, scores_linear)
plt.title('Mean accuracies for each C')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.show()


# In[10]:


# To train the SVM once more with the best parameters

def train_SVM(data, labels, k, best_c, best_gamma = None, kernel = 'linear'):
    f1_macro_folds = list()
    classification_scores = list()
    
    if(kernel == 'linear'):
        classifier = OneVsRestClassifier(SVC(kernel='linear', C=best_c))
    else:
        classifier = OneVsRestClassifier(SVC(kernel='rbf', C=best_c, gamma= best_gamma))

    np.random.seed(8)
    sampleSize = np.shape(data)[0]
    featureSize = np.shape(data)[1]
    
    randomIndexes = np.random.permutation(sampleSize)
    data = data[randomIndexes]
    labels = labels[randomIndexes]
    
    fold_size = int(sampleSize / k)
    
    for j in range(k):
        
        test_index_start = fold_size*j
        train_index_start = fold_size*(j+1)

        test_indeces = np.arange(test_index_start, train_index_start) % sampleSize
        train_indeces = np.arange(train_index_start, sampleSize + test_index_start) % sampleSize
        
        test_data = data[test_indeces]
        test_labels = labels[test_indeces]
        
        train_data = data[train_indeces]
        train_labels = labels[train_indeces]
        
        model = classifier.fit(train_data, train_labels)
        y_pred = classifier.predict(test_data)
        scores = classification_report(test_labels, y_pred)
        
        f1_macro_folds.append(f1_score(test_labels, y_pred, average='macro'))
        classification_scores.append(scores)
        
    return classification_scores, f1_macro_folds


# In[11]:


# Again train the SMV linear with best parameter with 5 fold cross validation

best_c_linear = best_params_linear['best_c']

classification_scores_linear, f1_macro_linear = train_SVM(data_inception_features, data_class_labels, 5,
                                                          best_c_linear, best_gamma = None, kernel = 'linear')


# In[12]:


print('SVM(linear) with best parameters:')
print('\n')

for i in range(5):
    print('Fold ' +str(i+1)+':')
    print(classification_scores_linear[i])
    print('\n')


# In[13]:


# Question 4.2

# SVM (rbf) parameter tuning

C_rbf = [10**-6, 10**-4, 10**-2, 1, 10**1, 10**10]

var_data = np.var(data_inception_features)
featuresize = np.shape(data_inception_features)[1]
scale = 1 / (featuresize * var_data)

gamma_rbf = [2**-4, 2**-2, 1, 2**2, 2**10, scale]

best_params_rbf, scores_rbf, time_rbf = parameter_tuning_SVM(data_inception_features, data_class_labels, 5,
                                                             C_rbf, gammas = gamma_rbf, kernel = 'rbf')


# In[14]:


print('Best parameters for SVM (rbf) : ' +str(best_params_rbf))
print('\n')
#print('The scores for SVM (rbf) : ' +str(scores_rbf))
print('Time elapsed on training SVM (rbf) : ' +str(time_rbf) + ' seconds')


# In[15]:


# Again train the SMV (rbf) with best parameter with 5 fold cross validation

best_c_rbf = best_params_rbf['best_c']
best_gamma_rbf = best_params_rbf['best_gamma']


# In[16]:


classification_scores_rbf, f1_macro_rbf = train_SVM(data_inception_features, data_class_labels, 5,
                                                    best_c_rbf, best_gamma = best_gamma_rbf, kernel = 'rbf')


# In[17]:


print('SVM(rbf) with best parameters:')
print('\n')
for i in range(5):
    print('Fold ' +str(i+1)+':')
    print(classification_scores_rbf[i])
    print('\n')


# In[19]:


x = ['Linear', 'RBF']
F1_macro_all = list()
F1_macro_all.append(f1_macro_linear)
F1_macro_all.append(f1_macro_rbf)

figureNum += 1
plt.figure(figureNum)
plt.boxplot(F1_macro_all)
plt.title('F1 Macro Comparison between SVM Linear and SVM Rbf ')
plt.xticks(np.arange(1,3), (r'Linear', r'RBF'))
plt.xlabel('F1')
plt.show()

