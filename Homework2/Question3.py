#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv

csvfile_train = open("q3_train_dataset.csv", 'r')
csvreader_train = csv.reader(csvfile_train)
q3_data_train = list()
for lines in csvreader_train:
    q3_data_train.append(lines)

csvfile_test = open("q3_test_dataset.csv", 'r')
csvreader_test = csv.reader(csvfile_test)
q3_data_test = list()
for lines in csvreader_test:
    q3_data_test.append(lines)

# Get rid of the header    
    
q3_data_train = np.array(q3_data_train)
q3_data_train = q3_data_train[1:]

q3_data_test = np.array(q3_data_test)
q3_data_test = q3_data_test[1:]

# First column is the label so divide the train and test data from their labels.
train_data = q3_data_train[:,1:]
train_labels = q3_data_train[:,0]

test_data = q3_data_test[:,1:]
test_labels = q3_data_test[:,0]

print('The size of the train data is: ' + str(train_data.shape))
print('The size of the train labels is: ' + str(train_labels.shape))
print('The size of the test data is: ' + str(test_data.shape))
print('The size of the test labels is: ' + str(test_labels.shape))


# In[2]:


# Data Manipulation (Labeling Categorical Data)

# Gender Data
train_gender = train_data[:,1]
for i in range(np.shape(train_gender)[0]):
    if(train_gender[i] == 'male'):
        train_gender[i] = 0
    else:
        train_gender[i] = 1

test_gender = test_data[:,1]
for i in range(np.shape(test_gender)[0]):
    if(test_gender[i] == 'male'):
        test_gender[i] = 0
    else:
        test_gender[i] = 1        


# In[3]:


# Data Manipulation (Labeling Categorical Data)

# Port of Embarkation Data
train_port = train_data[:,-1]
for i in range(np.shape(train_port)[0]):
    if(train_port[i] == 'S'):
        train_port[i] = 0
    elif(train_port[i] == 'C'):
        train_port[i] = 1
    else:
        train_port[i] = 2

test_port = test_data[:,-1]
for i in range(np.shape(test_port)[0]):
    if(test_port[i] == 'S'):
        test_port[i] = 0
    elif(test_port[i] == 'C'):
        test_port[i] = 1
    else:
        test_port[i] = 2          


# In[4]:


train_gender = train_gender.astype(np.int)
test_gender = test_gender.astype(np.int)
train_port = train_port.astype(np.int)
test_port = test_port.astype(np.int)


# In[5]:


# For the categorical features i.e gender
def one_hot_encoding(data):
    samplesize = np.shape(data)[0]
    uniques = np.unique(data)
    uniqueNumber = len(uniques)
    result = np.zeros((samplesize, uniqueNumber))
    for i in range(samplesize):
        a = data[i]
        result[i,a] = 1
    return result

# Standardizing and MinMax Normalization Functions

def standardize(data):  
    mean_data = np.mean(data)
    var_data = np.var(data)
    std_data = np.sqrt(var_data)    
    data = (data - mean_data) / std_data
    
    return data

def minmax_normalize(data):
    x_max = np.amax(data)
    x_min = np.amin(data)
    
    data = (data - x_min) / (x_max - x_min)
    return data


# In[6]:


# Convert categorical data to one hot encoded data

train_gender_ohe = one_hot_encoding(train_gender)
test_gender_ohe = one_hot_encoding(test_gender)
train_port_ohe = one_hot_encoding(train_port)
test_port_ohe = one_hot_encoding(test_port)


# In[7]:


# Instead of the categorical data, we append the one hot encoded data.
# _nc means data with no categorical data

indeces_nc = [0,2,3,4,5]

train_data_nc = train_data[:, indeces_nc]
test_data_nc = test_data[:, indeces_nc]

train_data_temp = np.append(train_data_nc, train_gender_ohe, axis=-1)
train_data_final = np.append(train_data_temp, train_port_ohe, axis=-1)

test_data_temp = np.append(test_data_nc, test_gender_ohe, axis=-1)
test_data_final = np.append(test_data_temp, test_port_ohe, axis=-1)

train_data_final = train_data_final.astype(np.float)
test_data_final = test_data_final.astype(np.float)

train_labels = train_labels.astype(np.float)
test_labels = test_labels.astype(np.float)


# In[8]:


# # Normalization Sample

# # Feature: Fare

# train_fare = train_data[:,5]
# train_fare = train_fare.astype(np.float)

# figureNum = 0
# plt.figure(figureNum)
# plt.plot(train_fare)
# plt.title('Fare over Samples')
# plt.ylabel('Fare')
# plt.xlabel('Samples')
# plt.show()


# In[9]:


# train_fare_normalized = minmax_normalize(train_fare)

# figureNum += 1
# plt.figure(figureNum)
# plt.plot(train_fare_normalized)
# plt.title('Normalized Fare over Samples')
# plt.ylabel('Fare')
# plt.xlabel('Samples')
# plt.show()


# In[10]:


# Normalization for all features of both final versions of Train and Test Data

for i in range(np.shape(train_data_final)[1]):     
    train_data_final[:,i] = minmax_normalize(train_data_final[:,i])
for i in range(np.shape(test_data_final)[1]):     
    test_data_final[:,i] = minmax_normalize(test_data_final[:,i])


# In[11]:


def sigmoid(x):     

    result = 1 / (1 + np.exp(-x))
    return result


# In[12]:


def linear_model(data, weights):
    samples = np.shape(data)[0]
    temp_data = np.c_[ np.ones(samples) , data]
    
    model = (temp_data).dot(weights)
    return model


# In[13]:


def predict(data, weights):
    model = linear_model(data, weights)
    y_pred = sigmoid(model)
    y_predicted = np.where(y_pred>0.5, 1, 0)
    return y_predicted


# In[14]:


def weight_gradient(data, label, y_predicted):
    samples = np.shape(data)[0]
    temp_data = np.c_[ np.ones(samples) , data]
    
    error =  label - y_predicted 
    dW = (1 / samples) * (temp_data.T).dot(error)

    return dW


# In[15]:


def confusion_matrix(label, pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(np.shape(label)[0]):
        if(label[i] == 0 and pred[i] == 0):
            TN += 1
        elif(label[i] == 0 and pred[i] == 1):
            FP += 1
        elif(label[i] == 1 and pred[i] == 0):
            FN += 1
        else:
            TP += 1
    return TP, TN, FP, FN


# In[16]:


def logistic_regression_mini_batch(train_data, train_label, test_data, test_label, learning_rate = 0.001, 
                                   iterations = 1000, batch_size=32):
    
    np.random.seed(15)
    
    sample_size = np.shape(train_data)[0]
    feature_size = np.shape(train_data)[1]
    weights = np.random.normal(loc=0, scale=0.01, size = feature_size+1)
    
    weights_per_100_iter = list()
    
    for i in range(iterations):
        
        # Randomize the dataset for each iteration
        randomIndexes = np.random.permutation(sample_size) 
        train_data = train_data[randomIndexes] 
        train_label = train_label[randomIndexes]  
            
        number_of_batches = int(sample_size / batch_size)
        for j in range(number_of_batches):

            # Mini batch start and end index
            start = int(batch_size*j)
            end = int(batch_size*(j+1))

            # Training data with the mini batch size
            train_data_batch = train_data[start:end]
            train_label_batch = train_label[start:end]

            # The prediction of the model
            y_predicted_train = predict(train_data_batch, weights)

            # The gradient of the weight
            dW = weight_gradient(train_data_batch, train_label_batch, y_predicted_train)

            # Weight update 
            weights += learning_rate * dW
         
    
    y_predicted_test = predict(test_data, weights)
    TP, TN, FP, FN = confusion_matrix(test_label, y_predicted_test)
    return TP, TN, FP, FN, weights_per_100_iter


# In[17]:


def logistic_regression_stochastic(train_data, train_label, test_data, test_label, learning_rate = 0.001, iterations = 1000):
    
    np.random.seed(15)
    
    sample_size = np.shape(train_data)[0]
    feature_size = np.shape(train_data)[1]
    weights = np.random.normal(loc=0, scale=0.01, size = feature_size+1)
    
    weights_per_100_iter = list()
    
    for i in range(iterations):
        
        # Randomize the dataset for each iteration
        randomIndexes = np.random.permutation(sample_size) 
        train_data = train_data[randomIndexes] 
        train_label = train_label[randomIndexes]  
    
        for i,x in enumerate(train_data):

            # Select 1 sample for stochastic gradient descent update

            x = np.array(x)
            x = x.reshape(1,-1)  

            # Predict the selected sample's label and calculate the error.
            y_predicted = predict(x, weights)

            # Compute the gradients.
            dW = weight_gradient(x, train_labels[i], y_predicted)

            # Weight update 
            weights += learning_rate * dW
                
     
    
    y_predicted_test = predict(test_data, weights)
    TP, TN, FP, FN = confusion_matrix(test_label, y_predicted_test)
    return TP, TN, FP, FN, weights_per_100_iter


# In[18]:


def logistic_regression_full_batch(train_data, train_label, test_data, test_label, learning_rate = 0.001, iterations = 1000):
    
    np.random.seed(15)
    
    sample_size = np.shape(train_data)[0]
    feature_size = np.shape(train_data)[1]
    weights = np.random.normal(loc=0, scale=0.01, size = feature_size+1)
    
    weights_per_100_iter = list()
    
    for i in range(iterations):
        
        if(i % 100 == 0):
            weights_per_100_iter.append(weights)
            
        # Predict the selected sample's label and calculate the error.            
        y_predicted = predict(train_data, weights)

        # Compute the gradients.
        dW = weight_gradient(train_data, train_labels, y_predicted)

        # Weight update 
        weights += learning_rate * dW     
    
    
    y_predicted_test = predict(test_data, weights)
    TP, TN, FP, FN = confusion_matrix(test_label, y_predicted_test)
    return TP, TN, FP, FN, weights_per_100_iter


# In[19]:


TP_minibatch, TN_minibatch, FP_minibatch, FN_minibatch, weights_minibatch = logistic_regression_mini_batch(train_data_final, 
                                                                                                           train_labels, 
                                                                                                           test_data_final, 
                                                                                                           test_labels, 
                                                                                                           learning_rate = 0.01, 
                                                                                                           iterations = 1000, 
                                                                                                           batch_size=32)


# In[20]:


print('Confusion Matrix for mini-batch gradient ascent')
print('\n')
print('True Positive : ' + str(TP_minibatch))
print('True Negative : ' + str(TN_minibatch))
print('False Positive : ' + str(FP_minibatch))
print('False Negative : ' + str(FN_minibatch))
print('----------------------------------')


# In[21]:


TP_sgd, TN_sgd, FP_sgd, FN_sgd, weights_sgd = logistic_regression_stochastic(train_data_final, train_labels, test_data_final, 
                                                                             test_labels, learning_rate = 0.0001, 
                                                                             iterations = 1000)


# In[22]:


print('Confusion Matrix for stochastic gradient ascent')
print('\n')
print('True Positive : ' + str(TP_sgd))
print('True Negative : ' + str(TN_sgd))
print('False Positive : ' + str(FP_sgd))
print('False Negative : ' + str(FN_sgd))
print('----------------------------------')


# In[23]:


TP_fbgd, TN_fbgd, FP_fbgd, FN_fbgd, weights_fbgd = logistic_regression_full_batch(train_data_final, train_labels, 
                                                                                  test_data_final, test_labels,
                                                                                  learning_rate = 0.001, iterations = 1000)


# In[24]:


print('Confusion Matrix for full-batch gradient ascent')
print('\n')
print('True Positive : ' + str(TP_fbgd))
print('True Negative : ' + str(TN_fbgd))
print('False Positive : ' + str(FP_fbgd))
print('False Negative : ' + str(FN_fbgd))
print('----------------------------------')


# In[25]:


def accuracy(TP, TN, FP, FN):
    result = (TP + TN) / (TP + TN + FP + FN)
    return result


# In[26]:


def precision(TP, TN, FP, FN):
    result = TP  / (TP + FP)
    return result


# In[27]:


def recall(TP, TN, FP, FN):
    result = TP / (TP + FN)
    return result


# In[28]:


def NPV(TP, TN, FP, FN):
    result = TN / (TN + FN)
    return result


# In[29]:


def FPR(TP, TN, FP, FN):
    result = FP / (FP + TN)
    return result


# In[30]:


def FDR(TP, TN, FP, FN):
    result = FP / (FP + TP)
    return result


# In[31]:


def F1(TP, TN, FP, FN):
    precision_1 = precision(TP, TN, FP, FN)
    recall_1 = recall(TP, TN, FP, FN)
    result = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
    return result


# In[32]:


def F2(TP, TN, FP, FN):
    precision_2 = precision(TP, TN, FP, FN)
    recall_2 = recall(TP, TN, FP, FN)
    result = (5 * precision_2 * recall_2) / (4 * precision_2 + recall_2)
    return result


# In[33]:


accuracy_minibatch = accuracy(TP_minibatch, TN_minibatch, FP_minibatch, FN_minibatch)
precision_minibatch = precision(TP_minibatch, TN_minibatch, FP_minibatch, FN_minibatch)
recall_minibatch = recall(TP_minibatch, TN_minibatch, FP_minibatch, FN_minibatch)
NPV_minibatch = NPV(TP_minibatch, TN_minibatch, FP_minibatch, FN_minibatch)
FPR_minibatch = FPR(TP_minibatch, TN_minibatch, FP_minibatch, FN_minibatch)
FDR_minibatch = FDR(TP_minibatch, TN_minibatch, FP_minibatch, FN_minibatch)
F1_minibatch = F1(TP_minibatch, TN_minibatch, FP_minibatch, FN_minibatch)
F2_minibatch = F2(TP_minibatch, TN_minibatch, FP_minibatch, FN_minibatch)

print('Scores for Mini-Batch Gradient Ascent ')
print('\n')
print('Accuracy                  : ' + str(accuracy_minibatch) )
print('Precision                 : ' + str(precision_minibatch) )
print('Recall                    : ' + str(recall_minibatch) )
print('Negative Predictive Value : ' + str(NPV_minibatch) )
print('False Positive Rate       : ' + str(FPR_minibatch) )
print('False Discovery Rate      : ' + str(FDR_minibatch) )
print('F1 Score                  : ' + str(F1_minibatch) )
print('F2 Score                  : ' + str(F2_minibatch) )

print('----------------------------------')


# In[34]:


accuracy_sgd = accuracy(TP_sgd, TN_sgd, FP_sgd, FN_sgd)
precision_sgd = precision(TP_sgd, TN_sgd, FP_sgd, FN_sgd)
recall_sgd = recall(TP_sgd, TN_sgd, FP_sgd, FN_sgd)
NPV_sgd = NPV(TP_sgd, TN_sgd, FP_sgd, FN_sgd)
FPR_sgd = FPR(TP_sgd, TN_sgd, FP_sgd, FN_sgd)
FDR_sgd = FDR(TP_sgd, TN_sgd, FP_sgd, FN_sgd)
F1_sgd = F1(TP_sgd, TN_sgd, FP_sgd, FN_sgd)
F2_sgd = F2(TP_sgd, TN_sgd, FP_sgd, FN_sgd)

print('Scores for Stochastic Gradient Ascent ')
print('\n')
print('Accuracy                  : ' + str(accuracy_sgd) )
print('Precision                 : ' + str(precision_sgd) )
print('Recall                    : ' + str(recall_sgd) )
print('Negative Predictive Value : ' + str(NPV_sgd) )
print('False Positive Rate       : ' + str(FPR_sgd) )
print('False Discovery Rate      : ' + str(FDR_sgd) )
print('F1 Score                  : ' + str(F1_sgd) )
print('F2 Score                  : ' + str(F2_sgd) )

print('----------------------------------')


# In[35]:


accuracy_fbgd = accuracy(TP_fbgd, TN_fbgd, FP_fbgd, FN_fbgd)
precision_fbgd = precision(TP_fbgd, TN_fbgd, FP_fbgd, FN_fbgd)
recall_fbgd = recall(TP_fbgd, TN_fbgd, FP_fbgd, FN_fbgd)
NPV_fbgd = NPV(TP_fbgd, TN_fbgd, FP_fbgd, FN_fbgd)
FPR_fbgd = FPR(TP_fbgd, TN_fbgd, FP_fbgd, FN_fbgd)
FDR_fbgd = FDR(TP_fbgd, TN_fbgd, FP_fbgd, FN_fbgd)
F1_fbgd = F1(TP_fbgd, TN_fbgd, FP_fbgd, FN_fbgd)
F2_fbgd = F2(TP_fbgd, TN_fbgd, FP_fbgd, FN_fbgd)

print('Scores for Full-Batch Gradient Ascent ')
print('\n')
print('Accuracy                  : ' + str(accuracy_fbgd) )
print('Precision                 : ' + str(precision_fbgd) )
print('Recall                    : ' + str(recall_fbgd) )
print('Negative Predictive Value : ' + str(NPV_fbgd) )
print('False Positive Rate       : ' + str(FPR_fbgd) )
print('False Discovery Rate      : ' + str(FDR_fbgd) )
print('F1 Score                  : ' + str(F1_fbgd) )
print('F2 Score                  : ' + str(F2_fbgd) )

print('\n')
print('Weights at each 100th iteration: ')
print('\n')
for i in range(len(weights_fbgd)):
    print(weights_fbgd[i])
    print('\n')
    
print('----------------------------------')

