#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv

csvfile = open("q2_dataset.csv", 'r')
csvreader = csv.reader(csvfile)
q2_data = list()
for lines in csvreader:
    q2_data.append(lines)

# Get rid of the Header

q2_data = np.array(q2_data)
q2_data = q2_data[1:]


# In[2]:


data = q2_data[:,0:-1]
labels = q2_data[:,7:]

data = np.array(data)
labels = np.array(labels)

data = data.astype(np.float64)
labels = labels.astype(np.float64)

print('The shape of data: ' +str(data.shape))
print('The shape of labels: ' +str(labels.shape))


# In[3]:


def predict(data, weights):
    samples = np.shape(data)[0]
    temp_data = np.c_[ np.ones(samples) , data]
    y_predicted = (temp_data).dot(weights)
    return y_predicted


# In[4]:


def predict_L1(data, weights_L1, bias_L1):
    y_predicted_L1 = data.dot(weights_L1) + bias_L1
    return y_predicted_L1


# In[5]:


def calculate_R2(y, y_pred):
    y = y[:,0]
    y_pred = y_pred[:,0]
    pearson = np.corrcoef(y, y_pred)[1, 0]
    return pearson ** 2


# In[6]:


def calculate_MSE(y, y_pred):

    error = y - y_pred
    loss = np.mean(error**2)
    return loss


# In[7]:


def calculate_MAE(y, y_pred):

    error = np.abs(y - y_pred)
    loss = np.mean(error)
    return loss


# In[8]:


def calculate_MAPE(y, y_pred):
    
    loss = 0
    n = np.shape(y)[0]
    for i in range(n):
        loss += np.abs((y[i] - y_pred[i]) / y[i])
    return np.mean(loss)


# In[9]:


def weight_update(data, label):
    samples = np.shape(data)[0]
    temp_data = np.c_[ np.ones(samples) , data]
    
    xt_x = np.linalg.inv((temp_data.T).dot(temp_data))
    xt_y = (temp_data.T).dot(label)
    weights = (xt_x).dot(xt_y)
    return weights


# In[10]:


def linear_regression(data, labels, k, iterations=500, regularization = False, lambdaa = 0.01, learning_rate = 0.001):
    
    R2 = list()
    MSE = list()
    MAE = list()
    MAPE = list()
    
    np.random.seed(8)
    sampleSize = np.shape(data)[0]
    featureSize = np.shape(data)[1]
    
    # Weights and bias for L1 regularization
    weights_L1 = np.zeros((featureSize,1))
    bias_L1 = 0
    
    randomIndexes = np.random.permutation(sampleSize)
    data = data[randomIndexes]
    labels = labels[randomIndexes]

    fold_size = int(sampleSize / k)
    
    # Implement a 5-fold cross validation
    for j in range(k):
        
        test_index_start = fold_size*j
        train_index_start = fold_size*(j+1)

        test_indeces = np.arange(test_index_start, train_index_start) % sampleSize
        train_indeces = np.arange(train_index_start, sampleSize + test_index_start) % sampleSize
        
        test_data = data[test_indeces]
        test_labels = labels[test_indeces]
        
        train_data = data[train_indeces]
        train_labels = labels[train_indeces]
        
        if(regularization == False):
            weights = weight_update(train_data, train_labels)
        else:
            # A fold will be trained with L1 Learning
            for i in range(iterations):
                
                # Predict the selected sample's label and calculate the error.
                
                y_predicted_L1 = predict_L1(train_data, weights_L1, bias_L1)
                error_L1 = y_predicted_L1 - train_labels

                # Compute the gradients.
                
                dW_L1 = ((train_data.T).dot(error_L1)) / np.shape(train_data)[0]
                dB_L1 = np.sum(error_L1) / np.shape(train_data)[0]

                # Weight and Bias update
                
                weights_L1_update = np.where(weights_L1 > 0 , dW_L1 + lambdaa, dW_L1 - lambdaa)
                weights_L1 -= learning_rate * weights_L1_update
                bias_L1 -= learning_rate * dB_L1  
        
        if(regularization == False):
            y_prediction_test = predict(test_data, weights)
        else:
            y_prediction_test = predict_L1(test_data, weights_L1, bias_L1)
            
        R2_s = calculate_R2(test_labels, y_prediction_test)
        MSE_s = calculate_MSE(test_labels, y_prediction_test)
        MAE_s = calculate_MAE(test_labels, y_prediction_test)
        MAPE_s = calculate_MAPE(test_labels, y_prediction_test)
        
        R2.append(R2_s)
        MSE.append(MSE_s)
        MAE.append(MAE_s)
        MAPE.append(MAPE_s)
            
    return R2, MSE, MAE, MAPE       


# In[11]:


R2, MSE, MAE, MAPE = linear_regression(data, labels, 5)


# In[12]:


# for i in range(len(R2)):
#     print('R2   : ' + str(R2[i]))
#     print('MSE  : ' + str(MSE[i]))
#     print('MAE  : ' + str(MAE[i]))
#     print('MAPE : ' + str(MAPE[i]))


# In[13]:


figureNum = 0
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x,R2)
plt.title('R2 Over 5 Folds')
plt.xlabel('Folds')
plt.ylabel('R2')
plt.show()


# In[14]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x, MSE)
plt.title('MSE Over 5 Folds')
plt.xlabel('Folds')
plt.ylabel('MSE')
plt.show()


# In[15]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x, MAE)
plt.title('MAE Over 5 Folds')
plt.xlabel('Folds')
plt.ylabel('MAE')
plt.show()


# In[16]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x, MAPE)
plt.title('MAPE Over 5 Folds')
plt.xlabel('Folds')
plt.ylabel('MAPE')
plt.show()


# In[17]:


R2_mean = np.mean(R2)
MSE_mean = np.mean(MSE)
MAE_mean = np.mean(MAE)
MAPE_mean = np.mean(MAPE)

print('The mean of the R2 for 5 folds is   : ' +str(R2_mean))
print('The mean of the MSE for 5 folds is  : ' +str(MSE_mean))
print('The mean of the MAE for 5 folds is  : ' +str(MAE_mean))
print('The mean of the MAPE for 5 folds is : ' +str(MAPE_mean))


# In[18]:


# L1 Regularization with lambda 0.01

R2_L1, MSE_L1, MAE_L1, MAPE_L1 = linear_regression(data, labels, 5, iterations = 500, regularization = True, lambdaa = 0.01, learning_rate = 0.00001)


# In[19]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x,R2_L1)
plt.title('R2 Over 5 Folds (with L1 Regularization lambda = 0.01)')
plt.xlabel('Folds')
plt.ylabel('R2')
plt.show()


# In[20]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x,MSE_L1)
plt.title('MSE Over 5 Folds (with L1 Regularization lambda = 0.01)')
plt.xlabel('Folds')
plt.ylabel('MSE')
plt.show()


# In[21]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x,MAE_L1)
plt.title('MAE Over 5 Folds (with L1 Regularization lambda = 0.01)')
plt.xlabel('Folds')
plt.ylabel('MAE')
plt.show()


# In[22]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x,MAPE_L1)
plt.title('MAPE Over 5 Folds (with L1 Regularization lambda = 0.01)')
plt.xlabel('Folds')
plt.ylabel('MAPE')
plt.show()


# In[23]:


R2_L1_mean = np.mean(R2_L1)
MSE_L1_mean = np.mean(MSE_L1)
MAE_L1_mean = np.mean(MAE_L1)
MAPE_L1_mean = np.mean(MAPE_L1)

print('The mean of the R2 for 5 folds is (L1 with 0.01 lambda)   : ' +str(R2_L1_mean))
print('The mean of the MSE for 5 folds is (L1 with 0.01 lambda)  : ' +str(MSE_L1_mean))
print('The mean of the MAE for 5 folds is (L1 with 0.01 lambda)  : ' +str(MAE_L1_mean))
print('The mean of the MAPE for 5 folds is(L1 with 0.01 lambda)  : ' +str(MAPE_L1_mean))


# In[24]:


R2_box1 = list()
R2_box1.append(R2)
R2_box1.append(R2_L1)

figureNum += 1
plt.figure(figureNum)
plt.boxplot(R2_box1)
plt.title('R2 Comparison Between Regularized and Non-regularized Models')
plt.xticks(np.arange(1,3), (r'w/o L1 reg', r'with L1 reg (lambda = 0.01)'))
plt.xlabel('Models')
plt.ylabel('R2')
plt.show()


# In[25]:


MSE_box1 = list()
MSE_box1.append(MSE)
MSE_box1.append(MSE_L1)

figureNum += 1
plt.figure(figureNum)
plt.boxplot(MSE_box1)
plt.title('MSE Comparison Between Regularized and Non-regularized Models')
plt.xticks(np.arange(1,3), (r'w/o L1 reg', r'with L1 reg (lambda = 0.01)'))
plt.xlabel('Models')
plt.ylabel('MSE')
plt.show()


# In[26]:


MAE_box1 = list()
MAE_box1.append(MAE)
MAE_box1.append(MAE_L1)

figureNum += 1
plt.figure(figureNum)
plt.boxplot(MAE_box1)
plt.title('MAE Comparison Between Regularized and Non-regularized Models')
plt.xticks(np.arange(1,3), (r'w/o L1 reg', r'with L1 reg (lambda = 0.01)'))
plt.xlabel('Models')
plt.ylabel('MAE')
plt.show()


# In[27]:


MAPE_box1 = list()
MAPE_box1.append(MAPE)
MAPE_box1.append(MAPE_L1)

figureNum += 1
plt.figure(figureNum)
plt.boxplot(MAPE_box1)
plt.xticks(np.arange(1,3), (r'w/o L1 reg', r'with L1 reg (lambda = 0.01)'))
plt.xlabel('Models')
plt.ylabel('MAPE')
plt.show()


# In[28]:


# L1 Regularization with lambda 1

R2_L1_1, MSE_L1_1, MAE_L1_1, MAPE_L1_1 = linear_regression(data, labels, 5, iterations = 500, regularization = True, lambdaa = 1, learning_rate = 0.00001)


# In[29]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x,R2_L1_1)
plt.title('R2 Over 5 Folds (with L1 Regularization lambda = 1)')
plt.xlabel('Folds')
plt.ylabel('R2')
plt.show()


# In[30]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x,MSE_L1_1)
plt.title('MSE Over 5 Folds (with L1 Regularization lambda = 1)')
plt.xlabel('Folds')
plt.ylabel('MSE')
plt.show()


# In[31]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x,MAE_L1_1)
plt.title('MAE Over 5 Folds (with L1 Regularization lambda = 1)')
plt.xlabel('Folds')
plt.ylabel('MAE')
plt.show()


# In[32]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,6)
plt.bar(x,MAPE_L1_1)
plt.title('MAPE Over 5 Folds (with L1 Regularization lambda = 1)')
plt.xlabel('Folds')
plt.ylabel('MAPE')
plt.show()


# In[33]:


R2_L1_1_mean = np.mean(R2_L1_1)
MSE_L1_1_mean = np.mean(MSE_L1_1)
MAE_L1_1_mean = np.mean(MAE_L1_1)
MAPE_L1_1_mean = np.mean(MAPE_L1_1)

print('The mean of the R2 for 5 folds is (L1 with 1 lambda)   : ' +str(R2_L1_1_mean))
print('The mean of the MSE for 5 folds is (L1 with 1 lambda)  : ' +str(MSE_L1_1_mean))
print('The mean of the MAE for 5 folds is (L1 with 1 lambda)  : ' +str(MAE_L1_1_mean))
print('The mean of the MAPE for 5 folds is(L1 with 1 lambda)  : ' +str(MAPE_L1_1_mean))


# In[34]:


R2_box2 = list()
R2_box2.append(R2)
R2_box2.append(R2_L1_1)

figureNum += 1
plt.figure(figureNum)
plt.boxplot(R2_box2)
plt.title('R2 Comparison Between Regularized and Non-regularized Models ')
plt.xticks(np.arange(1,3), (r'w/o L1 reg', r'with L1 reg (lambda = 1)'))
plt.xlabel('Models')
plt.ylabel('R2')
plt.show()


# In[35]:


MSE_box2 = list()
MSE_box2.append(MSE)
MSE_box2.append(MSE_L1_1)

figureNum += 1
plt.figure(figureNum)
plt.boxplot(MSE_box2)
plt.title('MSE Comparison Between Regularized and Non-regularized Models ')
plt.xticks(np.arange(1,3), (r'w/o L1 reg', r'with L1 reg (lambda = 1)'))
plt.xlabel('Models')
plt.ylabel('MSE')
plt.show()


# In[36]:


MAE_box2 = list()
MAE_box2.append(MAE)
MAE_box2.append(MAE_L1_1)

figureNum += 1
plt.figure(figureNum)
plt.boxplot(MAE_box2)
plt.title('MAE Comparison Between Regularized and Non-regularized Models ')
plt.xticks(np.arange(1,3), (r'w/o L1 reg', r'with L1 reg (lambda = 1)'))
plt.xlabel('Models')
plt.ylabel('MAE')
plt.show()


# In[37]:


MAE_box2 = list()
MAE_box2.append(MAPE)
MAE_box2.append(MAPE_L1_1)

figureNum += 1
plt.figure(figureNum)
plt.boxplot(MAE_box2)
plt.title('MAPE Comparison Between Regularized and Non-regularized Models')
plt.xticks(np.arange(1,3), (r'w/o L1 reg', r'with L1 reg (lambda = 1)'))
plt.xlabel('Models')
plt.ylabel('MAPE')
plt.show()

