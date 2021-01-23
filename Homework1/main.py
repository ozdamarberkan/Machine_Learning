#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 2

import csv
import numpy as np
import matplotlib.pyplot as plt
import time

# Extract data from the csv file into the tokenized_corpus list
csvfiledata = open("tokenized_corpus.csv", 'r')
csvreaderdata = csv.reader(csvfiledata) 
data = list()
for lines in csvreaderdata: 
    data.append(lines) 


# In[2]:


# Convert the tokenized_corpus into the numpy array and check the first line.
data = np.asarray(data)

# Example showing that the words in a line seperated to different columns.

print('First line: ' + str(data[0]))
print('\n')
print('First word of the first line: ' + str(data[0][0]))


# In[3]:


# Defined getUniqueWords function to set up our vocabulary matrix.
# words is the data we have extracted from the 
def getUniqueWords(words):
    
    '''
    This function returns a matrix that contains every word only once.

    INPUTS:
    
        words        : Matrix that contains whole words.
        
        
    RETURNS:
    
        uniqueWords  : Matrix that contains unique words.
    '''   
    
    uniqueWords = list()
    for i in range(np.shape(words)[0]):
        for j in range(np.shape(words[i])[0]):
            if(words[i][j] not in uniqueWords):
                uniqueWords.append(words[i][j])
    uniqueWords = np.asarray(uniqueWords)
    return uniqueWords   


# In[4]:


vocabulary = getUniqueWords(data)
print(vocabulary.shape)


# In[5]:


# createFeatures function creates the feature matrix M.

def createFeatures(data, vocabulary):
    
    '''
    This function creates the M matrix which is the feature matrix that contains frequency of each word for each message

    INPUTS:
    
        data         : Message data that contains all the messages(SMS).
        vocabulary   : Matrix that contains unique words.
        
    RETURNS:
    
        M            : Feature matrix
    '''          

    
    M = np.zeros((np.shape(data)[0], np.shape(vocabulary)[0]))
    for i in range(np.shape(data)[0]):
        S = np.zeros(np.shape(vocabulary)[0])
        for j in range(np.shape(data[i])[0]):
            location = np.where(vocabulary == data[i][j])[0]
            S[location] = S[location] + 1
        M[i][:] = S[:]
    return M


# In[6]:


M = createFeatures(data,vocabulary)


# In[7]:


print(M.shape)


# In[8]:


# Question 2.1

# Save the M matrix as feature_set.csv

np.savetxt('feature_set.csv', M, delimiter=',', fmt='%d')


# In[9]:


# Extract data from the csv file into the tokenized_corpus list
csvfilelabel = open("labels.csv", 'r')
csvreaderlabel = csv.reader(csvfilelabel) 
labels = list()
for lines in csvreaderlabel: 
    labels.append(lines) 
    
# Convert the tokenized_corpus into the numpy array and check the first line.
labels = np.asarray(labels)
print(labels.shape)


# In[10]:


def splitData(data, labels):
    
    '''
    This function splits the data and the labels into training data, training labels. test data and test labels.

    INPUTS:
    
        data         : Data to be splitted.
        labels       : Labels to be splitted.
        
    RETURNS:
    
        train_data   : Training data with size (4460, 9259)
        train_labels : Labels for training data with size (4460, 1)
        test_data    : Test data with size (1112, 9259)
        test_labels  : Labels for test data with size (1112, 1)
    '''    
    
    train_data = list()
    test_data = list()
    train_labels = list()
    test_labels = list()
    trainNumber = 4460
    testNumber = 1112
    size = np.shape(labels)[0]
    for i in range(size):
        if (i < trainNumber):
            train_data.append(data[i])
            train_labels.append(labels[i])
        else:
            test_data.append(data[i])
            test_labels.append(labels[i])
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels)
    return train_data, train_labels, test_data, test_labels


# In[11]:


train_data, train_labels, test_data, test_labels = splitData(M, labels)
print('The shape of train data: ' + str(train_data.shape))
print('The shape of train labels: ' + str(train_labels.shape))
print('The shape of test data: ' + str(test_data.shape))
print('The shape of test labels: ' + str(test_labels.shape))


# In[12]:


def naiveBayesTrain(train_data, train_labels, smoothing_constant):
    
    '''
    This function calculates the probability distribution of words that contains to ham and spam classes separately.
    Also, calculates the probability of a message(line) belongs to class ham or spam.

    INPUTS:
    
        train_data         : Training data
        train_labels       : Labels for the train_data
        smoothing_constant : Laplace smoothing constant
        
    RETURNS:
    
        probs_ham          : Probability distribution for the words in ham class
        probs_spam         : Probability distribution for the words in spam class
        prob_class_ham     : The probability that the class is of a message is ham
        prob_class_spam    : The probability that the class is of a message is spam
    '''

    locations_ham = np.where(train_labels == '0')[0]
    locations_spam = np.where(train_labels == '1')[0]
    probs_ham = np.zeros(np.shape(train_data)[1])
    probs_spam = np.zeros(np.shape(train_data)[1])
    probs_ham[:] = probs_ham[:] + smoothing_constant
    probs_spam[:] = probs_spam[:] + smoothing_constant
    
    for locations in locations_ham:
        probs_ham[:] = probs_ham[:] + train_data[locations][:]

    for locations in locations_spam:
        probs_spam[:] = probs_spam[:] + train_data[locations][:]        
        
    numberOfHam = len(locations_ham)
    numberOfSpam = len(locations_spam)    

    
    probs_ham = probs_ham / (np.sum(probs_ham)) 
    probs_spam = probs_spam / (np.sum(probs_spam)) 
    prob_class_ham = numberOfHam / np.shape(train_labels)[0]
    prob_class_spam = numberOfSpam / np.shape(train_labels)[0]
    
    
    return probs_ham, probs_spam, prob_class_ham, prob_class_spam


# In[13]:


def naiveBayesPredict(test_data, probs_ham, probs_spam, prob_class_ham, prob_class_spam):

    '''
    This function predicts the class of a message(line) with the given probabilities from naiveBayesTrain function.

    INPUTS:
    
        test_data          : Test data
        probs_ham          : Probability distribution for the words in ham class
        probs_spam         : Probability distribution for the words in spam class
        prob_class_ham     : The probability that the class is of a message is ham
        prob_class_spam    : The probability that the class is of a message is spam
        
    RETURNS:
    
        predicts        : The predictions matrix which contains the class predictions of messages in the test_data.
    '''       
    predicts = list()
    log_probs_ham = np.where(probs_ham == 0, 0, np.log(probs_ham))
    log_probs_spam = np.where(probs_spam == 0, 0, np.log(probs_spam))
                           
    for i in range(np.shape(test_data)[0]):
                
        ham =  test_data[i].dot(log_probs_ham)
        spam =  test_data[i].dot(log_probs_spam)
            
        ham = ham + np.log(prob_class_ham)
        spam = spam + np.log(prob_class_spam) 
                 
        if ( ham > spam ):
            predicts.append('0')
        else:
            predicts.append('1')
                 
    return predicts


# In[14]:


def accuracy(predictions, test_labels):
    
    '''
    This function calculates the accuracy between the predictions and the labels

    INPUTS:
    
        predictions  : The predictions matrix which contains the class predictions of messages in the test_data.
        labels       : Labels for test_data.
        
    RETURNS:
    
        accuracy     : Accuracy between predictions and labels
    '''     
    
    count = 0
    size = np.shape(predictions)[0]
    for i in range(size):
        if(predictions[i] == test_labels[i]):
            count = count + 1
    accuracy = (count / size)*100
    return accuracy


# In[15]:


# Training Naive Bayes without Laplace smoothing 

dist_ham, dist_spam, classProb_ham, classProb_spam = naiveBayesTrain(train_data, train_labels, 0)


# In[16]:


# print('For Naive Bayes with Laplace Smoothing a = 0')
# print('\n')
# print('The probability that the class is of ham: ' + str(classProb_ham))
# print('\n')
# print('The probability that the class is of spam: ' + str(classProb_spam))
# print('\n')
# print('Probability distribution for the words in ham class: ' +str(dist_ham))
# print('\n')
# print('Probability distribution for the words in spam class: ' +str(dist_spam))


# In[17]:


# Predictions for Naive Bayes with no laplace smoothing

predictions = naiveBayesPredict(test_data, dist_ham, dist_spam, classProb_ham, classProb_spam)


# In[18]:


# Accuracy for Naive Bayes without Laplace smoothing

accuracy_test = accuracy(predictions, test_labels)

print('Accuracy for laplace smoothing = 0 : ' + str(accuracy_test))


# In[19]:


# Save the accuracy for a = 0 as test_accuracy.csv

accuracy_test = np.asarray(accuracy_test).reshape(1,1)
np.savetxt('test_accuracy.csv', accuracy_test, delimiter=',', fmt='%f')


# In[20]:


# Training Naive Bayes with Laplace smoothing where a = 1

dist_ham_laplace, dist_spam_laplace, classProb_ham_laplace, classProb_spam_laplace = naiveBayesTrain(train_data, train_labels, 1)


# In[21]:


# print('For Naive Bayes with Laplace smoothing a = 1')
# print('\n')
# print('The probability that the class is of ham: ' + str(classProb_ham_laplace))
# print('\n')
# print('The probability that the class is of spam: ' + str(classProb_spam_laplace))
# print('\n')
# print('Probability distribution for the words in ham class: ' +str(dist_ham_laplace))
# print('\n')
# print('Probability distribution for the words in spam class: ' +str(dist_spam_laplace))


# In[22]:


# Predictions for Naive Bayes with Laplace smoothing

predictions_laplace = naiveBayesPredict(test_data, dist_ham_laplace, dist_spam_laplace, classProb_ham_laplace, classProb_spam_laplace)


# In[23]:


# Accuracy for Naive Bayes with Laplace smoothing

accuracy_test_laplace = accuracy(predictions_laplace, test_labels)
print('Accuracy for laplace smoothing = 1 : ' + str(accuracy_test_laplace))


# In[24]:


# Save the accuracy for a = 0 as test_accuracy_laplace.csv

accuracy_test_laplace = np.asarray(accuracy_test_laplace).reshape(1,1)
np.savetxt('test_accuracy_laplace.csv', accuracy_test_laplace, delimiter=',', fmt='%f')


# In[25]:


# Question 3

def frequencyOfWords(M):
    
    '''
    This function calculates the frequency of the words

    INPUTS:
    
        M            : Whole dataset
        
    RETURNS:
    
        frequencies  : The calculated frequency matrix 
    ''' 
    
    frequencies = np.zeros(np.shape(M)[1])
    
    for i in range(np.shape(M)[0]):
        frequencies[:] = frequencies[:] + M[i][:]
        
    return frequencies    


# In[26]:


def selectFrequentData(M, train_data, test_data, k):
    
    '''
    This function selects the words with frequencies higher than a certain threshold k

    INPUTS:
    
        M            : Whole dataset
        train_data   : The training data
        test_data    : The test data
        k            : Threshold
        
    RETURNS:
    
        data         : The training data with selected words
        testData     : The test data with selected words
    ''' 
    
    frequencies = frequencyOfWords(M)
    indeces = np.where(frequencies >= k)[0]
    data = train_data[:,indeces]
    data = np.asarray(data)
    testData = test_data[:, indeces]
    testData = np.asarray(testData)

    return data, testData
    


# In[27]:


def forwardSelect(M, train_data, train_labels, test_data, test_labels, k, smoothing_constant):
     
    '''
    This function selects features 1 by 1 until there is no more performance gain.

    INPUTS:
    
        M                  : Whole dataset     
        train_data         : Training data
        train_labels       : Labels for the train_data
        test_data          : Test data
        test_labels        : Test labels
        smoothing_constant : Laplace smoothing constant
        

        
    RETURNS:
    
        indeces            : Indeces for selected features
        performance_best   : Accuracy values for the set of selected features
    '''               
        
    train_data_10, test_data_10 = selectFrequentData(M, train_data, test_data, k)

    performances_best = list()
    indeces = list()
    indecesSet = list()
    exit_loop = 0
    best_accuracy = 0
    size = np.shape(train_data_10)[1]
    while (exit_loop == 0):     
        performances = np.zeros(size)
        for i in range(size):
            if(i not in indeces):

                indecesSet[:] = indeces[:]
                indecesSet.append(i)
                
                dist_ham, dist_spam, classProb_ham, classProb_spam = naiveBayesTrain(train_data_10[:, indecesSet], train_labels, smoothing_constant)
                prediction = naiveBayesPredict(test_data_10[:, indecesSet], dist_ham, dist_spam, classProb_ham, classProb_spam)    
                performances[i] = accuracy(prediction, test_labels)
                
        if (np.amax(performances) <= best_accuracy):
            return indeces, performances_best 
        
        index = np.argmax(performances)
        best_accuracy = performances[index]
        performances_best.append(best_accuracy)
        indeces.append(index)
    return indeces, performances_best  


# In[28]:


index_featureSet, performance_b = forwardSelect(M, train_data, train_labels, test_data, test_labels, 10, 1)


# In[40]:


print(index_featureSet)


# In[30]:


print(performance_b)


# In[31]:


# Save the indeces of selected words for forward selection as forward_selection.csv

np.savetxt('forward_selection.csv', index_featureSet, delimiter=',', fmt='%f')


# In[32]:


# Visualization of the Accuracy of the model of Forward Selection

figureNo = 0
size = np.shape(performance_b)[0]
x_axis = np.arange(0,size)
plt.figure(figureNo, figsize=(10,8))
plt.title('Accuracy of the Forward Selection for each selected word')
plt.xlabel('Number of Words')
plt.ylabel('Accuracy of the Model')
plt.plot(x_axis, performance_b)
plt.show()


# In[33]:


# Question 3.2

def sortFrequencies(M, train_data, test_data, k):
    
    '''
    This function finds the frequencies and sorts them from most frequent to least frequent

    INPUTS:
    
        M            : Whole dataset
        train_data   : The training data
        test_data    : The test data
        k            : Threshold
        
    RETURNS:
    
        trainData    : The training data with selected words
        testData     : The test data with selected words
    ''' 
    
    frequencies = frequencyOfWords(M)
    freq = frequencies[frequencies>=k]
    indeces = np.argsort(freq)[::-1]
    train_data_10, test_data_10 = selectFrequentData(M, train_data, test_data, k)

    trainData = train_data_10[:, indeces]
    trainData = np.asarray(trainData)
    testData = test_data_10[:, indeces]
    testData = np.asarray(testData)
    return trainData, testData


# In[34]:


def frequentWordsTrain(M, train_data, train_labels, test_data, test_labels, k, smoothing_constant):
    
    '''
    This function selects the most frequent features 1 by 1 and stores performances for each run.

    INPUTS:
    
        M                  : Whole dataset     
        train_data         : Training data
        train_labels       : Labels for the train_data
        test_data          : Test data
        test_labels        : Test labels
        k                  : Threshold
        smoothing_constant : Laplace smoothing constant
        

        
    RETURNS:
    
        performances       : Accuracy values for the set of selected features
    '''                   
    
    train_data_sorted, test_data_sorted = sortFrequencies(M, train_data, test_data, k)
    performances = list()
    for i in range(np.shape(train_data_sorted)[1]):
        dist_ham, dist_spam, classProb_ham, classProb_spam = naiveBayesTrain(train_data_sorted[:,0:i+1], train_labels, smoothing_constant)
        prediction = naiveBayesPredict(test_data_sorted[:,0:i+1], dist_ham, dist_spam, classProb_ham, classProb_spam)
        accuracy_ = accuracy(prediction, test_labels)
        performances.append(accuracy_)
    
    performances = np.asarray(performances)
    return performances
    
    


# In[35]:


performances_frequentWords = frequentWordsTrain(M, train_data, train_labels, test_data, test_labels, 10, 1)


# In[36]:


print(performances_frequentWords)


# In[37]:


# Save the accuracies for frequency selection as frequency_selection.csv

np.savetxt('frequency_selection.csv', performances_frequentWords, delimiter=',', fmt='%f')


# In[38]:


# Visualization of the frequency selection for each k value

figureNo += 1
size = performances_frequentWords.shape[0]
x_axis = np.arange(1,size+1)
plt.figure(figureNo, figsize=(10,8))
plt.title('Accuracy of the Frequency Selection for each k value')
plt.xlabel('k value')
plt.ylabel('Accuracy of the Model')
plt.plot(x_axis, performances_frequentWords)
plt.show()

