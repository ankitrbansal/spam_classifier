# -*- coding: utf-8 -*-
"""
This program implements the Naive Bayes Spam Classifier to classify
emails as spam/non-spam. The preprocessed data is present in 
SPARSE.* files 

Author: Ankit Bansal
Email: arbansal@umich.edu
Copyrights reserved 2017
"""
import numpy as np
import matplotlib.pyplot as plt

def readData(file):
    """
    Reads data from files and returns as numpy arrays
    Input: file location
    Output: freq_spam, freq_ham, spam, ham, N number of data points
    """
    Y = [] 
    spam = 0
    ham = 0
    freq_spam = {i:0 for i in range(1,1449)}
    freq_ham = {i:0 for i in range(1,1449)}
    
    emails = []
    with open(file,'r') as f:
        for line in f:
            emails += [line.split('\n')]
        emails = np.asarray(emails)
        emails = emails[:,0]
        
        for i,value in enumerate(emails):
            email = [value.split()]
            email = np.asarray(email)
            email = np.squeeze(email)
            Y += [email[0]]
            if(email[0] == '1'):
                spam += 1
            else:
                ham += 1
            count = 0
            for j in range(1,email.shape[0]):
                word = int(email[j].split(':')[0])
                count = int(email[j].split(':')[1])
                if(Y[i] == '1'):
                    freq_spam[word] += count
                else:
                    freq_ham[word] += count
        Y = np.asarray(Y)
        N = Y.shape[0]
    return freq_spam, freq_ham, spam, ham, N
        
def estimateParams(freq_spam, freq_ham, spam, ham, N):
    """
    Computes the parametes from training data
    Input: freq_spam, freq_ham, spam, ham, N
    Output: Parameters 
    """
    class_prior_spam = (spam)/N
    class_prior_ham = (ham)/N
                      
    total_spam_count = 0
    total_ham_count = 0
    prob_wj_ham = []
    prob_wj_spam = []
    
    for word,count in freq_ham.items():
        total_ham_count += (1 + count)
        
    for word,count in freq_ham.items():
        prob_wj_ham += [(1 + freq_ham[word])/total_ham_count]
    
    for word,count in freq_spam.items():
        total_spam_count += (1 + count)
        
    for word,count in freq_spam.items():
        prob_wj_spam += [(1 + freq_spam[word])/total_spam_count]
        
    prob_wj_spam = np.asarray(prob_wj_spam)
    prob_wj_ham = np.asarray(prob_wj_ham)
    return class_prior_spam, prob_wj_spam, class_prior_ham, prob_wj_ham

def predict(class_prior_spam, prob_wj_spam, class_prior_ham, prob_wj_ham):
    """
    Predicts the label of each document/email as spam or non-spam
    Input: Parameters from training 
    Output: 
    """
    emails = []
    with open('SPARSE.TEST','r') as f:
        for line in f:
            emails += [line.split('\n')]
        emails = np.asarray(emails)
        emails = emails[:,0]
        wrong = 0
        for i,value in enumerate(emails):
            email = [value.split()]
            email = np.asarray(email)
            email = np.squeeze(email)
            
            freq_test = {i:0 for i in range(1,1449)}
            
            for j in range(1,email.shape[0]):
                word = int(email[j].split(':')[0])
                count = int(email[j].split(':')[1])
                freq_test[word] += count   
                         
            log_likelihood_spam = 0
            log_likelihood_ham = 0
            
            for word,count in freq_test.items():
                log_likelihood_spam += count*np.log(prob_wj_spam[word-1])
                log_likelihood_ham += count*np.log(prob_wj_ham[word-1])
                                  
            log_prob_spam = log_likelihood_spam + np.log(class_prior_spam)
            log_prob_ham = log_likelihood_ham + np.log(class_prior_ham)
            
            if(log_prob_spam > log_prob_ham):
                if(email[0] != '1'):
                    wrong += 1
            else:
                if(email[0] != '-1'):
                    wrong += 1
    return wrong/len(emails)*100

def spamIndicative(prob_wj_spam, prob_wj_ham):
    """
    Finds the top 5 tokens that are indicative of the SPAM class
    Input: Parameters
    Output: Top 5 tokens
    """
    spam_indicative = {j:np.log(np.divide(prob_wj_spam[j-1],prob_wj_ham[j-1])) for j in range(1,1449)}
    res = list(sorted(spam_indicative, key=spam_indicative.__getitem__, reverse=True))
    return res[0:5]
    
 
freq_spam_train, freq_ham_train, spam, ham, N = readData('SPARSE.TRAIN')
class_prior_spam, prob_wj_spam, class_prior_ham, prob_wj_ham = estimateParams(freq_spam_train, freq_ham_train, spam, ham, N)
test_error = predict(class_prior_spam, prob_wj_spam, class_prior_ham, prob_wj_ham)
print('Test error: ', test_error)

test_errors = []
X = ['50', '100', '200', '400', '800', '1400']
for i,val in enumerate(X):
    freq_spam_train, freq_ham_train, spam, ham, N = readData('SPARSE.TRAIN.'+val)
    class_prior_spam, prob_wj_spam, class_prior_ham, prob_wj_ham = estimateParams(freq_spam_train, freq_ham_train, spam, ham, N)
    test_error = predict(class_prior_spam, prob_wj_spam, class_prior_ham, prob_wj_ham)
    print('Test error for SPARSE.TRAIN.'+val+': ', test_error)
    test_errors += [test_error]

plt.plot(X,test_errors)
plt.ylabel('Test Error')
plt.xlabel('Training Size')

spamIndicative(prob_wj_spam, prob_wj_ham)