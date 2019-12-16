
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score


# In[2]:

vocab = []
with open('traindata','r') as fread1:
    for line in fread1:
        for word in line.split():
            vocab.append(word)


# In[3]:

stop = []
with open('stoplist','r') as fread2:
    for line in fread2:
        for word in line.split():
            stop.append(word)


# In[4]:

newList = []
for word in vocab:
    if word in stop:
        k = 1
    else:
        if word in newList:
            k = 2
        else:
            newList.append(word)
newList = sorted(newList)                                  # (a) part
k = 1


# In[5]:

trainlabel = np.zeros(322)
k = 0
with open('trainlabels','r') as fread3:
    for line in fread3:
        trainlabel[k] = int(line)
        k += 1


# In[6]:

data = []
k = 0
with open('traindata','r') as fread1:
    for line in fread1:
        x = []
        for i in range(0,693):
            if newList[i] in line:
                x.append(1)
            else:
                x.append(0)
        x.append(trainlabel[k])
        k += 1
        data.append(x)
        
data = np.array(data)                                  # (b) part


# In[7]:

fwrite = open('preprocessed.txt','w')
for i in range(0,693):
    fwrite.write(newList[i])
    if i < 692:
        fwrite.write(",")
fwrite.write("\n")
for i in range(0,len(data)):
    fwrite.write(str((int(data[i][0]))))
    for j in range(1,len(data[i])):
        fwrite.write(",")
        fwrite.write(str(int(data[i][j])))
    fwrite.write("\n")                                  # (c) part


# In[8]:

xtest = []
with open('traindata','r') as fread1:
    for line in fread1:
        x = []
        for i in range(0,693):
            if newList[i] in line:
                x.append(1)
            else:
                x.append(0)
        xtest.append(x)
        
xtest = np.array(xtest)        


# In[9]:

ytest = np.zeros(322)
k = 0
with open('testlabels','r') as fread3:
    for line in fread3:
        ytest[k] = int(line)
        k += 1


# In[10]:

from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()
x_col = data[:,:-1]
y_col = data[:,-1]
y_pred = gnb.fit(x_col,y_col).predict(xtest)


# In[11]:

print(accuracy_score(ytest, y_pred))


# In[ ]:



