
# coding: utf-8

# In[4]:

import numpy as np
import pandas as pd
import itertools
import math
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


# In[5]:

df = pd.read_csv('linregdata.txt',header=None)
Female = np.where(df[0]=='F',1,0)
Infant = np.where(df[0]=='I',1,0)
Male = np.where(df[0]=='M',1,0)
df.insert(0,'M',Male,allow_duplicates=False)
df.insert(0,'I',Infant,allow_duplicates=False)
df.insert(0,'F',Female,allow_duplicates=False)      # (a)
df = df.drop(0,1)
label = df[8]
templabel = label
df = df.drop(df.columns[10],axis = 1)
pf = df
df = (df- np.mean(df))/np.std(df)
bias = np.ones(4177)
df.insert(0,'B',bias,allow_duplicates=False)        # (b)


# In[6]:

def mylinridgereg(X, Y, L):
    X = np.mat(X)
    Y = np.mat(Y)
    temp = X.transpose()*X
    Z = L*np.identity(len(temp))
    RegWeg = (np.linalg.inv(temp+Z)*X.transpose())*Y.transpose()
    return RegWeg
    
def mylinridgeregeval(X, weights):
    X = np.mat(X)
    Pred = X*weights
    return Pred

def meansquarederr(T, Tdash):
    T = np.mat(T)
    Tdash = np.mat(Tdash)
    MSE = ((T-Tdash.transpose()).transpose()*(T-Tdash.transpose())/len(T)).getA1()[0]
    return MSE                                      # (c)


# In[7]:

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size = .20)         # (d)


# In[8]:

Lrange = [0.01,0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,10,15,25,30,50,75,100]
Regarray = []
Pred = []
MSE = []
k = 0.1

for i in range(0,len(Lrange)):
    Regarray.append(mylinridgereg(X_train,y_train,Lrange[i]))
    Pred.append(mylinridgeregeval(X_test,Regarray[i]))
    MSE.append(meansquarederr(Pred[i],y_test))
from operator import itemgetter
minL = int(Lrange[(min(enumerate(MSE), key=itemgetter(1))[0])])


# In[9]:

minn = abs(Regarray[minL][0] - 0)
flag = 0
for i in range(1,len(Regarray[minL])):
    if minn > abs(Regarray[minL][i]-0):
        minn = abs(Regarray[minL][i]-0)
        flag = i
        

X_train = X_train.drop(X_train.columns[flag],axis = 1)          # (e)
X_test = X_test.drop(X_test.columns[flag],axis = 1)


# In[10]:

Regarray = []
Pred = []
MSE = []
for i in range(0,len(Lrange)):
    Regarray.append(mylinridgereg(X_train,y_train,Lrange[i]))
    Pred.append(mylinridgeregeval(X_test,Regarray[i]))
    MSE.append(meansquarederr(Pred[i],y_test))
minLL = Lrange[(min(enumerate(MSE), key=itemgetter(1))[0])]


# In[11]:

print (Lrange)


# In[12]:

import matplotlib.pyplot as plt                                    #  (f)
partition = [0.20,0.25,0.30,0.35,0.40,0.45]
minMSE22 = []
minlamindex = []
for i in range(0,len(partition)):
    MSE11 = []
    MSE22 = []
    for k in range(0,len(Lrange)):
        for j in range(0,25):
            X_train, X_test, y_train, y_test = train_test_split(pf, templabel, test_size = partition[i])
            X_train = (X_train - np.mean(X_train))/np.std(X_train)
            X_test = (X_test - np.mean(X_test))/np.std(X_test)
            MSE1 = 0
            MSE2 = 0
            Regarray = (mylinridgereg(X_train,y_train,Lrange[k]))
            Pred1 = (mylinridgeregeval(X_train,Regarray))
            Pred2 = (mylinridgeregeval(X_test,Regarray))
            MSE1 += (meansquarederr(Pred1,y_train))
            MSE2 += (meansquarederr(Pred2,y_test))
        MSE11.append(MSE1/25)
        MSE22.append(MSE2/25)
    minMSE22.append(min(MSE22))
    minlamindex.append(Lrange[(min(enumerate(MSE22), key=itemgetter(1))[0])])
    fig = plt.figure()
    plt.xscale('log')
    ax = fig.add_subplot(111)
    ax = plt.scatter(Lrange,MSE11)
    ax = plt.scatter(Lrange,MSE22)
    plt.xlabel('Lambda with partition = %f' %partition[i])
    plt.ylabel('MSE')
    plt.title('MSE vs Lambda')
    name = "e" + str(i) + ".png"
    fig.savefig(name)
    


# In[13]:

fig = plt.figure()
ax = fig.add_subplot(111)
ax = plt.scatter(partition,minMSE22)
plt.xlabel('Partition')
plt.ylabel('Minimum average MSE (testing)')
plt.title('Min avg MSE vs Partition')
fig.savefig('g1.png')


# In[14]:

fig = plt.figure()
ax = fig.add_subplot(111)
ax = plt.scatter(partition,minlamindex)
plt.xlabel('Partition')
plt.ylabel('Minimum lambda (testing)')
plt.title('Min lambda vs Partition')
fig.savefig('g2.png')                                   # (g)


# In[ ]:



