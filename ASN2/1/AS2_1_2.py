
# coding: utf-8

# In[39]:

import numpy as np
import pandas as pd
import itertools
data = np.genfromtxt('spam.data.txt', delimiter=' ', usecols = range(0,58))
from sklearn.metrics import accuracy_score
np.random.shuffle(data)


# In[40]:

x_col = data[:,:-1]
y_col = data[:,-1]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_col, y_col, test_size = .30)


# In[41]:

tempdata = np.column_stack((X_train,y_train))


# In[42]:

from sklearn.tree import DecisionTreeClassifier
# tempdata = []
# for i in range(0,len(X_train)):
#     x=[]
#     x=list(X_train[i])
#     x.append(y_train[i])
#     tempdata.append(x)
# tempdata = np.array(tempdata)


# In[43]:

Sensitivity1 = np.zeros(57)
for k in range(1,58):
    pred = []
    for i in range(0,10):
        clf = DecisionTreeClassifier(max_features=k)
        np.random.shuffle(np.array(tempdata))
        x_col1 = tempdata[:,:-1]
        y_col1 = tempdata[:,-1]
        X_small, X_temp, y_small, y_temp = train_test_split(x_col1, y_col1, test_size = .30)
        clf.fit(X_small, y_small)
        pre = (clf.predict(X_test))
        pred.append(pre)

    ans = np.zeros(len(pred[0]))
    summ = 0
    for j in range(0,len(pred[0])):
        summ = 0
        for i in range(0,len(pred)):
            if pred[i][j] == 1:
                summ += 1
        if summ > 5:
            ans[j] = 1
    c1 = 0
    c2 = 0
    for i in range(0,len(ans)):
        if ans[i] == y_test[i] and y_test[i] == 0:
            c1 += 1
        if ans[i] != y_test[i] and y_test[i] == 0:
            c2 += 1
    Sensitivity1[k-1] = (float)(c1/(c1+c2))
    
    
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax = plt.axis([0,len(Sensitivity1),min(Sensitivity1)-0.01,max(Sensitivity1)+0.01])
ax = plt.plot(range(0,len(Sensitivity1)),Sensitivity1)
fig.savefig('Sensitivity.png')


# In[45]:

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(oob_score=True)
classifier.fit(X_train,y_train)
K = (classifier.oob_score_)
prediction = classifier.predict(X_test)


# In[47]:

Sensitivity2 = np.zeros(57)
for k in range(1,58):
    clf = RandomForestClassifier(max_features=k)
    clf.fit(X_train,y_train)
    ans = (clf.predict(X_test))
    c1 = 0
    c2 = 0
    for i in range(0,len(ans)):
        if ans[i] == y_test[i] and y_test[i] == 0:
            c1 += 1
        if ans[i] != y_test[i] and y_test[i] == 0:
            c2 += 1
    Sensitivity2[k-1] = (float)(c1/(c1+c2))
    
    
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax = plt.axis([0,len(Sensitivity2),min(Sensitivity2)-0.01,max(Sensitivity2)+0.01])
ax = plt.plot(range(0,len(Sensitivity2)),Sensitivity2)
fig.savefig('Sensitivity2.png')


# In[50]:

OOB = np.zeros(57)
for k in range(1,58):
    clf = RandomForestClassifier(max_features=k,oob_score=True)
    clf.fit(X_train,y_train)
    score = (clf.oob_score_)
    OOB[k-1] = score
    ans = (clf.predict(X_test))
    
    
    
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax = plt.axis([0,len(OOB),min(OOB)-0.01,max(OOB)+0.01])
ax = plt.plot(range(0,len(OOB)),OOB)
fig.savefig('OOB.png')


# In[ ]:



