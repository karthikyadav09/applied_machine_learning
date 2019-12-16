
# coding: utf-8

# In[55]:

import numpy as np
import pandas as pd
import itertools
data = np.genfromtxt('spam.data.txt', delimiter=' ', usecols = range(0,58))
np.random.shuffle(data)


# In[56]:

x_col = data[:,:-1]
y_col = data[:,-1]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_col, y_col, test_size = .30)


# In[61]:

tempdata = np.column_stack((X_train,y_train))


# In[39]:

from sklearn.tree import DecisionTreeClassifier
# tempdata = []
# for i in range(0,len(X_train)):
#     x=[]
#     x=list(X_train[i])
#     x.append(y_train[i])
#     tempdata.append(x)
# tempdata = np.array(tempdata)


# In[62]:

pred = []
for i in range(0,20):
    clf = DecisionTreeClassifier()
    np.random.shuffle(np.array(tempdata))
    x_col1 = tempdata[:,:-1]
    y_col1 = tempdata[:,-1]
    X_small, X_temp, y_small, y_temp = train_test_split(x_col1, y_col1, test_size = .50)
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
    if summ > 10:
        ans[j] = 1
        
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, ans))


# In[63]:

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)
print(accuracy_score(y_test, prediction))


# In[ ]:



