
# coding: utf-8

# In[6]:

import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[7]:

data = np.genfromtxt('dataset1.txt', delimiter=' ', usecols = range(0,2))
X = np.array(data)
clf = KMeans(n_clusters=2)
clf.fit(X)
labels = clf.labels_


# In[8]:

colors = ["r.","b."]
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]])
plt.show()


# In[9]:

data2 = np.genfromtxt('dataset2.txt', delimiter=' ', usecols = range(0,2))
X2 = np.array(data2)
clf2 = KMeans(n_clusters=3)
clf2.fit(X2)
labels2 = clf2.labels_


# In[10]:

colors2 = ["r.","b.","g."]
for j in range(len(labels2)):
    plt.plot(X2[j][0],X2[j][1],colors2[labels2[j]])
plt.show()


# In[ ]:



