import numpy as np
import time
import pandas as pd
import time
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
start_time = time.time()
df = pd.read_csv('Training.csv')
k = df['Smoking_Pack-Years'].mean()
df['Smoking_Pack-Years'].fillna(df['Smoking_Pack-Years'].mean(),inplace=True)
df = df.dropna()
df = df.drop(df.columns[0],axis = 1)
#print(df.shape)
Y = df['KM_Overall_survival_censor']
X = df.iloc[:,0:17]
X_1 = pd.get_dummies(X)
#print(X_1)
accuracy_sc = np.zeros(50)
for i in range(0,50):
	X_train, X_test, y_train, y_test = train_test_split(X_1, Y, test_size = .20)
	classifier = DecisionTreeClassifier(criterion = 'entropy',max_leaf_nodes = i+2)
	classifier.fit(X_train, y_train)
	predictions = classifier.predict(X_test)
	accuracy_sc[i] = int(100*accuracy_score(y_test, predictions))

fig = plt.figure()
ax = fig.add_subplot(111)
ax = plt.axis([0,50,0,100])
ax = plt.plot(range(0,50),accuracy_sc)
fig.savefig('op.png')
# ax.show()