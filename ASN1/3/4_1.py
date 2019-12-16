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
X_train, X_test, y_train, y_test = train_test_split(X_1, Y, test_size = .20)
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(accuracy_score(y_test, predictions))
print((time.time() - start_time))

