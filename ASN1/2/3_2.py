import math
import numpy as np
import time
from sklearn.preprocessing import Imputer
def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))
class KNN():
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X_test):
        guess = []
        for j in X_test:
            l = self.closest(j)
            guess.append(l)
        return guess

    def closest(self, j):
        min_dist = get_distance(j,self.X_train[0])
        min_index = 0
        for i in range(1,len(self.X_train)):
            dist = get_distance(j, self.X_train[i])
            if dist < min_dist:
                min_dist = dist
                min_index = i
        return self.y_train[min_index]

start_time = time.time()
data = np.genfromtxt('read1.txt', delimiter=',', missing_values = '?', filling_values = np.nan, usecols = range(1,11))
imp = Imputer()
data = imp.fit_transform(data)
np.random.shuffle(data)
x_col = data[:,:-1]
y_col = data[:,-1]
tot_points=data.shape[0]
train_count=int(0.8 * tot_points)
test_count = tot_points - train_count
X_train = x_col[:train_count,:]
y_train = y_col[:train_count]
X_test = x_col[train_count:,:]
y_test = y_col[train_count:]
classifier = KNN()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
print((time.time() - start_time))