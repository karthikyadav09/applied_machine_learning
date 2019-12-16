import numpy as np
import time
from sklearn.preprocessing import Imputer
start_time = time.time()
data = np.genfromtxt('read1.txt', delimiter=',', missing_values = '?', filling_values = np.nan, usecols = range(1,11))
imp = Imputer()
data = imp.fit_transform(data)
np.random.shuffle(data)
x_col = data[:,:-1]
y_col = data[:,-1]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_col, y_col, test_size = .20)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
print((time.time() - start_time))