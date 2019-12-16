import numpy as np
a = np.array([[1,2,3],[4,5,6],[0,0,1]])
b = a[a[:,1].argsort()]
print(b)