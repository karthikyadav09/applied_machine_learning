import numpy as np
print("Enter a number")
n = int(input());
a = np.fromfunction(lambda i, j: (i+j)%2, (n, n), dtype=int)
print (a)