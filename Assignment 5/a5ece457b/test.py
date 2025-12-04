import numpy as np
test = [1]

print(len(test))  # This will print 1

a = np.array([1,1])
b = np.array([1,2])
c = a - b
print(c)  # This will print [0 -1]
print(np.linalg.norm(c))  # This will print the Euclidean distance between a and b