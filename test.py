import numpy as np

X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]
X = np.array(X)
print(X.shape)

rgen = np.random.RandomState(1)
"""
The weights were specified to need to be between [0.25 − 0.125],

To get the position statistically we the average value as the mean and the difference between
the lowest value (or highest value) and the mean as 3 times the std.  
"""
#mean_specified_weight = (0.25 + 0.125)/2
#std_of_weights = 3*(mean_specified_weight - 0.125)**(1/2)
w = rgen.normal(loc=1, scale=0.01, size=X.shape[1])
print(w.shape)


y = [2, 12]

y = np.array(y)

print(y.shape)