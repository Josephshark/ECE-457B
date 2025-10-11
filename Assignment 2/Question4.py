# coding: utf-8


import sys
#from python_environment_check import check_packages
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:



'''

d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'pandas': '1.3.2'
}
check_packages(d)
'''

# # Chapter 2 - Training Machine Learning Algorithms for Classification

# ### Overview
# 

# - [Artificial neurons – a brief glimpse into the early history of machine learning](#Artificial-neurons-a-brief-glimpse-into-the-early-history-of-machine-learning)
#     - [The formal definition of an artificial neuron](#The-formal-definition-of-an-artificial-neuron)
#     - [The perceptron learning rule](#The-perceptron-learning-rule)
# - [Implementing a perceptron learning algorithm in Python](#Implementing-a-perceptron-learning-algorithm-in-Python)
#     - [An object-oriented perceptron API](#An-object-oriented-perceptron-API)
#     - [Training a perceptron model on the Iris dataset](#Training-a-perceptron-model-on-the-Iris-dataset)
# - [Adaptive linear neurons and the convergence of learning](#Adaptive-linear-neurons-and-the-convergence-of-learning)
#     - [Minimizing cost functions with gradient descent](#Minimizing-cost-functions-with-gradient-descent)
#     - [Implementing an Adaptive Linear Neuron in Python](#Implementing-an-Adaptive-Linear-Neuron-in-Python)
#     - [Improving gradient descent through feature scaling](#Improving-gradient-descent-through-feature-scaling)
#     - [Large scale machine learning and stochastic gradient descent](#Large-scale-machine-learning-and-stochastic-gradient-descent)
# - [Summary](#Summary)






# # Artificial neurons - a brief glimpse into the early history of machine learning





# ## The formal definition of an artificial neuron





# ## The perceptron learning rule










# # Implementing a perceptron learning algorithm in Python

# ## An object-oriented perceptron API


# ## Training a perceptron model on the Iris dataset

# ...

# ### Reading-in the Iris data




try:
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')
    
except Exception:
    s = 'iris.data'
    print('From local Iris path:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')
    



# ### Training the perceptron model



# ### A function for plotting decision regions





def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')




class AdalineGD:
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    losses_ : list
      Mean squared eror loss function values in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        #self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.w_ = np.array([0.25, -0.125, 0.0625])
        self.b_ = np.float64(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression (as we will see later), 
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            
            #for w_j in range(self.w_.shape[0]):
            #    self.w_[w_j] += self.eta * (2.0 * (X[:, w_j]*errors)).mean()
            
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# # Adaptive linear neurons and the convergence of learning

# ...

# ## Minimizing cost functions with gradient descent

# And Data
# Inputs (features)
data = np.loadtxt('Assignment 2/rectangle.data', delimiter=',')  # load CSV-style data
X = data[:, :3]   # first 3 columns → features
y = data[:, 3]    # last column → labels

rectLearner = AdalineGD(0.01,10)
rectLearner.fit(X, y)
'''
plot_decision_regions(X, y, classifier=rectLearner, resolution=0.1)
plt.title('Adaline - Gradient descent')
plt.xlabel('X')
plt.ylabel('Output')
plt.legend(loc='upper left')

plt.show()
'''
plt.plot(range(1, len(rectLearner.losses_) + 1), rectLearner.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.show()

print("w: ["+str(rectLearner.w_[0])+", "+str(rectLearner.w_[1])+", "+str(rectLearner.w_[2])+"], b: "+str(rectLearner.b_)+".")

print(np.sum(rectLearner.predict(X) == y))