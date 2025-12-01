import sys
import os
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from gensamples import getsamples
import numpy as np



class DecTree:
    def __init__(self,layer):
        self.num_features = 0
        self.decision_tree = {}
        pass
    def fit(self, X, y):
        # Get the size of the dataset
        self.num_features = X.shape[1]

        # Check if the decision tree is empty
        if self.decision_tree == {}:
            # if empty, create the decision tree
            self.decision_tree = self.create_tree(X, y)

        else:
            # Optimize the existing decision tree
            self.decision_tree = self.create_branch(X, y)

    def create_tree(self, X, y):
        """ Function to create the decision tree """
        samples = X.shape[0]
        value = np.unique(y, return_counts=True)[1].tolist()

        # Make a decision
        # pick the decision that results in the highest information gain

        for feature in range(self.num_features):
            pass

        self.decision_tree = {
            "root":{
                "decision":{
                    "feature":0,
                    "threshold":0
                },
                "gini":0,
                "samples":0,
                "value":[],
                "right":{
                    "gini":0,
                    "samples":0,
                    "value":[],
                },
                "left":{
                    "gini":0,
                    "samples":0,
                    "value":[],
                }
            }
        }

    def create_branch(self, X, y):
        """ Recursive function to create branches of the decision tree """
        
    def predict(self, X):
        pass



class kNN:
    def __init__(self):
        pass
class NeuralNet:
    def __init__(self):
        pass


x, y = getsamples()
print(x)

testTree = {
    "root":{
        "decision":{
            "feature":0,
            "threshold":0.5
        },
        "gini":0.5,
        "samples":200,
        "value":[1,2],
        "right":{
            "decision":{
                "feature":1,
                "threshold":1.5
            },
            "gini":0.0,
            "samples":200,
            "value":[3,4],
        },
        "left":{
            "decision":{
                "feature":1,
                "threshold":1.5
            },
            "gini":0.0,
            "samples":200,
            "value":[5,6],
        }
    }
}
branch = testTree['root']['left']
print(branch)