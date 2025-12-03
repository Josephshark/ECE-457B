import sys
import os
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from gensamples import getsamples
import numpy as np



class DecTree:
    def __init__(self):
        self.num_features = 0
        self.decision_tree = {}
        self.depth = 0
        pass
    def fit(self, X, y):
        # Get the size of the dataset
        self.num_features = X.shape[1]
        # Check if the decision tree is empty
        self.decision_tree = {}
        self.decision_tree = self.create_tree(X, y)



    def create_tree(self, X, y):
        """ Function to create the decision tree """
        self.num_features = X.shape[1]
        # Calculate the root parameters
        samples = X.shape[0]
        value = np.unique(y, return_counts=True)[1].tolist()
        gini = 1.0 - sum((count / samples) ** 2 for count in value)

        self.decision_tree = {
            "root":{
                "gini":gini,
                "samples":samples,
                "value":value,
            }
        }
        # Make a decision and make a branch 
        # pick the decision that results in the highest information gain (slash creates the smallest gini)
        for feature in range(self.num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                # Split the dataset
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                y_left = y[left_indices]
                y_right = y[right_indices]

                # Calculate gini for left and right branches
                samples_left = y_left.shape[0]
                samples_right = y_right.shape[0]
                if samples_left == 0 or samples_right == 0:
                    continue

                value_left = np.unique(y_left, return_counts=True)[1].tolist()
                value_right = np.unique(y_right, return_counts=True)[1].tolist()

                gini_left = 1.0 - sum((count / samples_left) ** 2 for count in value_left)
                gini_right = 1.0 - sum((count / samples_right) ** 2 for count in value_right)

                # Weighted gini
                weighted_gini = (samples_left / samples) * gini_left + (samples_right / samples) * gini_right

                # Store the best decision
                if "decision" not in self.decision_tree["root"] or weighted_gini < self.decision_tree["root"].get("best_gini", float('inf')):
                    self.decision_tree["root"]["decision"] = {
                        "feature": feature,
                        "threshold": threshold
                    }
                    self.decision_tree["root"]["best_gini"] = weighted_gini
                    self.decision_tree["root"]["left"] = {
                        "gini": gini_left,
                        "samples": samples_left,
                        "value": value_left,
                        "indices": left_indices
                    }
                    self.decision_tree["root"]["right"] = {
                        "gini": gini_right,
                        "samples": samples_right,
                        "value": value_right,
                        "indices": right_indices
                    }
        self.create_branch(y, X, self.decision_tree["root"]["left"])
        self.create_branch(y, X, self.decision_tree["root"]["right"])
        self.depth += 1

    def create_branch(self, y, X, root):
        """ Recursive function to create branches of the decision tree """
        if self.depth < 5:
            samples = root["samples"]

            for feature in range(self.num_features):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left_indices = np.array([])
                    right_indices = np.array([])
                    for i in range(X.shape[0]):

                        # Calculate left and right indices based on the current feature and threshold
                        if root["indices"][i]:
                            left_indices = np.append(left_indices, X[i,feature] <= threshold)
                            right_indices = np.append(right_indices, X[i,feature] > threshold) 
                        else:
                            left_indices = np.append(left_indices, False)
                            right_indices = np.append(right_indices, False)
                    left_indices = left_indices.astype(bool)
                    right_indices = right_indices.astype(bool)
                    print(left_indices)
                    y_left = y[left_indices]
                    y_right = y[right_indices]
                    # Calculate gini for left and right branches
                    samples_left = y_left.shape[0]
                    samples_right = y_right.shape[0]
                    if samples_left == 0 or samples_right == 0:
                        continue

                    value_left = np.unique(y_left, return_counts=True)[1].tolist()
                    value_right = np.unique(y_right, return_counts=True)[1].tolist()

                    gini_left = 1.0 - sum((count / samples_left) ** 2 for count in value_left)
                    gini_right = 1.0 - sum((count / samples_right) ** 2 for count in value_right)

                    # Weighted gini
                    weighted_gini = (samples_left / samples) * gini_left + (samples_right / samples) * gini_right

                    # Store the best decision
                    if "decision" not in root or weighted_gini < root.get("best_gini", float('inf')):
                        self.decision_tree["decision"] = {
                            "feature": feature,
                            "threshold": threshold
                        }
                        root["best_gini"] = weighted_gini
                        root["left"] = {
                            "gini": gini_left,
                            "samples": samples_left,
                            "value": value_left,
                            "indices": left_indices
                        }
                        root["right"] = {
                            "gini": gini_right,
                            "samples": samples_right,
                            "value": value_right,
                            "indices": right_indices
                        }
            if self.decision_tree["root"]["left"]["gini"] != 0:
                self.create_branch(y, X, self.decision_tree["root"]["left"])
            if self.decision_tree["root"]["right"]["gini"] != 0:
                self.create_branch(y, X, self.decision_tree["root"]["right"])
        else:
            pass

        
    def predict(self, X):
        pass



class kNN:
    def __init__(self):
        pass
class NeuralNet:
    def __init__(self):
        pass


x, y = getsamples()


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

x = np.array([[0.1, 0.2], [0.3, 0.4],[0.5,0.4],[0.5, 0.6], [0.7, 0.8]])
y = np.array([0, 0,0,0,1])
tree = DecTree()
tree.create_tree(x,y)
#print(tree.decision_tree)