import sys
import os
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from gensamples import getsamples
import numpy as np

import matplotlib.pyplot as plt


class DecTree:
    def __init__(self):
        self.num_features = 0
        self.decision_tree = {}
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
        max_info_gain = -1

        root = {
            "root":{
                "gini":gini,
                "samples":samples,
                "value":value,
                "depth":0
            }
        }

        if gini != 0:
            # Make a decision and make a branch 
            # pick the decision that results in the highest information gain (slash creates the smallest gini)
            max_info_gain = -1
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
                    if len(value_left) == 1:
                        if y_left[0] == 0:
                            value_left = np.array([value_left[0], 0])
                        else:
                            value_left = np.array([0, value_left[0]])
                    value_right = np.unique(y_right, return_counts=True)[1].tolist()
                    if len(value_right) == 1:
                        if y_right[0] == 0:
                            value_right = np.array([value_right[0], 0])
                        else:
                            value_right = np.array([0, value_right[0]])

                    gini_left = 1.0 - sum((count / samples_left) ** 2 for count in value_left)
                    gini_right = 1.0 - sum((count / samples_right) ** 2 for count in value_right)

                    information_gain = gini - ((samples_left / samples) * gini_left + (samples_right / samples) * gini_right)
                    
                    # Store the best decision
                    if "decision" not in root["root"] or information_gain > max_info_gain:
                        max_info_gain = information_gain
                        root["root"]["decision"] = {
                            "feature": feature,
                            "threshold": threshold
                        }
                        root["root"]["left"] = {
                            "gini": gini_left,
                            "samples": samples_left,
                            "value": value_left,
                            "indices": left_indices,
                            "depth":1
                        }
                        root["root"]["right"] = {
                            "gini": gini_right,
                            "samples": samples_right,
                            "value": value_right,
                            "indices": right_indices,
                            "depth":1
                        }
            root["root"]["left"] = self.create_branch(y, X, root["root"]["left"])
            root["root"]["right"] = self.create_branch(y, X, root["root"]["right"])
        return root

    def create_branch(self, y, X, root):
        """ Recursive function to create branches of the decision tree """
        if root["depth"] < 5 and root["gini"] != 0:
            samples = root["samples"]
            max_info_gain = -1
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
                    y_left = y[left_indices]
                    y_right = y[right_indices]
                    # Calculate gini for left and right branches
                    samples_left = y_left.shape[0]
                    samples_right = y_right.shape[0]
                    if samples_left == 0 or samples_right == 0:
                        continue

                    value_left = np.unique(y_left, return_counts=True)[1].tolist()
                    if len(value_left) == 1:
                        if y_left[0] == 0:
                            value_left = np.array([value_left[0], 0])
                        else:
                            value_left = np.array([0, value_left[0]])
                    value_right = np.unique(y_right, return_counts=True)[1].tolist()
                    if len(value_right) == 1:
                        if y_right[0] == 0:
                            value_right = np.array([value_right[0], 0])
                        else:
                            value_right = np.array([0, value_right[0]])

                    gini_left = 1.0 - sum((count / samples_left) ** 2 for count in value_left)
                    gini_right = 1.0 - sum((count / samples_right) ** 2 for count in value_right)

                    # Weighted gini
                    information_gain = root["gini"] - ((samples_left / samples) * gini_left + (samples_right / samples) * gini_right)

                    # Store the best decision
                    if "decision" not in root or information_gain > max_info_gain:
                        max_info_gain = information_gain
                        root["decision"] = {
                            "feature": feature,
                            "threshold": threshold
                        }
                        root["left"] = {
                            "gini": gini_left,
                            "samples": samples_left,
                            "value": value_left,
                            "indices": left_indices,
                            "depth": root["depth"] + 1
                        }
                        root["right"] = {
                            "gini": gini_right,
                            "samples": samples_right,
                            "value": value_right,
                            "indices": right_indices,
                            "depth": root["depth"] + 1
                        }

            if root["left"]["gini"] != 0:
                root["left"] = self.create_branch(y, X, root["left"])
                
            
            if root["right"]["gini"] != 0:
                root["right"] = self.create_branch(y, X, root["right"])
            
        return root

        
    def predict(self, X,y=None):
        """ We are going to search the tree and find what value of the y the sample belongs to given the tree."""
        y_pred = np.array([])
        # Start at the root and work our way down
        if self.decision_tree == {}:
            raise Exception("The decision tree has not been created yet.")
        
        for i in range(X.shape[0]):
            y_pred = np.append(y_pred, self.predict_sample(X[i,:], self.decision_tree))
        if y is not None:
            accuracy = np.sum(y_pred.astype(int) == y.astype(int)) / y.shape[0]
            print(f"Accuracy: {accuracy*100:.2f}%") 
        return y_pred.astype(int)
    
    def predict_sample(self, x, tree):
        """ Predict the class of a single sample """
        node = tree["root"]
        while "decision" in node:
            feature = node["decision"]["feature"]
            threshold = node["decision"]["threshold"]
            if x[feature] <= threshold:
                node = node["left"]
            else:
                node = node["right"]
        # Return the class with the highest count
        return np.argmax(node["value"])
    def plot_decision(self,x,step_size=0.1):
        x1min, x1max = x[:,0].min() - 1, x[:,0].max() + 1
        x2min, x2max = x[:,1].min() - 1, x[:,1].max() + 1
        grid = np.mgrid[x1min:x1max:step_size, x2min:x2max:step_size].reshape(2, -1).T
        grid_pred = tree.predict(grid)
        for i in range(len(grid)):
            if grid_pred[i] == 0:
                plt.scatter(grid[i,0], grid[i,1], color='red')
            else:
                plt.scatter(grid[i,0], grid[i,1], color='blue')   
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2') 
        plt.legend(['Class 0 (red)','Class 1 (blue)'])
        plt.title('Data Points with True Labels')
        plt.show()  



class kNN:
    def __init__(self):
        self.k = 1
        self.distance_metric = 'euclidean'

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, y = None):
        y_pred = np.array([])
        
        for i in range(X.shape[0]):

            distances = []

            for j in range(self.X_train.shape[0]):

                distance = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))

                distances.append((distance, self.y_train[j]))

            distances.sort(key=lambda x: x[0])

            zero_votes = 0
            one_votes = 0

            for n in range(self.k):

                if distances[n][1] == 0:
                    zero_votes += 1
                else:
                    one_votes += 1  

            if zero_votes > one_votes:
                y_pred = np.append(y_pred, 0)
            else:
                y_pred = np.append(y_pred, 1)

        if y is not None:
            accuracy = np.sum(y_pred.astype(int) == y.astype(int)) / y.shape[0]
            print(f"kNN Accuracy: {accuracy*100:.2f}%")
        return y_pred.astype(int)




class NeuralNet:
    def __init__(self):
        """ Initialize weights and biases """
        self.hidden_weights = np.random.rand(2, 2)
        self.hidden_bias = np.random.rand(2)

        self.output_weights = np.random.rand(self.hidden_weights.shape[1], 1)
        self.output_bias = np.random.rand(1)
        self.learning_rate = 1
        self.num_epochs = 1
        
    def fit(self, X, y):
        """ Train the neural network using backpropagation """
        # Feature scaling
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

        for epoch in range(self.num_epochs):
            for i in range(X.shape[0]):
                # Initialize etas
                delta_hidden_weights = np.zeros(self.hidden_weights.shape)
                delta_hidden_bias = np.zeros(self.hidden_bias.shape)

                delta_output_weights = np.zeros(self.output_weights.shape)
                delta_output_bias = np.zeros(self.output_bias.shape)

                # Forward pass
                hidden_layer_input = np.dot(X[i], self.hidden_weights) + self.hidden_bias
                hidden_layer_output = self.sigmoid(hidden_layer_input)

                output_layer_input = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
                output = self.sigmoid(output_layer_input)

                # Backward pass
                output_error = output - y[i]
                output_delta = output_error * self.sigmoid_derivative(output)

                hidden_error = output_delta.dot(self.output_weights.T)
                hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_output)

                # Accumulate gradients
                delta_output_weights += np.outer(hidden_layer_output, output_delta)
                delta_output_bias += output_delta

                delta_hidden_weights += np.outer(X[i], hidden_delta)
                delta_hidden_bias += hidden_delta
                # Update weights and biases
                self.output_weights -= self.learning_rate * delta_output_weights
                self.output_bias -= self.learning_rate * delta_output_bias
                self.hidden_weights -= self.learning_rate * delta_hidden_weights
                self.hidden_bias -= self.learning_rate * delta_hidden_bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
                

    def predict(self, X, y=None):
        y_pred = np.array([])
        for i in range(X.shape[0]):
            # Forward pass
            hidden_layer_input = np.dot(X[i], self.hidden_weights) + self.hidden_bias
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
            output = self.sigmoid(output_layer_input)
            y_pred = np.append(y_pred, 1 if output >= 0.5 else 0)
        if y is not None:
            accuracy = np.sum(y_pred.astype(int) == y.astype(int)) / y.shape[0]
            print(f"Neural Network Accuracy: {accuracy*100:.2f}%")
        return y_pred.astype(int)
        

x = np.array([[0.1, 0.2], [0.3, 0.4],[0.5,0.4],[0.5, 0.6], [0.7, 0.8]])
y = np.array([1, 0,1,0,1])
x, y = getsamples()


"""
This works to fit and predict using the decision tree
tree = DecTree()
tree.fit(x, y)

x,y = getsamples()
print(tree.predict(x,y))
"""
"""
x,y = getsamples()
knn = kNN()
knn.fit(x, y)
knn.predict(x,y)"""

net = NeuralNet()

net.fit(x, y)
print(net.predict(x,y))
