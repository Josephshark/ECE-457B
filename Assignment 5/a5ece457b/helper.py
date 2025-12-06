import matplotlib.pyplot as plt

def plot_data(x,y):
    for i in range(len(x)):
        if y[i] == 0:
            plt.scatter(x[i][0], x[i][1], color='red', marker='x')
        else:
            plt.scatter(x[i][0], x[i][1], color='blue', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2') 
    plt.title('Data Visualization')
    plt.legend(['Class 0: red', 'Class 1: blue'])
    plt.show()