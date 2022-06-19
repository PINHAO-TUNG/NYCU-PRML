import heapq
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # load data
    x_train = pd.read_csv('x_train.csv').values
    y_train = pd.read_csv('y_train.csv').values[:, 0]
    x_test = pd.read_csv('x_test.csv').values
    y_test = pd.read_csv('y_test.csv').values[:, 0]

    # set parameters
    x1 = []
    x2 = []

    # 1. Compute the mean vectors mi, (i=1,2) of each 2 classes
    for i in range(len(y_train)):
        if y_train[i] == 0:
            x1.append(x_train[i])
        else:
            x2.append(x_train[i])
    m1 = np.mean(x1, axis=0)
    m2 = np.mean(x2, axis=0)
    print(f'mean vector of class 1: {m1}\nmean vector of class 2: {m2}\n')

    # 2. Compute the Within-class scatter matrix SW
    covariance_x1 = x1 - m1
    covariance_x2 = x2 - m2
    sw1 = np.dot(covariance_x1.T, covariance_x1)
    sw2 = np.dot(covariance_x2.T, covariance_x2)
    sw = sw1 + sw2
    assert sw.shape == (2, 2)
    print(f"Within-class scatter matrix SW: {sw}\n")

    # 3. Compute the Between-class scatter matrix SB
    covariance = m2 - m1
    covariance = covariance.reshape(2, 1)
    sb = np.dot(covariance, covariance.T)
    assert sb.shape == (2, 2)
    print(f"Between-class scatter matrix SB: {sb}\n")

    # 4. Compute the Fisher’s linear discriminant
    w = np.dot(np.linalg.inv(sw), covariance)
    assert w.shape == (2, 1)
    print(f"Fisher’s linear discriminant: {w}\n")

    # 5. Project the test data by linear discriminant and get the class prediction by nearest-neighbor rule. Calculate the accuracy score
    # you can use accuracy_score function from sklearn.metric.accuracy_score
    # projection data
    counter0 = 0
    counter1 = 0
    k = 9
    y = np.zeros((len(y_train)))
    for i in range(len(y_train)):
        y[i] = np.dot(w.T, x_train[i])
    # Predict data
    y_pred = np.zeros((len(y_test)))
    abs_value = np.zeros((len(y)))
    for i in range(len(y_test)):
        y_test_projection = np.dot(w.T, x_test[i])
        for j in range(len(y)):
            abs_value[j] = abs(y_test_projection - y[j])
        smallest = heapq.nsmallest(k, abs_value)
        for m in range(len(y)):
            for n in range(k):
                if abs_value[m] == smallest[n]:
                    if y_train[m] == 0:
                        counter0 = counter0 + 1
                    else:
                        counter1 = counter1 + 1
        if counter0 > counter1:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
        counter0 = 0
        counter1 = 0
        abs_value = np.zeros((len(y)))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy of test-set {acc}")

    # 6. Plot the 1) best projection line on the training data and show the slope and intercept on the title (you can choose any value of intercept for better visualization) 2) colorize the data with each class 3) project all data points on your projection line. Your result should look like this image
    # data
    x_train_type0 = x_train[y_train == 0]
    x_train_type1 = x_train[y_train == 1]
    w_fig = np.dot(np.linalg.inv(sw), (m2-m1))
    intercep = 0
    projection_x_train = np.dot(
        x_train, w_fig).reshape(-1, 1) * w_fig / (np.dot(w_fig, w_fig))
    projection_x1_train = projection_x_train[y_train == 0]
    projection_x2_train = projection_x_train[y_train == 1]

    # projection line
    x = [np.min(projection_x_train[:, 0]), np.max(projection_x_train[:, 0])]
    slope = w[1] / w[0]
    y = [slope*x[0]+intercep, slope*x[1]+intercep]
    plt.plot(x, y, c='k')

    # data to projection data
    for i in range(len(x_train)):
        plt.plot(
            [x_train[i, 0], projection_x_train[i, 0]],
            [x_train[i, 1], projection_x_train[i, 1]], lw=0.4, alpha=0.1, c='b')

    # projected data point
    plt.scatter(projection_x1_train[:, 0],
                projection_x1_train[:, 1], s=5, c='r')
    plt.scatter(projection_x2_train[:, 0],
                projection_x2_train[:, 1], s=5, c='b')

    # data point
    plt.scatter(x_train_type0[:, 0], x_train_type0[:,
                1], s=5, c='r')
    plt.scatter(x_train_type1[:, 0], x_train_type1[:, 1],
                s=5, c='b')

    # show figure
    title = "Projection line: w = " + \
        str(float(slope)) + ", b = " + str(intercep)
    plt.title(title)
    plt.gca().set_aspect('equal')
    plt.show()
