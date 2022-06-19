import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def shuffle_function(length):
    shuf = np.arange(length)
    np.random.shuffle(shuf)
    return shuf


def cross_validation(x_train, y_train, k=5):
    k_fold = []
    n_sample = len(y_train)
    shuf = shuffle_function(n_sample)
    first_fold = n_sample % k
    fold_size = n_sample // k + 1
    for i in range(k):
        if i < first_fold:
            size = fold_size
            start = size * i
        else:
            size = fold_size - 1
            start = first_fold * (fold_size) + (i - first_fold) * size
        validation = shuf[start: start + size]
        training = np.delete(shuf, np.arange(start, start + size))
        print("Split: %s, Training index: %s, Validation index: %s"
              % (i+1, training, validation))
        k_fold.append([training, validation])
    return k_fold


def Fold_function(x_train, y_train, kfold_data, k):
    train, val = [], []
    train.append(np.zeros((len(kfold_data[k][0]), len(x_train[0]))))
    train.append(np.zeros((len(kfold_data[k][0]), 1)))
    for i in range(len(kfold_data[k][0])):
        train[0][i] = x_train[kfold_data[k][0][i]]
        train[1][i] = y_train[kfold_data[k][0][i]]
    val.append(np.zeros((len(kfold_data[k][1]), len(x_train[0]))))
    val.append(np.zeros((len(kfold_data[k][1]), 1)))
    for i in range(len(kfold_data[k][1])):
        val[0][i] = x_train[kfold_data[k][1][i]]
        val[1][i] = y_train[kfold_data[k][1][i]]
    return train, val


def grid_search(x_train, y_train, kfold_data, C, gamma):
    parameter_logger = np.zeros((len(C), len(gamma)))
    best_c, best_gamma, best_acc = None, None, 0
    for i in range(len(C)):
        for j in range(len(gamma)):
            acc = np.zeros(len(kfold_data))
            for k in range(len(kfold_data)):
                clf = SVC(C=C[i], kernel='rbf', gamma=gamma[j])
                train, val = Fold_function(x_train, y_train, kfold_data, k)
                clf.fit(train[0], train[1].ravel())
                y_pred = clf.predict(val[0])
                acc[k] = accuracy_score(val[1], y_pred)
            # the average score of validation folds
            parameter_logger[i][j] = np.average(acc)
            # choice best parameter
            if parameter_logger[i][j] >= best_acc:
                best_c = C[i]
                best_gamma = gamma[j]
                best_acc = parameter_logger[i][j]
    return [best_c, best_gamma], parameter_logger


if __name__ == "__main__":
    # ## Load data
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    # 550 data with 300 features
    # print(x_train.shape)

    # It's a binary classification problem
    # print(np.unique(y_train))

    # ## Question 1
    # K-fold data partition: Implement the K-fold cross-validation function. Your function should take K as an argument and return a list of lists (len(list) should equal to K), which contains K elements. Each element is a list contains two parts, the first part contains the index of all training folds, e.g. Fold 2 to Fold 5 in split 1. The second part contains the index of validation fold, e.g. Fold 1 in  split 1
    kfold_data = cross_validation(x_train, y_train, k=10)
    # should contain 10 fold of data
    assert len(kfold_data) == 10
    # each element should contain train fold and validation fold
    assert len(kfold_data[0]) == 2
    # The number of data in each validation fold should equal to training data divieded by K
    assert kfold_data[0][1].shape[0] == 55

    # ## Question 2
    # Using sklearn.svm.SVC to train a classifier on the provided train set and conduct the grid search of “C”, “kernel” and “gamma” to find the best parameters by cross-validation.
    C = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]
    best_parameters, parameters_logger = grid_search(
        x_train, y_train, kfold_data, C, gamma)
    print(
        f"\nbest_parameters:\nC = {best_parameters[0]} gamma = {best_parameters[1]}")

    # ## Question 3
    # Plot the grid search results of your SVM. The x, y represents the hyperparameters of “gamma” and “C”, respectively. And the color represents the average score of validation folds
    # You reults should be look like the reference image ![image](https://miro.medium.com/max/1296/1*wGWTup9r4cVytB5MOnsjdQ.png)
    picture = plt.figure(figsize=((8, 5)))
    ax0 = picture.add_subplot(1, 1, 1)
    ax0.set_title("Hyperpparameter Gridsearch")
    ax0.set_xlabel('Gamma Parameter')
    ax0.set_ylabel('C Parameter')
    ax0.set_xticks(np.arange(len(gamma)))
    ax0.set_yticks(np.arange(len(C)))
    plt.yticks(rotation=90)
    ax0.set_xticklabels(gamma)
    ax0.set_yticklabels(C)
    for i in range(len(C)):
        for j in range(len(gamma)):
            ax0.text(j, i, f'{parameters_logger[i, j]:.2f}',
                     ha="center", va="center", color="w")
    color_bar = ax0.imshow(parameters_logger, cmap="RdBu")
    ax0.figure.colorbar(color_bar, ax=ax0)
    plt.show()

    # ## Question 4
    # Train your SVM model by the best parameters you found from question 2 on the whole training set and evaluate the performance on the test set.
    best_model = SVC(C=best_parameters[0],
                     kernel='rbf', gamma=best_parameters[1])
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    print("Accuracy score: ", accuracy_score(y_pred, y_test))
