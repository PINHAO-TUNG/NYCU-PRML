import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# function y = beta0 + beta1 * x
def linear_function(x, beta0, beta1):
    y = beta0 + x * beta1 
    return y

# calcuate loss function(Mean Square Error)
def mse_function(y_of_x, y):
    mse = 0.5*np.average((y_of_x-y)**2)
    return mse

# gradient descent
def gd_function(beta0, beta1, learning_rate, batch_size, batch_num, x_train, y_train):
    batch = np.random.randint(batch_num)
    temporary0, temporary1 = 0.0, 0.0

    #update weignts  
    for i in range(batch*batch_size, (batch+1)*batch_size):
        temporary0 += (linear_function(x_train[i], beta0, beta1) - y_train[i]) 
        temporary1 += (linear_function(x_train[i], beta0, beta1) - y_train[i])* x_train[i]   
    temporary0 = temporary0/batch_size
    temporary1 = temporary1/batch_size
    beta0 = beta0 - learning_rate * temporary0
    beta1 = beta1 - learning_rate * temporary1

    return beta0, beta1

# Train model
def train_model(beta0, beta1, learning_rate, batch_size, batch_num, x_train, y_train, iteration, converge_test):
    count = 0
    data_save = {'beta0': [], 'beta1': [], 'mse': []}
    
    while count < iteration: 
        mse = mse_function(linear_function(x_train, beta0, beta1), y_train)
        #check converge
        if len(data_save['mse']) > 0:
            if abs(mse - data_save['mse'][-1]) < converge_test:
                count += 1
            else:
                count = 0

        data_save['beta0'].append(beta0)
        data_save['beta1'].append(beta1)
        data_save['mse'].append(mse)
        beta0, beta1 = gd_function(beta0, beta1, learning_rate, batch_size, batch_num, x_train, y_train)
        
    return beta0, beta1, data_save

# main 
if __name__ == '__main__':
    # open train data
    train_df = pd.read_csv("train_data.csv")
    x_train, y_train = train_df['x_train'], train_df['y_train']

    beta0_random = np.random.normal()
    beta1_random = np.random.normal()

    # model type: Batch Gradient Descent
    learning_rate = 1e-2
    iteration = 100
    batch_size = 500
    batch_num = len(x_train)//batch_size
    converge_test = 3*1e-5
    beta0, beta1, data_save = train_model(beta0_random, beta1_random, learning_rate, batch_size, batch_num, x_train, y_train, iteration, converge_test)           

    # model type: Mini-Batch Gradient Descent
    mb_learning_rate = 1e-2
    mb_iteration = 100
    mb_batch_size = 50
    mb_batch_num = len(x_train)//mb_batch_size
    converge_test = 3*1e-5
    mb_beta0, mb_beta1, mb_data_save = train_model(beta0_random, beta1_random, mb_learning_rate, mb_batch_size, mb_batch_num, x_train, y_train, mb_iteration, converge_test)
    
    # model type: Stochastic Gradient Descent
    sto_learning_rate = 1e-2
    sto_iteration = 400
    sto_batch_size = 1
    sto_batch_num = len(x_train)//sto_batch_size
    converge_test = 1e-3
    sto_beta0, sto_beta1, sto_data_save = train_model(beta0_random, beta1_random, sto_learning_rate, sto_batch_size, sto_batch_num, x_train, y_train, sto_iteration, converge_test)
    
    # open test data
    test_data = pd.read_csv("test_data.csv")
    x_test, y_test = test_data['x_test'], test_data['y_test']

    # test model
    # Batch Gradient Descent test
    y_predict = linear_function(x_test, beta0, beta1)
    mse_loss = mse_function(y_predict, y_test)

    # Mini-Batch Gradient Descent test
    mb_y_predict = linear_function(x_test, mb_beta0, mb_beta1)
    mb_mse_loss = mse_function(mb_y_predict, y_test)

    # Stochastic Gradient Descent
    sto_y_predict = linear_function(x_test, sto_beta0, sto_beta1)
    sto_mse_loss = mse_function(sto_y_predict, y_test)

    # Output 
    # Batch Gradient Descent
    #print("MSE run times:", len(data_save['mse']))
    print("loss:", mse_loss)
    print("beta0:", beta0, ", beta1:", beta1)

    # Mini-Batch Gradient Descent
    print("Mini-Batch loss:", mb_mse_loss)
    print("Mini-Batch beta0:", mb_beta0, ", Mini-Batch beta1:", mb_beta1)

    # Stochastic Gradient Descent
    print("Stochastic loss:", sto_mse_loss)
    print("Stochastic beta0:", sto_beta0, ", Stochastic beta1:", sto_beta1)

    # figure
    fig0 = plt.figure()

    ax01 = fig0.add_subplot(2, 1, 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax01.plot(x_test, y_test, '.k')
    ax01.plot(x_test, beta0 + beta1 * x_test, 'r', label='Training line')
    plt.legend(loc='lower right')

    ax02 = fig0.add_subplot(2, 1, 2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    ax02.plot(data_save['mse'], 'r', label='Training loss')
    plt.legend()
    plt.show()

    # figure compare
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.plot(x_test, y_test, '.k')
    ax1.plot(x_test, beta0 + beta1 * x_test, 'r', label='Training line')
    ax1.plot(x_test, mb_beta0 + mb_beta1 * x_test, 'b', label='MB-Train line')
    ax1.plot(x_test, sto_beta0 + sto_beta1 * x_test, 'g', label='STO-Train line')
    plt.legend(loc='lower right')

    ax2 = fig.add_subplot(2, 1, 2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    ax2.plot(data_save['mse'], 'r', label='Training loss')
    ax2.plot(mb_data_save['mse'], 'b', label='MB-Train loss')
    ax2.plot(sto_data_save['mse'], 'g', label='STO-Train loss')
    plt.legend()
    plt.show()
