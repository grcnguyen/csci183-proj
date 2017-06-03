import numpy as np
import math

#################################################################################

# Sigmoid Function
# @param x                 double

def sigmoidFunction(x):
    
    return 1 / (1 + math.exp(-x))

#################################################################################

# Cost Function
# @param X                 nxm Matrix
# @param y                 1xm Vector
# @param theta             1xn Vector
# @return

def computeCost(X, y, theta):
    
    m = y.length
    
    # theta * X    (1xn * nxm = 1xm)
    hypothesis = (np.array(theta) * np.array(X)).tolist()

    # h = sigmoid(theta * X)
    for idx in range(m):
        hypothesis[idx] = sigmoidFunction(hypothesis[idx])

    # log(h) * y' + log(1 - h) * (1 - y)'
    sum = 0
    for idx in range(m):
        sum += y[idx] * math.log(hypothesis[idx]) + (1 - y[idx]) * math.log(1 - hypothesis[idx])

    # -(log(h) * y' + log(1 - h) * (1 - y)') / m
    return -sum / m

#################################################################################

# Gradient Descent
# @param X                 nxm Matrix
# @param y                 1xm Vector
# @param theta             1xn Vector
# @param alpha             step factor
# @param maxInteration     Maximum iteration for convergence
# @return

def gradientDescent(X, y, theta, alpha, maxIteration):
    
    n = X.length
    m = y.length
    
    # size: maxIteration
    costHistory = []

    for it in range(maxIteration):

        # size: n
        temp_theta = []

        # size: m
        hypothesis = []

        # get values for hypothesis vector
        for i in range(m):
            
            # g_value = theta^T * X[i] <-- dot product
            g_value = 0
            for j in range(n):
                g_value += theta[j]*X[j][i]
            
            hypothesis[i] = sigmoidFunction(g_value)

        # for each feature, do gradient descent
        for i in range(n):

            sum = 0
            for j in range(m):
                sum += (hypothesis[j] - y[j])*X[i][j]

            temp_theta[i] = theta[i] - (alpha/m)*sum

        # update theta
        for i in range(n):
            theta[i] = temp_theta[i]

        # compute and record cost
        costHistory[it] = computeCost(X, y, theta)
        print("Done iteration: {}. Current Cost: {}.".format(it,costHistory[it]))

    return costHistory
