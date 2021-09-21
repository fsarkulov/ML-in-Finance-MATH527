import numpy as np
from numpy.linalg import norm
import copy
import os

def relu(x):
    return x*(np.sign(x)+1.)/2.
def sigmoid(x):
    return 1./(1.+np.exp(-x))
def softmax(x):
    return np.exp(x)/sum(np.exp(x))
def mynorm(Z):
    return np.sqrt(np.mean(Z**2))

def myANN(Y, Xtrain, Xpred, W01, W02, W03, b01, b02, b03):
    # Initialization of Weights and Biases
    W1 = copy.copy(W01)
    W2 = copy.copy(W02)
    W3 = copy.copy(W03)
    b1 = copy.copy(b01)
    b2 = copy.copy(b02)
    b3 = copy.copy(b03)

    # Initialize ad hoc variables
    k = 1
    change = 999

    # Begin the training loop
    while (change > 0.001 and k < 201):
        print("Iteration", k)

        ## Begin Feedforward (assume learning rate is one)
        # Hidden Layer 1
        Z1 = relu(W1 @ Xtrain + b1)
        # Hidden Layer 2
        Z2 = sigmoid(W2 @ Z1 + b2)
        # Output Layer
        Yhat = softmax(W3 @ Z2 + b3)
        # Find cross-entropy loss
        loss = -Y @ np.log(Yhat)
        print("Current Loss:",loss)

        ## Find gradient of loss with respect to the weights
        # Output Later
        dLdb3 = Yhat - Y
        dLdW3 = np.outer(dLdb3, Z2)
        # Hidden Layer 2
        dLdb2 = (W3.T @ (dLdb3)) * Z2 * (1-Z2)
        dLdW2 = np.outer(dLdb2,Z1)
        # Hidden Layer 1
        dLdb1 = (W2.T @ (dLdb2)) * Z1 * (1-Z1)
        dLdW1 = np.outer(dLdb1, Xtrain)

        ## Update Weights by Back Propagation
        # Output Layer
        b3 -= dLdb3 # (learning rate is one)
        W3 -= dLdW3
        # Hidden Layer 2
        b2 -= dLdb2
        W2 -= dLdW2
        # Hidden Layer 1
        b1 -= dLdb1
        W1 -= dLdW1

        change = norm(dLdb1)+norm(dLdb2)+norm(dLdb3)+norm(dLdW1)+norm(dLdW2)+norm(dLdW3)
        k += 1

    Z1pred = W1 @ Xpred + b1
    Z2pred = W2 @ relu(Z1pred) + b2
    Z3pred = W3 @ sigmoid(Z2pred) + b3
    Ypred = softmax(Z3pred)
    print("")
    print("Summary")
    print("Target Y \n", Y)
    print("Fitted Ytrain \n", Yhat)
    print("Xpred\n", Xpred)
    print("Fitted Ypred \n", Ypred)
    print("Weight Matrix 1 \n", W1)
    print("Bias Vector 1 \n", b1)
    print("Weight Matrix 2 \n", W2)
    print("Bias Vector 2 \n", b2)
    print("Weight Matrix 3 \n", W3)
    print("Bias Vector 3 \n", b3)

    #Defining initial weights for the network
W0_1 = np.array([[0.1,0.3,0.7], [0.9,0.4,0.4]])
b_1 = np.array([1.,1.])

W0_2 = np.array([[0.4,0.3], [0.7,0.2]])
b_2 = np.array([1.,1.])
W0_3 = np.array([[0.5,0.6], [0.6,0.7], [0.3,0.2]])
b_3 = np.array([1.,1.,1.])

X_train = np.array([0.1,0.7,0.3])
YY = np.array([1.,0.,0.])
X_pred = X_train
myANN(YY, X_train, X_pred, W0_1, W0_2,W0_3,b_1,b_2,b_3)
