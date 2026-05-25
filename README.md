
# NAME : KAILASH PRABHU S
# REGNO : 212224240068
# DATE : 25/05/2026

# Aim:
To implement a perceptron for classification using Python

## THEORY:
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows:

XOR truth table
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:
![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).
![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

## Algorithm:

Step 1 : Initialize the input patterns for XOR Gate<BR>
Step 2: Initialize the desired output of the XOR Gate<BR>
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron  and 1 output neuron<BR>
Step 3: Repeat the  iteration  until the losses become constant and  minimum<BR>
    (i)  Compute the output using forward pass output<BR>
    (ii) Compute the error<BR>
	(iii) Compute the change in weight ‘dw’ by using backward progatation algorithm. <BR>
    (iv) Modify the weight as per delta rule.<BR>
    (v)  Append the losses in a list <BR>
Step 4 : Test for the XOR patterns.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt

# Input and Output
x = np.array([[0,0,1,1],
              [0,1,0,1]])

y = np.array([[0,1,1,0]])

# Network architecture
n_x = 2
n_h = 2
n_y = 1
m = x.shape[1]

# Learning rate
lr = 0.1

# Initialize weights and biases
w1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))

w2 = np.random.randn(n_y, n_h)
b2 = np.zeros((n_y, 1))

losses = []

# Sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward propagation
def forward_prop(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    return z1, a1, z2, a2

# Backpropagation
def back_prop(w1, w2, z1, a1, z2, a2, x, y):

    dz2 = a2 - y
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    dz1 = np.dot(w2.T, dz2) * a1 * (1 - a1)
    dw1 = np.dot(dz1, x.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    return dw1, db1, dw2, db2

# Training
iterations = 10000

for i in range(iterations):

    z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)

    loss = -(1/m) * np.sum(y*np.log(a2) + (1-y)*np.log(1-a2))
    losses.append(loss)

    dw1, db1, dw2, db2 = back_prop(w1, w2, z1, a1, z2, a2, x, y)

    # Update parameters
    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1

    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2

# Plot loss graph
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

# Prediction function
def predict(w1, b1, w2, b2, test):

    _, _, _, a2 = forward_prop(w1, b1, w2, b2, test)

    if a2 >= 0.5:
        print([i[0] for i in test], "-> 1")
    else:
        print([i[0] for i in test], "-> 0")

print("Input  Output")

predict(w1, b1, w2, b2, np.array([[0],[0]]))
predict(w1, b1, w2, b2, np.array([[0],[1]]))
predict(w1, b1, w2, b2, np.array([[1],[0]]))
predict(w1, b1, w2, b2, np.array([[1],[1]]))
```

## OUTPUT:

<img width="802" height="602" alt="image" src="https://github.com/user-attachments/assets/937edf1d-de22-4d5a-baec-2f95d001c8ed" />




## Result:
Thus, XOR classification problem can be solved using MLP in Python 
