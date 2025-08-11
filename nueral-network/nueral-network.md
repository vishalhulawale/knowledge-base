# Neural Network
Human brain contain hundreds of billions of neurons, which are connected to each other by synapses. These synapses are used to transmit signals between neurons.

Artificial Neural Networks (ANNs) are inspired by the structure and function of the human brain. They consist of interconnected nodes (neurons) that process information in a way similar to biological neural networks.

## Problem
||Input|Output|
|---|---|---|
Example 1|0 , 0 , 1 | 0 |
Example 2|1 , 1 , 1 | 1 |
Example 3|1 , 0 , 1 | 1 |
Example 4|0 , 1 , 1 | 0 |
| | | |
|New Situation|1 , 0 , 0 | ? |



## Designing our architcecture
We have 3 inputs and 1 output. We can use a single neuron to solve this problem, but we will use a neural network with 3 inputs and 1 output to demonstrate the concept.








































https://colab.research.google.com/drive/1-lbGiUqYh43GlXt0-WlFOgVqIMouah9L?usp=sharing

MLPClassifier

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Multilayer Perceptron Classifier

Activation Functions:
- Identity: No operation is made, f(x) = x
- Logistic: Sigmoid function, f(x) = 1 / (1 + exp(-x)). If we have two classes and only one output neuron, we can use this function to predict probabilities.
- Hyperbolic Tangent: f(x) = tanh(x)
- ReLU (Rectified Linear Unit): f(x) = max(0, x)
- Softmax: Returns the probability for each one of classes. It is used in the output layer of a neural network for multi-class classification problems.

Solvers:
Solvers are used to optimize the weights of the neural network.
- Adam: A stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
- SGD (Stochastic Gradient Descent): A stochastic gradient descent method that uses a fixed learning rate.
- LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno): An optimization algorithm that approximates the Hessian matrix to find the minimum of a function.

Learning Rate:
- Constant: The learning rate is fixed.
- Invscaling: The learning rate is decreased at each iteration using an inverse scaling exponent.
- Adaptive: The learning rate is adapted based on the training progress.

Batch Size:
Size of the mini-batches used in stochastic optimization. The default is 200, but it can be set to a different value.

Hidden Layer Sizes:
A tuple representing the number of neurons in each hidden layer. For example, (100,) means one hidden layer with 100 neurons, while (100, 50) means two hidden layers with 100 and 50 neurons respectively.


