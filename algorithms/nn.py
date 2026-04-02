# The neural network learns weights using forward propagation and updates them using backpropagation and gradient descent to minimize classification error.
import numpy as np

class NeuralNetwork:

    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr = 0.003):
        self.lr = lr

        # weights
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def one_hot(self, y):
        one_hot = np.zeros((y.size, 10))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

    def backward(self, X, y_true):
        m = X.shape[0]

        dZ2 = self.A2 - y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=25, batch_size=64):
        y_onehot = self.one_hot(y)
        n = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch)

        # compute loss on full data (for monitoring)
        y_pred_full = self.forward(X)
        loss = self.compute_loss(y_pred_full, y_onehot)

        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")



    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)