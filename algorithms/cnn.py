import numpy as np

class SimpleCNN:

    def __init__(self, num_filters=16, filter_size=3, lr=0.001):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.lr = lr

        # filters
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1

        # dense layer
        self.W = np.random.randn(num_filters * 13 * 13, 10) * 0.01
        self.b = np.zeros((1, 10))

    def convolve(self, image):
        h, w = image.shape
        f = self.filter_size

        output = np.zeros((self.num_filters, h - f + 1, w - f + 1))

        for n in range(self.num_filters):
            for i in range(h - f + 1):
                for j in range(w - f + 1):
                    region = image[i:i+f, j:j+f]
                    output[n, i, j] = np.sum(region * self.filters[n])

        return output
    
    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def forward(self, image):
        self.conv_out = self.convolve(image)
        self.relu_out = self.relu(self.conv_out)

        self.pooled = self.max_pool(self.relu_out)

        self.flat = self.pooled.flatten()

        logits = np.dot(self.flat, self.W) + self.b
        self.out = self.softmax(logits)

        return self.out
    
    def train(self, X, y, epochs=3):
        for epoch in range(epochs):
            correct = 0

            for i in range(len(X)):
                image = X[i].reshape(28, 28)
                label = y[i]

                probs = self.forward(image)

                if np.argmax(probs) == label:
                    correct += 1

                # one-hot
                target = np.zeros(10)
                target[label] = 1

                error = probs - target

                # update dense layer ONLY
                self.W -= self.lr * np.outer(self.flat, error)
                self.b -= self.lr * error

            print(f"Epoch {epoch+1}, Accuracy: {correct/len(X):.4f}")


    def max_pool(self, feature_map, size=2):
        num_filters, h, w = feature_map.shape
        output = np.zeros((num_filters, h//2, w//2))

        for n in range(num_filters):
            for i in range(0, h, 2):
                for j in range(0, w, 2):
                    region = feature_map[n, i:i+2, j:j+2]
                    output[n, i//2, j//2] = np.max(region)

        return output