# Naive Bayes from Scratch.
# “Which digit is most probable given the pixel values?”
# Formula: P(Class | Data) ∝ P(Data | Class) * P(Class)
# We compute this for all digits (0–9) and pick the highest.
# Naive Bayes assumes: All features (pixels) are independent

import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._pdf(c, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, c, x):
        mean = self.mean[c]
        # var = self.var[c] + 1e-6   # avoid divide by zero
        var = self.var[c] + 1e-2

        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / (denominator + 1e-9) + 1e-9   # 🔥 add epsilon here