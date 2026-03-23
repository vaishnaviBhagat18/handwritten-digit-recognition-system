# STEP-1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Test the Setup
print("Environment setup successful!")


# STEP-2
# Test Dataset Loading
from preprocessing import load_dataset

train_images, train_labels, test_images, test_labels = load_dataset()

print("Training images:", train_images.shape)
print("Training labels:", train_labels.shape)
print("Testing images:", test_images.shape)
print("Testing labels:", test_labels.shape)

# Visualize a Digit
image = train_images[0]
label = train_labels[0]

image = image.reshape(28,28)

plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.show()


# STEP-4
# Use KNN in Main File
from preprocessing import load_and_preprocess
from algorithms.knn import KNN

# Load data
train_images, train_labels, test_images, test_labels = load_and_preprocess()

# Initialize model
knn_model = KNN(k=3)

# Train (just storing data)
knn_model.fit(train_images, train_labels)

# Predict
knn_predictions = knn_model.predict(test_images[:100])

# Accuracy
knn_accuracy = np.sum(knn_predictions == test_labels[:100]) / 100

print("KNN Accuracy:", knn_accuracy)


# STEP-5
# Use Naive Bayes in Main File
from preprocessing import load_and_preprocess
from algorithms.naive_bayes import NaiveBayes
import numpy as np

# original normalized data
train_images, train_labels, test_images, test_labels = load_and_preprocess()

# Train model
nb_model = NaiveBayes()
nb_model.fit(train_images, train_labels)

# for naive bayes only
n=100

# Predict
nb_predictions = nb_model.predict(test_images[:n])

# Accuracy
nb_accuracy = np.sum(nb_predictions == test_labels[:n]) / n

print("Naive Bayes Accuracy:", nb_accuracy)