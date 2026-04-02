# # STEP-1
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Test the Setup
# print("Environment setup successful!")


# # STEP-2
# # Test Dataset Loading
# from preprocessing import load_dataset

# train_images, train_labels, test_images, test_labels = load_dataset()

# print("Training images:", train_images.shape)
# print("Training labels:", train_labels.shape)
# print("Testing images:", test_images.shape)
# print("Testing labels:", test_labels.shape)

# # Visualize a Digit
# image = train_images[0]
# label = train_labels[0]

# image = image.reshape(28,28)

# plt.imshow(image, cmap='gray')
# plt.title(f"Label: {label}")
# plt.show()


# # STEP-4
# # Use KNN in Main File
# from preprocessing import load_and_preprocess
# from algorithms.knn import KNN

# # Load data
# train_images, train_labels, test_images, test_labels = load_and_preprocess()

# # Initialize model
# knn_model = KNN(k=3)

# # Train (just storing data)
# knn_model.fit(train_images, train_labels)

# # Predict
# knn_predictions = knn_model.predict(test_images[:100])

# # Accuracy
# knn_accuracy = np.sum(knn_predictions == test_labels[:100]) / 100

# print("KNN Accuracy:", knn_accuracy)


# # STEP-5
# # Use Naive Bayes in Main File
# from preprocessing import load_and_preprocess
# from algorithms.naive_bayes import NaiveBayes
# import numpy as np

# # original normalized data
# train_images, train_labels, test_images, test_labels = load_and_preprocess()

# # Train model
# nb_model = NaiveBayes()
# nb_model.fit(train_images, train_labels)

# # for naive bayes only
# n=100

# # Predict
# nb_predictions = nb_model.predict(test_images[:n])

# # Accuracy
# nb_accuracy = np.sum(nb_predictions == test_labels[:n]) / n

# print("Naive Bayes Accuracy:", nb_accuracy)

# # STEP-6
# # Use Neural Network in Main File

# import numpy as np
# from algorithms.nn import NeuralNetwork
# from preprocessing import load_and_preprocess

# train_images, train_labels, test_images, test_labels = load_and_preprocess()

# # use smaller dataset for speed
# X_train = train_images[:10000]
# y_train = train_labels[:10000]

# X_test = test_images[:1000]
# y_test = test_labels[:1000]

# # model
# nn = NeuralNetwork()

# # train
# nn.fit(X_train, y_train, epochs=10)

# # predict
# nn_predictions = nn.predict(X_test)

# # accuracy
# nn_accuracy = np.mean(nn_predictions == y_test)

# print("Neural Network Accuracy:", nn_accuracy)

# STEP-7
# Use CNN in Main File
import numpy as np
from algorithms.nn import NeuralNetwork
from preprocessing import load_and_preprocess

train_images, train_labels, test_images, test_labels = load_and_preprocess()
from algorithms.cnn import SimpleCNN

cnn = SimpleCNN()

X_train = train_images[:3000]
y_train = train_labels[:3000]

X_test = test_images[:500]
y_test = test_labels[:500]

cnn.train(X_train, y_train, epochs=5)

# test
correct = 0
for i in range(len(X_test)):
    pred = np.argmax(cnn.forward(X_test[i].reshape(28,28)))
    if pred == y_test[i]:
        correct += 1

print("CNN Accuracy:", correct/len(X_test))