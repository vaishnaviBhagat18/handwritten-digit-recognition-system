# STEP-2
# Create Dataset Loader Script
from mnist import MNIST
import numpy as np

def load_dataset():

    mndata = MNIST('dataset')

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels


# STEP-3
# Final Preprocessing Function
from mnist import MNIST
import numpy as np

def load_and_preprocess():

    mndata = MNIST('dataset')

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images = np.array(train_images) / 255.0
    test_images = np.array(test_images) / 255.0

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Reduce dataset size
    train_images = train_images[:10000]
    train_labels = train_labels[:10000]

    test_images = test_images[:2000]
    test_labels = test_labels[:2000]

    return train_images, train_labels, test_images, test_labels
