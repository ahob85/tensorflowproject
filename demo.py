from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# This is a 3D dataset (60000 x 28 x 28)
print(train_images.shape)

# This is a 1D dataset
print(len(train_labels))
print(train_labels)

# This is a 3D dataset (10000 x 28 x 28)
print(test_images.shape)

# This is a 1D dataset
print(len(test_labels))
print(test_labels)

# Preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Make all images grayscale
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the model by first configuring the layers of the model
model = keras.Sequential([
    # First layer transforms 2D array to 1D array.
    keras.layers.Flatten(input_shape=(28, 28)),
    # 128 nodes (neurons)
    keras.layers.Dense(128, activation=tf.nn.relu),
    # each node has score that indicates probability that current image is one of 10 classes
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy', test_acc)

# Make predictions
predictions = model.predict(test_images)
