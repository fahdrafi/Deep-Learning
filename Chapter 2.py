# Databricks notebook source
import keras
keras.__version__

# COMMAND ----------

from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical

# COMMAND ----------

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# COMMAND ----------

network.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# COMMAND ----------

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28*28)).astype('float32')/255
test_images = test_images.reshape((10000, 28*28)).astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# COMMAND ----------

network.fit(train_images, train_labels, epochs=5, batch_size=128)

# COMMAND ----------

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test accuracy: ', test_acc)

# COMMAND ----------

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(train_images[4], cmap=plt.cm.binary)
display(plt.show())