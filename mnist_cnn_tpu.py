# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:09:14 2022

@author: hardh
"""
import os
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tflite_runtime.interpreter as tflite

def open_jpg(filename):
    return Image.open(filename)

def save_jpg(a, filename):
    ia = (a * 255).astype(np.uint8)
    ia = np.reshape(ia, (28,28))
    Image.fromarray(ia).save(filename)
    
def show_img(a, dim=(28, 28)):
    img = np.reshape(a, dim)
    imgplot = plt.imshow(img, cmap = "gray")
    plt.show()

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

print(tf.test.gpu_device_name())

#Model
def get_model():
    model = models.Sequential(name = "mnist-CNN-tf")
    model.add(layers.Conv2D(8, 3, padding = 'same', activation='relu', input_shape=(28, 28, 1), name = "layer1"))
    model.add(layers.AveragePooling2D((2, 2), name = "layer2"))
    model.add(layers.Conv2D(13, 3, padding = 'same', activation='relu', name = "layer3"))
    model.add(layers.Flatten(name = "layer4"))
    model.add(layers.Dense(14*14*13, name = "layer5"))
    model.add(layers.Dense(10, name = "layer6"))
    return model

model = get_model()
model.summary()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test))


#TFlite
def convert_to_tflite(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

mnist_tflite_filename = "mnist.tflite"
convert_to_tflite(model, mnist_tflite_filename)

def load_tflite_model(modelpath):
    # Load the TFLite model and allocate tensors.
    # Load using CPU
    # interpreter = tf.lite.Interpreter(model_path=modelpath)
    # Load using TPU
    interpreter = tflite.Interpreter(model_path=modelpath,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(mnist_tflite_filename)

def tflite_predict(interpreter, data):
    input_data = data.reshape(1, 28, 28, 1).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

pred = tflite_predict(interpreter, x_test[9])
print(pred.argmax(1), y_test[9])

show_img(x_test[9])

