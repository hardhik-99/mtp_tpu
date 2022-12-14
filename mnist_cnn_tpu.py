# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:09:14 2022

@author: hardh
"""
from PIL import Image
import tensorflow as tf
import numpy as np
import tflite_runtime.interpreter as tflite
from tqdm import tqdm

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

mnist_tflite_filename = "mnist.tflite"

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

def tpu_tflite_predict(interpreter, data):
    input_data = data.reshape(1, 28, 28, 1).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

y_pred = []

for i in tqdm(range(x_test.shape[0])):
    pred = tpu_tflite_predict(interpreter, x_test[i])
    y_pred.append(pred.argmax(1)[0])
    #print("Pred: ", y_pred[i], " True: ", y_test[i])
    
print("TPU accuracy: ", 100 * np.sum(y_pred == y_test) / len(y_pred), "%")
