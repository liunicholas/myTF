# Python 3.8.6
import time

# tensorflow 2.4.0
# matplotlib 3.3.3
# numpy 1.19.4
# opencv-python 4.4.0
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets as datasets
import tensorflow.keras.optimizers as optimizers
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cv2

fout = open('test.txt', 'w')
now = time.strftime("%H:%M:%S", time.localtime())
print("[TIMER] Process Time:", now)
print("[TIMER] Process Time:", now, file = fout, flush = True)

# File location to save to or load from
MODEL_SAVE_PATH = './cifar_net.pth'
# Set to zero to use above saved model
TRAIN_EPOCHS = 20
# If you want to save the model at every epoch in a subfolder set to 'True'
SAVE_EPOCHS = False
# If you just want to save the final output in current folder, set to 'True'
SAVE_LAST = False
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4

devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    print('[INFO] GPU is detected.')
    print('[INFO] GPU is detected.', file = fout, flush = True)
else:
    print('[INFO] GPU not detected.')
    print('[INFO] GPU not detected.', file = fout, flush = True)
print('[INFO] Done importing packages.')
print('[INFO] Done importing packages.', file = fout, flush = True)

class Net():
    
