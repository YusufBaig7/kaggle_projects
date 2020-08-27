import numpy as np
import pandas as pd
np.random.seed(1)

import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from scipy import signal
import cv2
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import keras.backend as K

import tensorflow as tf

train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
sample = pd.read_csv('../input/digit-recognizer/test.csv')

y_train = train.label

x_train = train.drop(['label'], axis =1).values

x_test = test.values

def plot_figure(im,interp=False):
    f=plt.figure(figsize=(3,6))
    plt.gray()
    plt.imshow(im,interpolation=None if interp else 'none')

plot_figure(x_test.reshape(-1, 28, 28)[0])

x_train = x_train/255.0
x_test = x_test/255.0

y_train = np_utils.to_categorical(y_train)
y_train

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = 2),
    
    tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
    tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax'),    
])

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 50, batch_size = 128, verbose = True)

model.summary()

y_test = model.predict(x_test)

def find_result(y_test):
    ResultList = []
    for w in y_test:
        ResultList.append(find_max(w))
    return np.array(ResultList)
def find_max(arr):
    maxval = arr[0]
    index = 0
    for i in range(0,10):
        if arr[i]>maxval:
            maxval = arr[i]
            index = i
    return index

final_result = find_result(y_test)

final_result[0:4]

for i in range(0,4):
    plot_figure(x_test.reshape(-1, 28, 28)[i])
    
df = pd.DataFrame(final_result)
df.to_csv('Result.csv')
