import tensorflow as tf
import matplotlib as plt
import numpy as np
import pandas as pd

x_train = pd.read_csv("../input/digit-recognizer/train.csv")
x_test1 = pd.read_csv("../input/digit-recognizer/test.csv")
train_features = x_train.iloc[:, 1:785]
train_labels = x_train.iloc[:, 0]
x_test = x_test1.iloc[:, 0:784]
print(x_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28,1)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_features,train_labels, epochs = 6)

model.predict(x_test)