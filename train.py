import matplotlib.pyplot as plt
import os, numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorboard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten #, AveragePooling2D, MaxPool2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from sklearn.preprocessing import StandardScaler

# Define the Keras TensorBoard callback.

logdir="logs/dogNoDog/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
BATCH_SIZE=75

imageSize = (12,12)
noDog = os.path.join('.', 'nodog')
dog = os.path.join('.', 'dog')

labeledSet = 'dataset'

dds = tf.keras.preprocessing.image_dataset_from_directory(  labeledSet, 
                                                            labels='inferred', 
                                                            label_mode='categorical', 
                                                            class_names=['dog','nodog'],
                                                            color_mode='grayscale',  
                                                            image_size=imageSize, 
                                                            batch_size=BATCH_SIZE)

X = []
y = []
for data,labels in dds:
    X.extend(data)
    y.extend(labels)

model = Sequential()    
model.add(Flatten(input_shape=[*imageSize]))
model.add(Dense(units=144, activation='selu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
# model.build(input_shape=(None, *imageSize, 3))
# print(model.summary())


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#X, y = np.concatenate([[x,y] for x,y in dds])

# Fit the model to the training data

model.fit(
    x=tf.convert_to_tensor(X), #tf.convert_to_tensor(list(X)),
    y=tf.convert_to_tensor(y), #tf.convert_to_tensor(list(y)),
    epochs=250,
    shuffle=True,
    verbose=2,
    callbacks=[tensorboard_callback]

)


