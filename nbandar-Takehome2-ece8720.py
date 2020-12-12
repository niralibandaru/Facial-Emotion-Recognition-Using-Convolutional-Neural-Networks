# FACIAL EMOTION RECOGNITION USING CNN

### Nirali Bandaru - Clemson Spring 2020 - ECE 8720 - Dr. Robert J. Schalkoff

# Importing libraries

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras import backend as K
from __future__ import division
from keras.layers import Dense
from keras.models import model_from_json
import os

# loading dataset from CSV file - A sample of 35,888 pictures with an assigned value corresponding to an emotion
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

df = pd.read_csv('fer2013.csv',engine='python');

print ("Data loaded")

# Model Parameters 
output_nodes = 7
batch_size = 100
epochs = 50

#Splitting into training and testing data sets in 80%-20% ratio

y = df.emotion
images = df['pixels'].tolist()

x = []
for image in images:
    arr = [int(var) for var in image.split(' ')]
    arr = np.asarray(arr).reshape(48, 48)
    x.append(arr.astype('float32'))

x = np.asarray(x)
x = np.expand_dims(x, -1)

y = pd.get_dummies(df['emotion']).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

print("x-train shape: ",x_train.shape)
print("y-train shape: ",y_train.shape)
print("x-test shape: ",x_test.shape)
print("y-test shape: ",y_test.shape)

#save numpy files

np.save('xtest_data', x_test)
np.save('ytest_data', y_test)

print("Numpy files of testing data have been saved.")
print("Image Dimension: ",str(len(x[0])),'x',str(len(x[0])))
print("Sample size:", len(x))


# Building the Convolutional Neural Network Model
# Using the same parameters as the reference paper:
model = Sequential()
# First, 2D Convolution Layer:
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1), data_format='channels_last'))
model.add(BatchNormalization())

# Followed by Max-Pooling Layer:
model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))

# Dropout:
model.add(Dropout(0.25))

# Second set of 2D Convolution Layers: (Conv --> ReLU)*2 Layers
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())

# Followed by Max-Pooling Layer:
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Dropout:
model.add(Dropout(0.25))

# The resultant matrices are put into a single column matrix - the flattening layer:
model.add(Flatten())

# Fully-connected Layer:
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output layer
model.add(Dense(output_nodes, activation='softmax'))

model.summary()

from tensorflow.keras.losses import categorical_crossentropy


#TRAINING
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy','mse'])

model.fit(np.array(x_train), np.array(y_train),
          batch_size=100,
          epochs=50,
          verbose=1,
          shuffle=True)

savefile = model.to_json()
with open("savefile", "w") as json_file:
    json_file.write(savefile)
model.save_weights("ferweights")

#PLOTTING THE OUTPUTS FROM TRAINING AND VALIDATION TESTS

from matplotlib import pyplot as plt

# list all data in history
print(history.history.keys())

#plotting accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plotting mse
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model mean-squared-error')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#TESTING THE MODEL

# Testing the accuracy of the test set

testdata = open('fer.json','r')
readdata = testdata.read()
testdata.close()
loaded_model = model_from_json(readdata)
loaded_model.load_weights("fer.h5")
print("Testing data loaded.")

# evaluate loaded model on testing data
loaded_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy','mse'])
score = loaded_model.evaluate(x, y, verbose=0)
print("Accuracy on test set :"+str(acc)+"%")

