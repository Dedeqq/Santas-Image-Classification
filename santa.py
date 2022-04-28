#%% Liblaries
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np
import pandas as pd
from random import randint

#%% Load the data
labels = ['santa', 'not-a-santa']
img_size = 100
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
            resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
            data.append([resized_arr, class_num])
    return np.array(data, dtype=object)
#Now we can easily fetch our train and validation data.
train = get_data('is that santa\\train')
test = get_data('is that santa\\test')
#print(train[100][0]) 

#%% Preparing the data for neural network
x_train = []
y_train = []
x_test = []
y_test = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in test:
  x_test.append(feature)
  y_test.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

#%% Visualize some images
row=3; col=4;    
plt.figure()
for i in range(row*col):
    plt.subplot(row,col,i+1)
    plt.imshow(train[randint(0,299)][0])
    plt.axis('off')
plt.suptitle("Santa")
plt.show()

   
plt.figure()
for i in range(row*col):
    plt.subplot(row,col,i+1)
    plt.imshow(train[randint(300,599)][0])
    plt.axis('off')
plt.suptitle("Not Santa")
plt.show()
    
#%% Model
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(img_size,img_size,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()


opt = Adam(lr=0.0001)
number_of_epochs=20
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])


history = model.fit(x_train,y_train,epochs = number_of_epochs , validation_data = (x_test, y_test))

#%% Accuracy and loss plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(number_of_epochs)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()


#%% 
predict_x=model.predict(x_test) 
predictions=np.argmax(predict_x,axis=1)

predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_test, predictions, target_names = ['Santa (Class 0)','Not a santa (Class 1)']))
print(confusion_matrix(y_test,predictions))

#%% Show some of the missclasified images
misclassified_as_santa=np.where((y_test-predictions) == 1) #indexes on which missclasisfication happens
misclassified_as_not_santa=np.where((y_test-predictions) == -1) 

row=4; col=4;    
plt.figure()
for i in range(row*col):
    plt.subplot(row,col,i+1)
    plt.imshow(test[misclassified_as_not_santa[0][i]][0])
    plt.axis('off')
plt.suptitle("Model should have classified this as Santa")
plt.show()
   
plt.figure()
for i in range(row*col):
    plt.subplot(row,col,i+1)
    plt.imshow(test[misclassified_as_santa[0][i]][0])
    plt.axis('off')
plt.suptitle("Model should have classified this as Not Santa")
plt.show()

#%%
"""
#%% My tests
my_photo=cv2.imread("palpatine.jpg")[...,::-1]
my_photo = cv2.resize(my_photo, (img_size, img_size))
my_prediction=model.predict(np.array([my_photo])/255)
names=["santa", "not a santa"]
print(f"My photo prediction is: {names[np.argmax(my_prediction)]}")
"""
