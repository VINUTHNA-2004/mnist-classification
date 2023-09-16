# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

<img width="386" alt="268429125-0ebb09e5-c9e0-4f53-8e4c-721e079240bd" src="https://github.com/VINUTHNA-2004/mnist-classification/assets/95067307/34104236-0ef0-447b-9ef7-09c5a2c5d1e5">


## DESIGN STEPS


Step 1: 
Start by importing all the necessary libraries. And load the Data into Test sets and Training sets.

Step 2: 
Then we move to normalization and encoding of the data.

Step 3:
The Model is then built using a Conv2D layer, MaxPool2D layer, Flatten layer, and 2 Dense layers of 16 and 10 neurons respectively.

Step 4: 
The necessary Validating parameters are visualized for inspection.

Step 5: 
Finally, we pass handwritten digits to the model for prediction.

## PROGRAM
```
Name:D.R.Vinuthna
Reg.no:212221230017
```
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[1500]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(16,activation='tanh'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=64, validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/9.png')

type(img)

img = image.load_img('l1.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
```
## OUTPUT


### Training Loss, Validation Loss Vs Iteration Plot

![3 1](https://github.com/VINUTHNA-2004/mnist-classification/assets/95067307/c59f20be-65a8-43d7-a2d3-f72f32be9e45)
![3 2](https://github.com/VINUTHNA-2004/mnist-classification/assets/95067307/aa35834f-95df-45d0-8915-4ebe65134360)


### Classification Report
![3 3](https://github.com/VINUTHNA-2004/mnist-classification/assets/95067307/8a7d750b-075a-43c6-a60e-959839d7968b)


### Confusion Matrix
![3 4](https://github.com/VINUTHNA-2004/mnist-classification/assets/95067307/d9cdb0ec-0e0e-4985-92d2-633337a91f0d)


### New Sample Data Prediction
![3 5](https://github.com/VINUTHNA-2004/mnist-classification/assets/95067307/cf851f99-58e3-4706-9b93-d60921e865bd)
![3 6](https://github.com/VINUTHNA-2004/mnist-classification/assets/95067307/8cef75b2-1fc2-4643-814e-7ae2aba902f5)


## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.


