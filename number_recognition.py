

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

"""**Load and preprocess the MNIST dataset**"""

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the input images from 28x28 pixels to 784-dimensional vectors
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32')

# Normalize the pixel values between 0 and 1
X_train /= 255
X_test /= 255

# Convert the target labels to categorical one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

"""**Build the neural network model**"""

model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""**Train the model**"""

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)

"""**Evaluate the model**"""

accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)

"""**Plot the training history**"""

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

"""**Make predictions**"""

# Predict on a single image
digit = X_test[0]
digit = np.expand_dims(digit, axis=0)
prediction = model.predict(digit)
predicted_class = np.argmax(prediction)
print("Predicted digit:", predicted_class)