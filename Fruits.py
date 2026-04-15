import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix

from PIL import Image
print("PIL installed correctly")

DATASET_PATH = "fruits-360_100x100"
IMG_SIZE = (100, 100)
BATCH_SIZE = 32

#Load dataset
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH + "/fruits-360/Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    DATASET_PATH + "/fruits-360/Test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

#Number of classes
num_classes = len(train_data.class_indices)
print("Number of classes:", num_classes)

#CNN Model
model = Sequential()

#Layer 1
model.add(Conv2D(16, (2,2), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 2
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Subsequent Layers
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout
model.add(Dropout(0.3))

# Flatten
model.add(Flatten())

#Fully connected layer
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.4))

#Output layer
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    epochs=30,
    validation_data=test_data
)

test_loss, test_acc = model.evaluate(test_data)
print("\nTest Accuracy:", test_acc)

#Confusion Matrix
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes
cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:\n", cm)