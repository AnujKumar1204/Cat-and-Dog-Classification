import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , Flatten , Dense , MaxPooling2D , Dropout , BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping




train_dest = "C:\\Users\\Anuj Kumar\\OneDrive\\Desktop\\Cat and Dog Classification\\Dataset\\train"
val_dest =   "C:\\Users\\Anuj Kumar\\OneDrive\\Desktop\\Cat and Dog Classification\\Dataset\\test"

train_generator = ImageDataGenerator(
    rescale = 1./255.0,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    shear_range = 0.2,
    fill_mode = 'nearest'
)

val_generator = ImageDataGenerator(
    rescale = 1./255.0)

train_data = train_generator.flow_from_directory(
    train_dest,
    target_size = (180 , 180),
    batch_size = 32,
    class_mode = 'binary',
    shuffle = True
)

val_data = val_generator.flow_from_directory(
    val_dest,
    target_size = (180 , 180),
    batch_size = 32,
    class_mode = 'binary'
)




early_stopping = EarlyStopping(patience=3, restore_best_weights = True)

model = Sequential([
    Conv2D(32, (3,3), input_shape=(180,180,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    
    Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])
    
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data , epochs = 10 , validation_data = val_data , callbacks = [early_stopping])

loss , accuracy = model.evaluate(val_data)

model.save("cat_dog_model.keras")

# 88% accuracy on validation data