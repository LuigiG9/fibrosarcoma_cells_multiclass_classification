# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:45:41 2022

@author: matti
"""

import numpy as np 
import pandas as pd 
from subprocess import check_output
import math
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import RMSprop, Adam
import seaborn as sns
import keras.callbacks
from PIL import Image
import os
from keras.utils.np_utils import to_categorical 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import os
import warnings

# filter warnings
warnings.filterwarnings('ignore')

#%% LOAD DATA IN FOLD 
def load_data_kfold(k):
    
    path_images = r'C:\Users\matti\OneDrive\Desktop\Contest finale\DM\TrainImage'
    path_label = r'C:\Users\matti\OneDrive\Desktop\Contest finale\DM\training.xlsx'
    batch_size = 8
    datagen = keras.preprocessing.image.ImageDataGenerator()

        
    data_generator = datagen.flow_from_directory(
        path_images,
        target_size=(128,128),
        batch_size=1374,
        class_mode='categorical',
        color_mode='grayscale'
        ) 
    x, y = data_generator.next()
    # x_train, x_test, y_train, y_test = train_test_split(x, y,
    #                                                 test_size=0.1,
    #                                                 train_size=0.9,
    #                                                 random_state=42,
    #                                                 stratify=y
    #                                                 )

   
    
    
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x, y.argmax(1)))
    
    
    return  folds,x, y#,x_test,y_test
k = 10
folds,X_train, y_train= load_data_kfold(k)

#%% CREATE THE MODEL 
def get_model():
    model = keras.Sequential()
    ##################################
    model.add(keras.layers.Conv2D(16,(3,3), activation = 'relu', padding = 'same', input_shape = (128,128,1)))
    model.add(keras.layers.Conv2D(16,(3,3), activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format=None))
    ##################################
    model.add(keras.layers.Conv2D(32,(3,3), activation = 'relu', padding = 'same'))
    model.add(keras.layers.Conv2D(32,(3,3), activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format=None))
    ##################################
    model.add(keras.layers.Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
    model.add(keras.layers.Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format=None))
    ##################################
    model.add(keras.layers.Conv2D(128,(2,2), activation = 'relu', padding = 'same'))
    model.add(keras.layers.Conv2D(128,(2,2), activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format=None))
    ##################################
    model.add(keras.layers.Conv2D(256,(2,2), activation = 'relu', padding = 'same'))
    model.add(keras.layers.Conv2D(256,(2,2), activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format=None))
    ##################################
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation = 'relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128,activation = 'relu'))
    model.add(keras.layers.Dense(5,activation='softmax')) 
    ##################################
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.summary()
    model.compile(optimizer=optimizer,loss="categorical_crossentropy", metrics=["accuracy"])  
    return model

#%% CALLBACKS AND IMAGEGENERATOR
learning_rate_reduction1=ReduceLROnPlateau(monitor='val_loss',patience=2,factor=0.8,min_lr=0.0000001)
model = get_model()
model.summary()

batch_size=32

train_datagen = ImageDataGenerator(rotation_range = 90,
                                   zoom_range = [0.7, 1.3],
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   samplewise_std_normalization=True
                                   )
valid_datagen = ImageDataGenerator(samplewise_std_normalization=True)
result=[]
s=1

#%% CROSS VALIDATION K=10
for j, (train_idx, test_idx) in enumerate(folds):
    
    print('\nFold ',j)
    X_train_cv = X_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_test_cv = X_train[test_idx]
    y_test_cv= y_train[test_idx]
    
    folds2 = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X_train_cv, y_train_cv.argmax(1)))
   
    for h, (train_idx2, val_idx) in enumerate(folds2):
         X_train_cv2 = X_train[train_idx2]
         y_train_cv2 = y_train[train_idx2]
         X_valid_cv = X_train[val_idx]
         y_valid_cv = y_train[val_idx]
   
         train_generator=train_datagen.flow(X_train_cv2, y_train_cv2,
                                         batch_size = 32,#batch_size, 
                                         shuffle=True
                                         )
         valid_generator=valid_datagen.flow(X_valid_cv,y_valid_cv,batch_size=32,shuffle=True)
           
         model = get_model()
         history=model.fit_generator(
                    train_generator,
                    steps_per_epoch=len(X_train_cv2)/batch_size,
                    epochs=1,
                    shuffle=True,
                    verbose=1,
                    validation_data = valid_generator,
                    callbacks = [learning_rate_reduction1]
                    )
         history_dict = history.history
         loss_values = history_dict['loss']
         val_loss_values = history_dict['val_loss']
         accuracy = history_dict['accuracy']
         val_accuracy = history_dict['val_accuracy']
         
         epochs = range(1, len(loss_values) + 1)
         fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        #
        # Plot the model accuracy vs Epochs
        #
         ax[0].plot(epochs, accuracy, 'r', label='Training accuracy')
         ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
         ax[0].set_title('Training & Validation Accuracy', fontsize=16)
         ax[0].set_xlabel('Epochs', fontsize=16)
         ax[0].set_ylabel('Accuracy', fontsize=16)
         ax[0].legend()
        #
        # Plot the loss vs Epochs
        #
         ax[1].plot(epochs, loss_values, 'r', label='Training loss')
         ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
         ax[1].set_title('Training & Validation Loss', fontsize=16)
         ax[1].set_xlabel('Epochs', fontsize=16)
         ax[1].set_ylabel('Loss', fontsize=16)
         ax[1].legend()
         my_path = os.path.abspath(r'C:\Users\matti\OneDrive\Desktop\DataSet\CrossVal') # Figures out the absolute path for you in case your working directory moves around.
         my_file = 'graph.png'
         ...
         fig.savefig(os.path.join(my_path, str(s) +'.png'))   
        
        #Valutazione del modello sul test set locale e salvataggio risultati
        
         test_generator=valid_datagen.flow(X_test_cv,y_test_cv,batch_size=32)
         result.append(model.evaluate(test_generator))
         s=s+1
    #print(model.evaluate(test_generator))



