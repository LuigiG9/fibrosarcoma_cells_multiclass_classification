import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import imageio as io
import keras.callbacks
from PIL import Image
import os

# import librerie per la modifica del dataset
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from sklearn.model_selection import train_test_split

# import librerie per la creazione del modello
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from imblearn.over_sampling import SMOTE, RandomOverSampler

import tensorflow as tf

# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

#%% carico immagini

path_images = r'C:\Users\luigi\Desktop\Università\MACHINE LEARNING\Training\Training'
path_label = r'C:\Users\luigi\Desktop\Università\MACHINE LEARNING\training.xlsx'
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

#%% genero test set locale (10% del dataset totale) e dataset di training e validation

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.1,
                                                    train_size=0.9,
                                                    random_state=42,
                                                    stratify=y
                                                    )
x_training, x_valid, y_training, y_valid = train_test_split(x_train, y_train,
                                                            test_size=0.20,
                                                            train_size=0.80,
                                                            random_state=42,
                                                            stratify=y_train
                                                            )

#%% data augumentation normalizzazione delle immagini

train_datagen = ImageDataGenerator(rotation_range = 90,
                                   zoom_range = [0.7, 1.3],
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   samplewise_std_normalization=True
                                   )
train_generator = train_datagen.flow(x_training, y_training,
                                     batch_size = 32,#batch_size, 
                                     shuffle=True
                                     )

valid_datagen = ImageDataGenerator(samplewise_std_normalization=True)
valid_generator = valid_datagen.flow(x_valid, y_valid,
                                     batch_size =32,#batch_size,
                                     shuffle=True
                                     )

#%%MODEL DEFINITION AND COMPILING

model = keras.Sequential()
##################################v
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
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128,activation = 'relu'))
model.add(keras.layers.Dense(5,activation='softmax'))


optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
model.summary()
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction1=ReduceLROnPlateau(monitor='val_loss',patience=2,factor=0.8,min_lr=0.0000001)

epochs = 150
batch_size = 32

#%% addestramento

history = model.fit(
    train_generator,
    batch_size=batch_size,
    epochs = epochs, 
    verbose = 1,
    validation_data = valid_generator,
    class_weight = None,
    shuffle = True,
    callbacks=[learning_rate_reduction1]
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


#%% normalizzazione del test set locale e calcolo della matrice di confusione

test_datagen= ImageDataGenerator(samplewise_std_normalization=True)
test_generator= test_datagen.flow(x_test, y_test,
                                  batch_size = 138
                                  ) 

x_test,y_test = test_generator.next()

predictions = model.predict(x_test)

y_pred = np.argmax(predictions, axis=1)
true_classes = y_test
rounded_labels=np.argmax(true_classes, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(rounded_labels, y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

class_labels = list(data_generator.class_indices.keys())  

report = classification_report(rounded_labels, y_pred, target_names = class_labels)
print(report)

#%% Salvataggio del modello

path = r'C:\Users\luigi\Desktop\model_piubuono'
model.save(path)

#%% Caricamento del dataset di test

path_test = r'C:\Users\luigi\Desktop\Università\MACHINE LEARNING\Test\Test'

files = os.walk(path_test, topdown = True)

for file in files:
  filenames = file[2]

listanomi = np.asarray(filenames)

prediction = np.zeros((344,1), np.int64)
i = 0
count = 0

for nome in filenames:
  img = keras.preprocessing.image.load_img(path_test+'/'+nome, 
                                           target_size = (128,128),
                                           color_mode='grayscale'
                                           )
  input_arr = keras.preprocessing.image.img_to_array(img)
  input_arr = np.array([input_arr])
  input_arr = (input_arr - np.mean(input_arr))/np.std(input_arr)
  pred = np.argmax(model.predict(input_arr))
  prediction[i] = pred
  if pred == 1:
    count=count+1
  i = i+1

#%% Creazione del file csv

d = pd.DataFrame({'ID':listanomi.flatten(), 'Class':prediction.flatten()})
print(d)

pd.DataFrame(d).to_csv( r'C:\Users\luigi\Desktop\model_piubuono\test_l.csv', index=False)
