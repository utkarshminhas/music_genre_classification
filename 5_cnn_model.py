# multi label classification -  predict multiple genres for a song
# multi clas classification -  predict one genre for a song

from pprint import pprint
import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from sklearn.metrics import classification_report

import shutil
from keras.preprocessing.image import ImageDataGenerator
import random

print("tensorflow version",tf.__version__)

if tf.test.is_gpu_available(cuda_only=True):
    print("TensorFlow can access CUDA")
else:
    print("TensorFlow cannot access CUDA")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


data_dir = "/home/utkarsh/Desktop/music_genre_classification/data/sample_download/mel_spectograms_top5"
data_dir = "/home/utkarsh/Desktop/music_genre_classification/data/sample_download/mel_spectograms_top5"

total_songs = 0
num_classes = len(os.listdir(data_dir))

for genre in os.listdir(data_dir):
    songs_per_genre = len(os.listdir(os.path.join(data_dir,genre)))
    print(str(songs_per_genre).zfill(3),":",genre)
    total_songs += songs_per_genre

print("Total songs",total_songs)

datagen = ImageDataGenerator(
    rescale=1./255,
     validation_split=0.2,
    )


image_size = (288,432)
image_size = (640,190)
batch_size = 64
num_epochs = 30

train_generator = datagen.flow_from_directory(
    data_dir,
    subset = 'training',
    target_size = image_size,
    color_mode = "rgba",
    class_mode = 'categorical',
    batch_size = batch_size
    )

val_generator = datagen.flow_from_directory(
    data_dir,
    subset = 'validation',
    target_size = image_size,
    color_mode = 'rgba',
    class_mode = 'categorical',
    batch_size = batch_size
    )


def GenreModel(input_shape, classes):
  np.random.seed(9)
  X_input = Input(input_shape)

  X = Conv2D(8,kernel_size=(3,3),strides=(1,1),kernel_initializer = glorot_uniform(seed=9))(X_input)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(16,kernel_size=(3,3),strides = (1,1),kernel_initializer=glorot_uniform(seed=9))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(32,kernel_size=(3,3),strides = (1,1),kernel_initializer = glorot_uniform(seed=9))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  X = Conv2D(64,kernel_size=(3,3),strides=(1,1),kernel_initializer=glorot_uniform(seed=9))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  
  X = Flatten()(X)

  X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=9))(X)

  model = Model(inputs=X_input,outputs=X,name='GenreModel')

  return model



model = GenreModel(input_shape=(image_size[0],image_size[1],4),classes=num_classes)
pprint(model.summary())
opt = Adam(learning_rate=0.00005)
model.compile(optimizer = opt,loss='categorical_crossentropy',metrics=['accuracy'])  



model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = val_generator, 
    validation_steps = val_generator.samples // batch_size,
    epochs = num_epochs
    )

# y_pred = model.predict(x_test, batch_size=64, verbose=1)
# y_pred_bool = np.argmax(y_pred, axis=1)

# print(classification_report(y_test, y_pred_bool))