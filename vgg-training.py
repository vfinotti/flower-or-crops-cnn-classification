
# coding: utf-8

# Using the tutorial from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# Since some images showed problems with the EXIF data, I followed the suggetion to delete this information: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/31558

# import warnings; warnings.simplefilter('ignore')


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

np.random.seed(42)

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'weights/bottleneck_fc_model_vgg16.h5'
train_data_dir = 'images/train'
validation_data_dir = 'images/test'
nb_train_samples = 16000
nb_validation_samples = 2000
epochs = 15
batch_size = 20 # the num of training and test samples must be divisible by this

def does_file_exist(fname):
    import os.path
    return os.path.isfile(fname) 

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save('weights/bottleneck_features_train_vgg16.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('weights/bottleneck_features_validation_vgg16.npy', bottleneck_features_validation)

def train_top_model():
    train_data = np.load('weights/bottleneck_features_train_vgg16.npy')
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load('weights/bottleneck_features_validation_vgg16.npy')
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

if (not does_file_exist('weights/bottleneck_features_train_vgg16.npy') or not does_file_exist('weights/bottleneck_features_train_vgg16.npy')):
    save_bottlebeck_features()
else:
    print('Bottleneck npy files exist, not saving it!')
    
if (not does_file_exist(top_model_weights_path)):
    train_top_model()
else:
    print('Full Connected h5 file exists, not training it!')

# save_bottlebeck_features()
# train_top_model()


# In[5]:


'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense

import numpy as np

np.random.seed(42)

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'weights/bottleneck_fc_model_vgg16.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'images/train'
validation_data_dir = 'images/test'
nb_train_samples = 16000
nb_validation_samples = 2000
epochs = 5
batch_size = 72*2 # number of cores of Phi02 *2, according to Intel's suggestion 

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False,
                                input_shape = (img_width, img_height, 3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# top_model.add(Dense(4096, activation='relu'))
# top_model.add(Dense(512, activation='relu'))
# top_model.add(Dense(128, activation='relu'))
top_model.add(Dense(64, activation='relu'))
top_model.add(Dropout(0.2))
# top_model.add(Dense(4096, activation='relu'))
# top_model.add(Dense(512, activation='relu'))
# top_model.add(Dense(128, activation='relu'))
top_model.add(Dense(64, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
#model.add(top_model)
model = Model(input= base_model.input, output= top_model(base_model.output))

#input_tensor = Input(shape=(150,150,3))
#base_model = VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)
#top_model = Sequential()
#top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
#top_model.add(Dense(256, activation='relu'))
#top_model.add(Dropout(0.5))
#top_model.add(Dense(1, activation='sigmoid'))
#top_model.load_weights('bootlneck_fc_model.h5')
#model = Model(input= base_model.input, output= top_model(base_model.output))

# set the first 13 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
# for layer in model.layers[:12]:
#     layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Training just the FC part
# model = model
# model.fit_generator(
#     train_generator,
#     samples_per_epoch=nb_train_samples,
#     epochs=epochs,
#     validation_data=validation_generator,
#     nb_val_samples=nb_validation_samples)

# model.save_weights('./weights/vgg_top.h5')

# set the first 13 layers (up to the last conv block)
# to trainable (weights will be updated)
for layer in model.layers[:12]:
    layer.trainable = True
    
# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model.save_weights('./weights/vgg_finetuned.h5')

