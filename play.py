#script to train the model --- we are using the squeezenet pretrained model. 
import cv2
import os
import numpy as np
from keras.applications import MobileNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential


IMAGE_SAVE_PATH = 'image_path'

class_map = {
    'rock':0,
    'paper':1,
    'scissor':2,
    'none':3
}

def mapper(val):
    return class_map[val]

num_classes = len(class_map)

def baseline_model():
    model = Sequential([
        MobileNet(input_shape=(227, 227, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(num_classes, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model


dataset = []

for directory in os.listdir(IMAGE_SAVE_PATH):
    path = os.path.join(IMAGE_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        if item.startswith('.'):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

'''
structure of dataset -- [
    [img, label].....
]
'''  

data, labels = zip(*dataset)
labels = list(map(mapper, labels))

labels = np_utils.to_categorical(labels)

model = baseline_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# start training
model.fit(np.array(data), np.array(labels), epochs=10)

# save the model for later use
model.save("rock-paper-scissors-model.h5")


    
