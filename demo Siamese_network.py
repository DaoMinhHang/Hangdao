# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:37:39 2021

@author: Dell
"""
# The tutorial that I followed to build the model
# https://github.com/sudharsan13296/Hands-On-Meta-Learning-With-Python/blob/master/02.%20Face%20and%20Audio%20Recognition%20using%20Siamese%20Networks/2.5%20Audio%20Recognition%20using%20Siamese%20Network.ipynb
import weakref

import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import librosa
from sklearn.model_selection import train_test_split
import glob
import IPython
from random import randint
#data processing
import librosa
import numpy as np

#modelling
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

raw_data = pickle.load(open("./raw_data.pkl", 'rb'))


audios = raw_data[0]
labels = raw_data[1]
# A huge number of records will be created (up to 40 millions records), so I only took a smaller sample for demo purpose.

list_audios=list(audios[:80])
list_labels=list(labels[:80])

# Create all possible pair combinations
possible_pairs=[]
for idx,a in enumerate(list_audios):
    for b in list_audios[idx+1:]:
        possible_pairs.append([a,b])
possible_labels=[(a,b)for idx,a in enumerate(list_labels) for b in list_labels[idx+1:]]

# Create new labels for them: 0 and 1. 0: negative pair. 1: positive pair
siamese_labels=[]
for pair_label in possible_labels:
    if pair_label[0]==pair_label[1]:
        siamese_labels.append(1)
    else:
        siamese_labels.append(0)
        
train_labels=np.array(siamese_labels)   

train_pairs=np.array(possible_pairs)

X_train, X_test, y_train, y_test = train_test_split(train_pairs, train_labels, test_size=0.2)

y_train=y_train.astype('float32')
y_test=y_test.astype('float32')

def build_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

input_dim = X_train.shape[2]

audio_a = Input(shape=input_dim)
audio_b = Input(shape=input_dim)
audio_a.get_shape() #TensorShape([None, 11025])
audio_b.get_shape() #TensorShape([None, 11025])

base_network = build_base_network(input_dim)

feat_vecs_a = base_network(audio_a)
feat_vecs_b = base_network(audio_b)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
epochs = 13
rms = RMSprop()
model = Model([audio_a, audio_b], distance)
# Lastly, we define our loss function as contrastive_loss and compile the model.

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
model.compile(loss=contrastive_loss, optimizer=rms)

audio_1 = X_train[:, 0, :]
audio_2 = X_train[:, 1, :]

model.fit([audio_1, audio_2], y_train, validation_split=.25,
          batch_size=128, verbose=2, epochs=epochs)
