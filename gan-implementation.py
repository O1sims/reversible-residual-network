#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:29:16 2019
"""

import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import enable_eager_execution

from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, ELU, Dense, \
    Flatten, Input, Reshape, Activation, Conv2DTranspose


# Why this isn't enabled by default is beyond me...
enable_eager_execution()

# Set visible devices used to '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load up the MNIST data and partition into test and train acorss both axes
(x_training_data, y_training_data), (x_test_data, y_test_data) = mnist.load_data()
x_training_data, x_test_data = [x/255.0 for x in [x_training_data, x_test_data]]
y_training_data, y_test_data = [tf.keras.utils.to_categorical(x) for x in [y_training_data, y_test_data]]

# Call optimiser
optimizer = Adam()
z_dimensions = 100

# Create discriminative network
network_d = input_ = Input((28,28))
network_d = Reshape((28,28,1))(network_d)

network_d = Conv2D(64, (5,5), padding='same', strides=(2,2))(network_d)
network_d = BatchNormalization()(network_d)
network_d = ELU()(network_d)

network_d = Conv2D(128, (5,5), padding='same', strides=(2,2))(network_d)
network_d = BatchNormalization()(network_d)
network_d = ELU()(network_d)

network_d = Flatten()(network_d)
network_d = Dense(128)(network_d)
network_d = BatchNormalization()(network_d)
network_d = ELU()(network_d)
network_d = Dense(1, activation='sigmoid')(network_d)

# Instantiate and provide summary of discriminative network
dm = Model(input_, network_d)
dm.compile(optimizer, 'binary_crossentropy')
dm.summary()

# Create generative network
x = in1 = Input((z_dimensions,))

x = Dense(7*7*64)(x)
x = BatchNormalization()(x)
x = ELU()(x)
x = Reshape((7,7,64))(x)

x = Conv2DTranspose(128, (5,5), strides=(2,2), padding='same')(x)
x = BatchNormalization()(x)
x = ELU()(x)

x = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same')(x)
x = Activation('sigmoid')(x)
x = Reshape((28,28))(x)

gm = Model(in1, x)
gm.compile('adam', 'mse')
gm.summary()

# GAN
dm.trainable = False
x = dm(gm.output)
tm = Model(gm.input, x)
tm.compile(optimizer, 'binary_crossentropy')

dlosses, glosses = [], []



BS = 256

# GAN training loop
for i in range(5000):
  # train discriminator
  dm.trainable = True
  real_i = x_training_data[np.random.choice(x_training_data.shape[0], BS)]
  fake_i = gm.predict_on_batch(np.random.normal(0,1,size=(BS, z_dimensions)))
  dloss_r = dm.train_on_batch(real_i, np.ones(BS))
  dloss_f = dm.train_on_batch(fake_i, np.zeros(BS))
  dloss = (dloss_r + dloss_f)/2

  # train generator
  dm.trainable = False
  gloss_0 = tm.train_on_batch(np.random.normal(0,1,size=(BS, z_dimensions)), np.ones(BS))
  gloss_1 = tm.train_on_batch(np.random.normal(0,1,size=(BS, z_dimensions)), np.ones(BS))
  gloss = (gloss_0 + gloss_1)/2

  if i%50 == 0:
    print("%4d: dloss:%8.4f   gloss:%8.4f" % (i, dloss, gloss))
  dlosses.append(dloss)
  glosses.append(gloss)
    
  if i%250 == -1%250:
    plt.figure(figsize=(16,16))
    plt.imshow(np.concatenate(gm.predict(np.random.normal(size=(10, z_dimensions))), axis=1))
    plt.show()

# Plot the discriminator model loss and the generator model loss over time
plt.plot(dlosses[100:], label="Discriminator Loss")
plt.plot(glosses[100:], label="Generator Loss")
plt.legend()

# Plot the full set of generated MNIST data
x = []
for i in range(10):
  x.append(np.concatenate(gm.predict(np.random.normal(size=(10, z_dimensions))), axis=1))
plt.imshow(np.concatenate(x, axis=0))


