from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import matplotlib.pyplot as plt

# If you don't have standalone Keras 2.4.1 directing to tf.keras: change keras.X to tf.keras.X
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import callbacks
from keras.utils.layer_utils import print_summary

# USE = "KERAS"
USE = "TENSORFLOW"

if USE == "KERAS":
    from Keras_2_0.clockwork import CWRNN
else:
    from TF2_3.clockwork import CWRNN

from utils import getSinusoid

import numpy as np
seed = 1337
np.random.seed(seed)

timesteps = 1024
data_dim = 1
epochs = 50

prediction = True

SIN, x_train, y_train, subsignals = getSinusoid(timesteps, prediction=prediction, id=1, noise_std=0.05)

###################################################

modules = 8
periods = np.power(2, range(0, modules, 1)).astype('i')  # power 2 series: [1,2,4,8,16,32...]

print(periods)

models = []
for layer, params in ((LSTM, dict()), (CWRNN, {'periods':periods, 'activation':'linear'})):

    print('Creating Model '+ str(layer))
    model = Sequential()

    model.add(layer(64, return_sequences=True,
                    input_shape=(timesteps-int(prediction), data_dim), **params))


    model.add(TimeDistributed(Dense(data_dim)))

    es = callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min')

    model.compile(loss='mse',
                  optimizer=Adam(lr=0.02))

    models.append(model)
    print_summary(model)

model.add(TimeDistributed(Dense(data_dim)))

es = callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min')

model.compile(loss='mse',
              optimizer=Adam(lr=0.02))

print_summary(model)

f, axes = plt.subplots(2, 1, sharey=True, sharex=True)
print('\n\n\n\nTraining')

plotiter = 2
for i in range(epochs):
    if (i % 2 == 0 or i == epochs - 1):

        [ax.cla() for ax in axes]
        for a, ax in enumerate(axes):
            ax.plot(y_train[0], 'k', linestyle='-')
            ax.grid(True)
            plt.xlim(0, timesteps)
            plt.ylim(-1.1, 1.1)
    for m, model in enumerate(models):
        ax = axes[m]
        ax.set_title(model.layers[0].name)
        print('Epoch', i, '/', epochs)
        history = model.fit(x_train,
                    y_train,
                    verbose=1,
                    epochs=1,
                    callbacks=[es],
                    shuffle=False)

        # prediction:
        predicted_output = model.predict(x_train, batch_size=1)
        ax.plot(predicted_output[0], label=model.layers[0].name)
        plt.pause(0.1)
plt.hold()