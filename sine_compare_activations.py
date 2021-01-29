from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
seed = 1337
np.random.seed(seed)

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

from keras import callbacks
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import *
from keras.utils.layer_utils import print_summary
from utils import getSinusoid

# USE = "KERAS"
USE = "TENSORFLOW"

if USE == "KERAS":
    from Keras_2_0.clockwork import CWRNN
    from utils import get_activations_simple as get_activations
else:
    from TF2_3.clockwork import CWRNN
    from utils import get_activations_simple_tf as get_activations

timesteps = 256
data_dim = 1
epochs = 51
prediction = True

SIN, x_train, y_train, subsignals = getSinusoid(timesteps, prediction=prediction, id=1, noise_std=0.05)

# geometric series as periods:
modules = 8
periods = np.power(2,range(0,modules,1), dtype='i').tolist() # 1,2,4,8,16,..128
# periods = np.power(2,range(0,modules,2)) # 1,4,16,64,,..
print(periods)


models = []
for layer, params in ((LSTM,dict()), (CWRNN, {'periods':periods, 'activation':'linear'})):

    print('Creating Model '+ str(layer))
    model = Sequential()

    model.add(layer(64, return_sequences=True, input_shape=(timesteps-int(prediction), data_dim), **params))

    model.add(TimeDistributed(Dense(data_dim)))

    es = callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min')

    model.compile(loss='mse', optimizer=Adam(lr=0.02))

    models.append(model)
    print_summary(model)

print('Training')
plt.ion() # this command takes a while to start (in python 3)

f, axes = plt.subplots(2, 1, sharey=True, sharex=True) # sequence
f2, axes2 = plt.subplots(2, 1) # hid states

plotiter = 2 # refreshes plot every N iterations
for i in range(epochs):
    if (i % plotiter == 0 or i == epochs - 1):
        [ax.cla() for ax in axes]
        for a, ax in enumerate(axes[0:2]):
            ax.plot(y_train[0], 'k', linestyle='-')
            ax.grid(True)
            ax.set_xlim(0, timesteps)
            ax.set_ylim(-1.1, 1.1)

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

        predicted_output = model.predict(x_train, batch_size=1)
        ax.plot(predicted_output[0], '#1f77b4', label=model.layers[0].name)
        # ax.plot(predicted_output[0], '#1f77b4', label=model.layers[0].name)

        ax = axes2[m]
        ax.set_title(model.layers[0].name)

        act = np.array(get_activations(model, x_train, layerid=0))
        act = np.transpose(act[0,0,:,:])

        sns_map_act = mpl.colors.ListedColormap(sns.color_palette("RdBu_r", 256))

        p = ax.imshow(act, cmap=sns_map_act, interpolation='nearest')
        #cbar_ax = f2.add_axes([0.94, 0.1, 0.02, 0.4])
        cbar_ax = f2.add_axes([0.94, 0.1, 0.01, 0.75])
        cbar = f2.colorbar(p, cax=cbar_ax)
        cbar.set_label('Activation')

        p.set_clim(np.min(act), np.max(act))
        cbar.update_bruteforce(p)

        f.canvas.draw()
        f2.canvas.draw()

        plt.pause(0.05)

    f.suptitle('Epoch: '+str(i), fontsize=10)
    f2.suptitle('Epoch: ' + str(i), fontsize=10)

    # f.savefig("./img/f" +str(i)+ ".png", dpi=150)
    # f2.savefig("./img/g" +str(i)+ ".png", dpi=150)

plt.hold()
