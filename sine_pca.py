from __future__ import (absolute_import,
                        print_function, unicode_literals)

import numpy as np
from mpl_toolkits.mplot3d import Axes3D # needed for 3D plotting

seed = 1337
np.random.seed(seed)
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.decomposition import PCA, KernelPCA

from keras import callbacks
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import *

from utils import getSinusoid

# USE = "KERAS"
USE = "TENSORFLOW"

if USE == "KERAS":
    from Keras_2_0.clockwork import CWRNN
    from utils import get_activations_simple as get_activations
else:
    from TF2_3.clockwork import CWRNN
    from utils import get_activations_simple_tf as get_activations


COLORS=["#e3e6d3",
        "#e7a9a7",
        "#98d4e4",
        "#dac0a3",
        "#b3c0e1",
        "#c5dfb6",
        "#dcbed8",
        "#a2caba"]

COLORS_SSIG=["#3b7a9a",
            "#2679a7",
            "#0f303a",
            "#456a74",
            "#13526c"]

# Options: "PCA", "TSNE", None
DIM_RED = "PCA"

PLOT_PCA_FIT = True
PLOT_DENSITY = False
PLOT_3D = True

PLOT_EXRA = False

NORM = True
# set next 2 to False if you want the plots to display less information
PLOT_SUBSIGNALS = False
PLOT_NOISELESS = True
PLOT_ERROR = False

# Prevents permanent update and rescale of the DIM_RED plot's scale
STATIC_DIM_AXIS = False

#possible bug for normalization + plot:
#if PLOT_SUBSIGNALS: NORM=False

# original T: 256
timesteps = 1024
data_dim = 1
epochs = 100


prediction = True


SIN, x_train, y_train, subsignals = getSinusoid(timesteps, prediction=prediction, id=1, noise_std=0.)

# if PLOT_NOISELESS:
#     S,_,_, _ = getSinusoid(timesteps, prediction=prediction, id=1, noise_std=00)
#     subsignals = [S]
#     PLOT_SUBSIGNALS = True

###################################################
model = Sequential()

size = 64
modules = 8

mod_size = size // modules
print('mod-size = ', mod_size)

# geometric series as periods:
periods = np.power(2,range(0,modules,1), dtype='i').tolist() # 1,2,4,8,16,..128
# periods = np.power(2,range(0,modules,2)) # 1,4,16,64,,..
print(periods)

# model.add(SimpleRNN(size, activation='tanh', return_sequences=True, name='hid'))

# model.add(LSTM(size, activation='tanh', return_sequences=True, name='hid'))

model.add(CWRNN(size, input_shape=(timesteps-int(prediction), data_dim),
                periods=periods, unidirectional=True,
                return_sequences=True, activation='tanh', name='cwrnn'))

model.add(TimeDistributed(Dense(data_dim, activation='linear')))

es = callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, min_delta=0.001)

model.compile(loss='mse', optimizer=Adam(lr=0.01))

print('Training')
plt.ion() # this command takes a while to start (python 3)

f, axes = plt.subplots(2, 1)#, gridspec_kw={'width_ratios': [3, 1]})  # sequence

f.canvas.set_window_title('Output / States (View in Fullscreen)')


if DIM_RED:
    f3 = plt.figure(3)
    f3.canvas.set_window_title('Module Cluster (' + str(DIM_RED) + ')')

    if PLOT_DENSITY:
        f3 = plt.figure(4)
        f3.canvas.set_window_title('Density + PCA')

    pc_minmax_x = [-0.5, 0.5]
    pc_minmax_y = [-0.5, 0.5]

closable_figures = []
plt.figure(1)
for epoch in range(epochs):
    print('Epoch', epoch, '/', epochs)
    # this needs to be first line:
    ax = axes[0]

    ax.cla()

    ax.plot(y_train[0], 'k', linestyle='-', linewidth=2, label='target sequence')
    ax.set_ylabel('Output')

    if PLOT_SUBSIGNALS:
        [ax.plot(f_sub, alpha=0.7, color=COLORS_SSIG[ssidx], label='subsignal '+str(ssidx)) for ssidx,f_sub in enumerate(subsignals)]

    ax.grid(True)
    ax.set_xlim(0, timesteps)
    if NORM:
        ax.set_ylim(np.min(SIN)-0.1, np.max(SIN)+0.1)

    history = model.fit(x_train,
                        y_train,
                        verbose=1,
                        epochs=1,
                        callbacks=[es],
                        shuffle=False)
    # prediction:
    predicted_output = model.predict(x_train, batch_size=1)
    ax.plot(predicted_output[0], 'r--', linewidth=2, label=model.layers[0].name)
    if PLOT_ERROR:
        ax.plot((predicted_output[0]-y_train[0])*(predicted_output[0]-y_train[0]), 'r--', alpha=0.3, label='error')

    ax = axes[1]
    act = np.array(get_activations(model, x_train, layerid=0))
    act = np.transpose(act[0, 0, :, :])

    plt.legend()
    axes[0].legend(bbox_to_anchor=(1.1, 1), loc=1, borderaxespad=0., prop={'size': 12})

    # init
    if epoch == 0:
        act_filter = 10 # plot: higher numbers increasingly ignore small activations around 0

        ###### diverging color palettes useful for activations ######

        # this is useful with around sep= :
        # sns_map_act = mpl.colors.ListedColormap(sns.diverging_palette(10, 220, sep=act_filter, n=256))

        # sns_map_act = mpl.colors.ListedColormap(sns.diverging_palette(255, 133, l=60, n=256, center="light", sep=act_filter))

        # some of the best and popular:
        sns_map_act = mpl.colors.ListedColormap(sns.color_palette("RdBu_r", 256))

        # like rdbu_r but a little less contrast:
        # sns_map_act = mpl.colors.ListedColormap(sns.color_palette("coolwarm", 256))

        p = ax.imshow(act, extent=[0, timesteps, 0, size], interpolation='nearest', cmap=sns_map_act, aspect='auto')

        ax.set_aspect('auto')

        # make horizontal lines to separate modules:
        ax.yaxis.set_ticks(np.arange(0, size+1, int(mod_size)))
        # ax.invert_yaxis()
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Hidden Units')

        # put module labels on right side:
        ax_cl = ax.twinx()
        ax_cl.yaxis.set_ticks([0, size])
        modticks = np.arange(0, size + mod_size, mod_size)[::-1]
        ax_cl.set_ylabel('Modules')

        ax_cl.yaxis.set(ticks=modticks, ticklabels=range(modules))
        [ax_cl.axhline(y=y, color='k', linestyle='-') for y in modticks]
        ax_cl.grid(False)

        cbar_ax = f.add_axes([0.94, 0.1, 0.01, 0.75])

        cbar = f.colorbar(p, cax=cbar_ax)
        cbar.set_label('Activation')

        f.tight_layout()
        # f.subplots_adjust(right=0.95)
    else:
        p.set_array(act)


    p.set_clim(np.min(act), np.max(act))
    # p.set_clim(-1, 1) #TODO for recording images
    cbar.update_bruteforce(p)
    f.canvas.draw()

    #######################
    # 2nd Figure visualizes how the modules self-organize to fire at separate times
    # Observe clusters of module-periods to see that they operate on different scales
    if DIM_RED and epoch % 2==0:

        if DIM_RED == 'PCA':
            pca = PCA(n_components=3 if PLOT_3D else 2)#, whiten=True, svd_solver='full')
            # pca = KernelPCA(kernel="sigmoid", fit_inverse_transform=True, gamma=10)

            # transpose back to original output for PCA on temporal axis:
            Y = pca.fit(np.transpose(act)).transform(np.transpose(act))
            Y2 = pca.fit(act).transform(act)
        # t-SNE takes a while to calculate
        elif DIM_RED == 'TSNE':
            tsne = manifold.TSNE(n_components=2, init='pca',perplexity=40.0,
                 early_exaggeration=4.0, learning_rate=20.0, n_iter=3000,
                 n_iter_without_progress=300,random_state=1337)
            # tsne = manifold.SpectralEmbedding(n_components=2, n_neighbors=10)
            # tsne = manifold.Isomap(4, 2)
            # tsne = manifold.MDS(2, max_iter=100, n_init=1)
            # act = get_activations(model, len(model.layers)-1, x_train)
            # act = np.array(get_activations(model, 0, x_train))[0]
            # act = act[0, :, :]

            # transpose back to original output for PCA on temporal axis:
            # act = np.transpose(act[0, :, :])

            Y = tsne.fit_transform(np.transpose(act).astype(np.float64))
            Y2 = tsne.fit_transform(act.astype(np.float64))
        else:
            continue

        if PLOT_EXRA:
            palette_sns = sns.cubehelix_palette(timesteps - int(prediction), reverse=True, as_cmap=True)

            # TODO Find a way to redraw the seaborn figure
            if 'f_needed' in locals():
                plt.close(f_needed.fig)
            f_needed = sns.jointplot(x=Y[:,0], y=Y[:,1], kind='hex', color='b', label='cubehelix_palette')
            closable_figures.append(f_needed.fig)


        # palette_sns = itertools.cycle(sns.cubehelix_palette(modules, start=.5, rot=-.75))
        palette_sns = itertools.cycle(sns.cubehelix_palette(modules, reverse=True))

        plt.figure(3)
        plt.clf()
        for c, i in zip(palette_sns, range(modules)):

            # c = next(palette_sns)
            if PLOT_3D:
                ax = plt.gca(projection='3d')
                ndims = 3
            else:
                ax = plt
                ndims = 2

            if PLOT_PCA_FIT:
                ax.plot(*np.swapaxes(Y[i * mod_size:(i + 1) * mod_size, :ndims], 0, 1),
                         c=c,
                         linewidth=2, zorder=1
                         #label='line'+str(i)
                         )

            ax.scatter(*np.swapaxes(Y[i * mod_size:(i + 1) * mod_size, :ndims], 0, 1),
                        c=c,
                        s = 100,
                        cmap=plt.cm.Spectral,
                        label=i, zorder=2
                        )

        palette_sns = itertools.cycle(sns.cubehelix_palette(modules, start=.5, rot=-.75))

        # same figure but with density plot:

        if PLOT_DENSITY:
            plt.figure(4)
            plt.clf()
            for c, i in zip(palette_sns, range(modules)):

                if PLOT_PCA_FIT:
                    plt.plot(Y[i * mod_size:(i + 1) * mod_size, 0],
                             Y[i * mod_size:(i + 1) * mod_size, 1],
                             c=c,
                             linewidth=2,
                             label = i
                             )

                # TODO concatenate modules from PCA and make one single kdeplot (below scatter can stay in loop)
                sns.kdeplot(Y[i * mod_size:(i + 1) * mod_size, 0],
                            Y[i * mod_size:(i + 1) * mod_size, 1],
                            c=c,
                            s=100,
                            cmap='viridis',
                            label = i
                            )

                plt.scatter(Y[i * mod_size:(i + 1) * mod_size, 0],
                            Y[i * mod_size:(i + 1) * mod_size, 1],
                            c=c,
                            s = 100,
                            cmap=plt.cm.Spectral,
                            label=i
                            )


    if STATIC_DIM_AXIS:
        # plot limit offset: 20% of used value range
        offset = max(np.abs(pc_minmax_x[0] - pc_minmax_x[1]),
                    np.abs(pc_minmax_y[0] - pc_minmax_y[1])) / 20
        # offset = 0.5

        # keep track of limits to avoid permanent rescaling:
        if np.min(Y[:, 0]) < pc_minmax_x[0]: pc_minmax_x[0] = np.min(Y[:, 0])-offset
        if np.max(Y[:, 0]) > pc_minmax_x[1]: pc_minmax_x[1] = np.max(Y[:, 0])+offset
        if np.min(Y[:, 1]) < pc_minmax_y[0]: pc_minmax_y[0] = np.min(Y[:, 1])-offset
        if np.max(Y[:, 1]) > pc_minmax_y[1]: pc_minmax_y[1] = np.max(Y[:, 1])+offset

        # if you want rescaling every N epochs:
        if epoch % 50:
            pc_minmax_x[0], pc_minmax_x[1] = np.min(Y[:, 0])-offset, np.max(Y[:, 0])+offset
            pc_minmax_y[0], pc_minmax_y[1] = np.min(Y[:, 1])-offset, np.max(Y[:, 1])+offset

        plt.gcf().gca().set_xlim(pc_minmax_x[0], pc_minmax_x[1])
        plt.gcf().gca().set_ylim(pc_minmax_y[0], pc_minmax_y[1])

    plt.figure(1)

    #######################
    plt.pause(0.01)


# plt.savefig('seq_state.png')
