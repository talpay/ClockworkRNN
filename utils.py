from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import keras.backend as K
import numpy as np

def getSinusoid(timesteps, prediction, harmonics=1, id=None, norm=True, noise_std=0, random=False, seed=None):
    TR = 2 * np.pi
    T = np.linspace(0, TR, timesteps)

    if random:
        if seed is not None:
            np.random.seed(seed)

        subsignals_p = []
        for n in range(harmonics):
            subsignals_p.append((np.random.uniform(0,1),
                                 np.random.uniform(0,2*TR*timesteps*0.49),
                                 np.random.uniform(0,TR/2)) )
    else: # predefined
        if id == 0:
            subsignals_p = [(2, 1.5, 0), (3, 1.8, 0), (4, 1.1, 0)]
        elif id == 1:
            subsignals_p = [(0.0572, 4.667, 0), (0.05, 4.6, TR / 2), (0.0218, 7.22, 0), (0.1, 1, 0)]
        else:
            subsignals_p = [(0.0572, 4.667, 0), (0.05, 4.6, TR / 2), (0.0218, 7.22, 0), (0.1, 1, 0)]

    subsignals = [amp * np.sin(TR * freq * T + phase) for amp, freq, phase in subsignals_p]

    SIN = np.array(sum(subsignals))
    SIN += np.random.normal(0, noise_std, timesteps)

    if norm:

        def normalize(low, up, arr):
            ma = max(arr)
            mi = min(arr)
            return (up - (low)) / (ma - mi) * (arr - ma) + up

        SIN = normalize(-1, 1, SIN)
        subsignals = [normalize(-1,1,ssig) for ssig in subsignals]

    if prediction:
        x = np.array(SIN[:-1][None, :, None]).astype('f')
        y = np.asarray(SIN[1:][None, :, None]).astype('f')
    else:
        x = np.zeros((1, timesteps, 1)).astype('f')
        y = np.array(SIN[None, :, None]).astype('f')

    return SIN, x, y, subsignals

def getSine(id, timesteps, prediction=True):
    ''' predict a sine curve per output neuron '''
    large = 1
    small = 0.25

    if id==0:
        a,b = large,large
    elif id ==1:
        a,b = large, small
    elif id==2:
        a,b = small, large
    elif id==3:
        a,b = small,small
    else:
        raise Exception

    TR = 2*np.pi
    T1 = np.linspace(0, TR/2-.01, timesteps/2)
    T2 = np.linspace(TR/2+.01, TR, timesteps/2)

    data = np.zeros((timesteps,2))
    sig = 1
    for i in range(2):
        a_sin = a * np.sin(sig*TR * (timesteps / (TR / 2)) / (timesteps) * T1)
        b_sin = b * np.sin(sig*TR * (timesteps / (TR / 2)) / (timesteps) * T2)

        data[:,i] = np.concatenate((a_sin.T, b_sin.T), axis=-1)

        sig *= -1

    if prediction:
        x = np.array(data[:-1][None, :]).astype('f')
        y = np.asarray(data[1:][None, :]).astype('f')
    else:
        x = np.zeros((1, timesteps, 2)).astype('f')
        y = np.array(data[None, :]).astype('f')

    # data shapes: (1, T, 2)
    return data, x, y

def get_activations_simple(model, X_batch, layerid=0, layer_name=None):
    # Keras SO:
    if isinstance(layer_name, str):
        layer = model.get_layer(layer_name)
    else:
        layer = model.layers[layerid]

    get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output])
    activations = get_activations([X_batch, 0])

    return activations

def get_activations_simple_tf(model, X_batch, layerid=0, layer_name=None):
    # TF2:
    from tensorflow.python.keras.backend import eager_learning_phase_scope

    get_activations = K.function([model.input], [model.layers[layerid].output])

    with eager_learning_phase_scope(value=0):
        activations = get_activations([X_batch])

    return activations

# test:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    timesteps = 1024
    sin, x,y, subsignals = getSinusoid(timesteps, 10, random=True)

    f = plt.figure(1)
    plt.plot(sin)
    [plt.plot(f_sub, alpha=0.7) for f_sub in subsignals]
    plt.xlim(0,len(sin))

    print('y', y)
    print('x', x.shape)
    print('y', y.shape)

    plt.show()
