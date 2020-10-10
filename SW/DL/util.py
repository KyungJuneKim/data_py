import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import History
from typing import List
from warnings import warn


def split_data_set(x: List, y: List, ratio: List[float]):
    if not ratio:
        raise AttributeError('Nothing in the `ratio`')
    if len(x) != len(y):
        raise AssertionError('lists, `x` and `y`, are not the same length')
    if np.sum(ratio) > 1:
        raise AssertionError('The sum of the elements of the `ratio` must not exceed 1')
    if any(r * len(x) < 1 for r in ratio):
        warn('Too little ratio')

    error = int(np.around(np.log10(len(x)))) - 15
    indices = np.around(np.multiply(np.cumsum(ratio), len(x)), -error).astype(np.int32)
    split = [(x[:indices[0]], y[:indices[0]])]
    for i, j in zip(indices[:-1], indices[1:]):
        split.append((x[i:j], y[i:j]))
    split.append((x[indices[-1]:], y[indices[-1]:]))

    return split


def plot_model(history: History):
    if not history.history:
        raise RuntimeError('You must fit model first. Use `model.fit(x, y)`')
    keys = []
    for key1 in history.history:
        if key1[:4] == 'val_':
            continue
        for key2 in history.history:
            if key2 == 'val_' + key1:
                keys.append((key1, key2))
                break
        else:
            keys.append(tuple(key1, ))

    fig, axes = plt.subplots(nrows=1, ncols=len(keys), sharex='all', figsize=(15, 6))

    for axis, key in zip(axes, keys):
        axis.set_title('Model ' + key[0])
        axis.plot(history.history[key[0]], label=key[0])
        if len(key) is 2:
            axis.plot(history.history[key[1]], label=key[1])
        axis.set_xlabel('Epoch')
        axis.set_ylabel(key[0].capitalize())
        axis.legend()

    fig.tight_layout()
    plt.show()
