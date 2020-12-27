import numpy as np
from itertools import product
from os import getcwd
from random import randrange, shuffle
from tensorflow.keras import activations, losses, metrics, optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Any, Tuple

import SW.signal
from SW.DL.DataSet import DataSet
from SW.DL.util import plot_model
from SW.signal import dic, get_signal


class PressureData(DataSet):
    def __init__(self, group: dict, bounds: Tuple[int, int]):
        self.group = group
        self.bounds = bounds
        super(PressureData, self).__init__(list(group.keys()), bounds[1] - bounds[0], len(group.keys()))
        self.idx_x = 0

    def _load_raw_data(self) -> Any:
        raw = {}
        for key in self.factors:
            sigs = []
            for val in self.group[key]:
                sigs += get_signal(key + '_' + val['name'], 'pressure', val['path'], self.bounds)
            shuffle(sigs)
            raw[key] = sigs

        return raw

    def single_x(self, idx: int, factor):
        return self._raw[factor][idx]

    def single_y(self, factor):
        return np.eye(len(self.factors))[self.factors.index(factor)]

    def reshape_x(self, x) -> np.ndarray:
        return np.array(x, dtype=np.float32).reshape((-1, self.input_size, 1))

    def reshape_y(self, y) -> np.ndarray:
        return np.array(y, dtype=np.float32)


if __name__ == '__main__':
    data_group = {
        'copper': [
            dic('0919_2'),
            dic('0919_3'),
            dic('0919_4')
        ]
        #'glass': [
        #
        #]
    }

    data = PressureData(
        group=data_group,
        bounds=(-1200, 2600)
    )

    raw_data = data._load_raw_data()
    print(len(raw_data['copper'][99])) #297개 list, each list 3800
