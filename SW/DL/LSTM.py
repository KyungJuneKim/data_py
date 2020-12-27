from tensorflow.keras import losses, optimizers, Sequential, layers

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from random import shuffle

from SW.DL.DL import PressureData
from SW.signal import dic


class LSTMData(PressureData):
    def __init__(self, group: dict, bounds: Tuple[int, int]):
        super(LSTMData, self).__init__(group, bounds)
        self.group = group

    def raw_to_LSTMData(self, raw_: dict):
        res = []
        x_ = []
        y_ = []
        x_data = []
        y_data = []

        for key in raw_.keys():
            for lst in raw_[key]:
                lst = [ele / 2000 for ele in lst]
                if len(lst) < self.bounds[1] - self.bounds[0]:
                    print(key)
                    continue
                x_.append(lst)
                y_.append(self.factors.index(key))

                res.append([x_, y_])

        shuffle(res)

        for i in range(len(res)):
            x_data.append(res[0])
            y_data.append(res[1])

        x_data = np.array(x_data).reshape((-1, self.bounds[1]-self.bounds[0], 1))
        y_data = np.eye(len(self.factors))[y_data]

        x_train = x_data[:int(len(x_data) * 3 / 4)]
        y_train = y_data[:int(len(y_data) * 3 / 4)]
        x_test = x_data[int(len(x_data) * 3 / 4):]
        y_test = y_data[int(len(y_data) * 3 / 4):]

        return [x_train, y_train, x_test, y_test]


class CategoricalLSTM(Sequential):
    def __init__(
            self,
            epoch: int, batch_size: int, learning_rate: float, output_size: int,
            lstm_size: int
    ):
        super(CategoricalLSTM, self).__init__()
        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.output_size: int = output_size
        self.lstm_size: list = lstm_size

        self.add(layers.LSTM(lstm_size))
        self.add(layers.Dense(output_size))
        self.add(layers.Activation('softmax'))

        self.compile(
            loss=losses.categorical_crossentropy,
            optimizer=optimizers.Adam(learning_rate),
            metrics=[
                'accuracy'
            ]
        )


if __name__ == '__main__':
    data_group = {
        'fabric30': [
            dic('1031_4'),
            dic('1031_5'),
            dic('1031_6')
        ],
        'glass': [
            dic('1107_1'),
            dic('1107_2'),
            dic('1107_3')
        ],
        'copper(1)': [
            dic('1031_4'),
            dic('1031_5'),
            dic('1031_6')
        ],
        'zinc(1)': [
            dic('1107_1')
        ]
    }

    data = LSTMData(
        group=data_group,
        bounds=(-200, 1200)
    )

    raw_data = data._load_raw_data()

    [train_x, train_y, test_x, test_y] = data.raw_to_LSTMData(raw_data)

    model = CategoricalLSTM(
        epoch=10, batch_size=1, learning_rate=0.001, lstm_size=100, output_size=len(data_group.keys())
    )

    print(model.summary())

    # model.fit(train_x, train_y, batch_size=model.batch_size, epochs=model.epoch, validation_split=0.3)

    # model.evaluate(test_x, test_y, batch_size=model.batch_size)