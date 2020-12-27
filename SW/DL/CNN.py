from tensorflow.keras import losses, optimizers, Sequential, layers

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from random import shuffle

from SW.DL.DL import PressureData
from SW.signal import dic


class CNNData(PressureData):
    def __init__(self, group: dict, bounds: Tuple[int, int], sampfreq: int):
        super(CNNData, self).__init__(group, bounds)
        self.group = group
        self.sampfreq = sampfreq

    def raw_to_spectrogram(self, raw_: dict):
        x_data = np.empty((0, 129, 28, 1))
        y_data = np.empty((0, len(self.group.keys())))
        res = []

        for key in raw_.keys():
            for lst in raw_[key]:
                if len(lst) < self.bounds[1]-self.bounds[0]:
                    print(key)
                    print(len(y_))
                    continue
                lst = np.array(lst)
                spec, freq, t, im = plt.specgram(x=lst, Fs=self.sampfreq)
                spec = spec*100
                #print(spec)
                x_ = spec.reshape(spec.shape+(1,))
                y_ = self.single_y(key)

                res.append([x_, y_])

        shuffle(res)
        for item in res:
            x_data = np.append(x_data, item[0].reshape((1, item[0].shape[0], item[0].shape[1], item[0].shape[2])), axis=0)
            y_data = np.append(y_data, item[1].reshape(1, item[1].shape[0]), axis=0)

        y_data = self.reshape_y(y_data)

        x_train = x_data[:int(len(x_data)*3/4)]
        y_train = y_data[:int(len(y_data)*3/4)]
        x_test = x_data[int(len(x_data)*3/4):]
        y_test = y_data[int(len(y_data)*3/4):]

        return [x_train, y_train, x_test, y_test]

    def raw_to_2D(self, raw_: dict):
        x_data = np.empty((0, 95, 40, 1))
        y_data = np.empty((0, len(self.group.keys())))
        res = []

        for key in raw_.keys():
            for lst in raw_[key]:
                lst = [ele/2000 for ele in lst]
                if len(lst) < self.bounds[1]-self.bounds[0]:
                    print(key)
                    continue
                x_ = np.array(lst).reshape((95, 40, 1))
                y_ = self.single_y(key)

                res.append([x_, y_])

        shuffle(res)
        for item in res:
            x_data = np.append(x_data, item[0].reshape((1, item[0].shape[0], item[0].shape[1], item[0].shape[2])),
                               axis=0)
            y_data = np.append(y_data, item[1].reshape(1, item[1].shape[0]), axis=0)

        y_data = self.reshape_y(y_data)

        x_train = x_data[:int(len(x_data) * 3 / 4)]
        y_train = y_data[:int(len(y_data) * 3 / 4)]
        x_test = x_data[int(len(x_data) * 3 / 4):]
        y_test = y_data[int(len(y_data) * 3 / 4):]

        return [x_train, y_train, x_test, y_test]

    def raw_to_1D(self, raw_: dict):
        x_data = np.empty((0, 1, self.bounds[1]-self.bounds[0], 1))
        y_data = np.empty((0, len(self.group.keys())))
        res = []

        for key in raw_.keys():
            for lst in raw_[key]:
                lst = [ele / 2000 for ele in lst]
                if len(lst) < self.bounds[1]-self.bounds[0]:
                    print(key)
                    continue
                x_ = np.array(lst).reshape((1, self.bounds[1]-self.bounds[0], 1))
                y_ = self.single_y(key)

                res.append([x_, y_])

        shuffle(res)
        for item in res:
            x_data = np.append(x_data, item[0].reshape((1, item[0].shape[0], item[0].shape[1], item[0].shape[2])),
                               axis=0)
            y_data = np.append(y_data, item[1].reshape(1, item[1].shape[0]), axis=0)

        y_data = self.reshape_y(y_data)

        x_train = x_data[:int(len(x_data) * 3 / 4)]
        y_train = y_data[:int(len(y_data) * 3 / 4)]
        x_test = x_data[int(len(x_data) * 3 / 4):]
        y_test = y_data[int(len(y_data) * 3 / 4):]

        return [x_train, y_train, x_test, y_test]


class CategoricalCNN(Sequential):
    def __init__(
            self,
            epoch: int, batch_size: int, learning_rate: float, output_size: int,
            layers_: list
    ):
        super(CategoricalCNN, self).__init__()
        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.output_size: int = output_size
        self.layers_: list = layers_

        self.layer_depth: int = len(layers_) + 2

        for layer in layers_:
            _type = layer.pop('layer')
            self.add(getattr(layers, _type)(**layer))  # call function named _type

        self.add(layers.Dense(output_size))
        self.add(layers.Activation('softmax'))

        self.compile(
            loss=losses.categorical_crossentropy,
            optimizer=optimizers.Adam(learning_rate),
            metrics=[
                'accuracy'
            ]
        )

    def plot(self, h):
        plt.subplot(2, 1, 1)
        plt.plot(h.history['accuracy'], label='Train')
        plt.plot(h.history['val_accuracy'], label='Validation')
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')

        plt.subplot(2, 1, 2)
        plt.plot(h.history['loss'], label='Train')
        plt.plot(h.history['val_loss'], label='Validation')
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    train_data = {
        'fabric30': [
            dic('1121_fabric30(4)_1'),
            dic('1121_fabric30(4)_2'),
            dic('1121_fabric30(4)_3')
        ],
        'glass': [
            dic('1121_glass(1)_1'),
            dic('1121_glass(1)_2'),
            dic('1121_glass(1)_3')
        ],
        'copper(1)': [
            dic('1121_copper(1)_1'),
            dic('1121_copper(1)_2'),
            dic('1121_copper(1)_3')
        ],
        'zinc(1)': [
            dic('1121_zinc(1)_1'),
            dic('1121_zinc(1)_2'),
            dic('1121_zinc(1)_3')
        ],
        'aluminium(1)': [
            dic('1121_aluminium(1)_1'),
            dic('1121_aluminium(1)_2'),
            dic('1121_aluminium(1)_3')
        ],
        'fabric14(4)': [
            dic('1121_fabric14(4)_1'),
            dic('1121_fabric14(4)_2'),
            dic('1121_fabric14(4)_3')
        ]
    }

    # predict_data = {
    #     'copper(1)': [
    #         dic('copper(1)_1113_1'),
    #         dic('copper(1)_1113_2')
    #     ],
    #     'glass(1)': [
    #         # dic('glass(1)_1113_2'),
    #         dic('glass(1)_1113_3')
    #     ]
    # }

    train = CNNData(
        group=train_data,
        bounds=(-200, 1200),
        sampfreq=1300
    )

    # predict = CNNData(
    #     group=predict_data,
    #     bounds=(-200, 1200),
    #     sampfreq=1300
    # )

    raw_data = train._load_raw_data()
    # predict = predict._load_raw_data()

    [train_x, train_y, test_x, test_y] = train.raw_to_1D(raw_data)
    input_shape = train_x.shape[1:]

    model = CategoricalCNN(
        epoch=10, batch_size=1, learning_rate=0.001, output_size=len(train_data.keys()),
        layers_=[
            # 'layer': function name in tensorflow.keras.layers

            {'layer': 'Conv2D', 'filters': 64, 'kernel_size': (1, 2), 'strides': 1, 'input_shape': input_shape,
             'activation': 'relu'},
            {'layer': 'MaxPooling2D', 'pool_size': (1, 2), 'strides': 2},
            # {'layer': 'Activation', 'activation': 'relu'},

            {'layer': 'Conv2D', 'filters': 128, 'kernel_size': (1, 2), 'padding': "same", 'activation': 'relu'},
            {'layer': 'MaxPooling2D', 'pool_size': (1, 2), 'strides': 2},

            {'layer': 'Conv2D', 'filters': 128, 'kernel_size': (1, 2), 'padding': "same", 'activation': 'relu'},
            {'layer': 'MaxPooling2D', 'pool_size': (1, 2), 'strides': 2},

            {'layer': 'Conv2D', 'filters': 64, 'kernel_size': (1, 2), 'padding': "same", 'activation': 'relu'},
            # {'layer': 'MaxPooling2D', 'pool_size': (1, 2), 'strides': 2},
            #
            # {'layer': 'Conv2D', 'filters': 64, 'kernel_size': (1, 2), 'padding': "same", 'activation': 'relu'},
            # # {'layer': 'MaxPooling2D', 'pool_size': (1, 2), 'strides': 2},
            # #
            # # {'layer': 'Conv2D', 'filters': 64, 'kernel_size': (1, 2), 'padding': "same", 'activation': 'relu'},
            # # {'layer': 'MaxPooling2D', 'pool_size': (1, 2), 'strides': 2},
            # #
            # # {'layer': 'Conv2D', 'filters': 32, 'kernel_size': (1, 2), 'padding': "same", 'activation': 'relu'},

            {'layer': 'Flatten'},
            # {'layer': 'Dropout', 'rate': 0.5},

            {'layer': 'Dense', 'units': 100},
            {'layer': 'Activation', 'activation': 'relu'},
            # {'layer': 'Dropout', 'rate': 0.5}
        ]
    )

    # print(model.summary())

    history = model.fit(train_x, train_y, batch_size=model.batch_size, epochs=model.epoch, validation_split=0.3)

    model.evaluate(test_x, test_y, batch_size=model.batch_size)

    # for key in predict.keys():
    #     print(key)
    #     for item in predict[key]:
    #         print(model.predict(x=np.array(item).reshape((1, 1, 1400, 1))))

    model.plot(history)
