import numpy as np
from itertools import product
from os import getcwd
from random import randrange
from tensorflow.keras import activations, losses, metrics, optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Any

from SW.DL.DataSet import DataSet
from SW.DL.util import plot_model


class RegressionPWM(DataSet):
    def __init__(
            self,
            period: int = 20, cycle: int = 5
    ):
        self.period: int = period
        self.cycle: int = cycle

        duration = np.arange(0., 1., 0.05).tolist()
        phase = range(period)
        factors = list(product(duration, phase))
        super().__init__(factors, period * cycle, 1)

    def _load_raw_data(self) -> Any:
        return None

    def single_x(self, factor):
        x = []
        scaling = 1000
        for i in range(self.cycle):
            for j in range(self.period):
                if (j + factor[1]) % self.period < factor[0] * self.period:
                    x.append((800 + randrange(200)) / scaling)
                else:
                    x.append(randrange(200) / scaling)

        return x

    def single_y(self, factor):
        return factor[0]

    def reshape_x(self, x) -> np.ndarray:
        return np.array(x, dtype=np.float32).reshape((-1, self.input_size, 1))

    def reshape_y(self, y) -> np.ndarray:
        return np.array(y, dtype=np.float32)


class RegressionLSTM(Sequential):
    def __init__(
            self,
            epoch: int, batch_size: int,
            learning_rate: float, lstm_size: int
    ):
        super(RegressionLSTM, self).__init__()
        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.lstm_size: int = lstm_size

        self.add(LSTM(lstm_size))
        self.add(Dense(1, activation=activations.sigmoid))
        self.compile(
            loss=losses.MeanSquaredError(),
            optimizer=optimizers.Adam(learning_rate),
            metrics=[
                metrics.mean_absolute_error
            ]
        )


if __name__ == '__main__':
    data = RegressionPWM(
        period=20,
        cycle=5
    ).generate(
        num=50,
        ratio=[0.7, 0.2]
    )

    data.plot_x((0.75, 5))

    model = RegressionLSTM(
        epoch=5,
        batch_size=1,
        learning_rate=0.01,
        lstm_size=5
    )

    history = model.fit(
        data.x_train, data.y_train,
        batch_size=model.batch_size,
        epochs=model.epoch,
        validation_data=(data.x_val, data.y_val)
    )

    plot_model(history)

    model.save(getcwd() + '/RegressionPWM')

    loss, mae = model.evaluate(data.x_test, data.y_test)
    print(loss, mae)

    pred = model.predict(
        data.reshape_x(data.single_x((0.75, 5)))
    )
    print(pred)
