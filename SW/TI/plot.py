import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from SW import signal
from SW.signal import BOUNDS, dic, get_signal


def single_signal(sig: List[float]) -> Tuple:
    length = len(sig)
    fs = 962
    t = np.arange(length) / fs
    f = np.arange(int(length / 2 + 1)) * fs / length

    fft_signal = np.abs((np.fft.fft(sig) / fs)[:int(length / 2 + 1)])

    return t, sig, f, fft_signal


def plot(group: List[List[dict]], category: List[str], bounds: Tuple[int, int] = BOUNDS):
    for j, data_list in enumerate(group):
        if not data_list:
            print('Error')
            continue

        fig_name = str(j + 1)
        for data in data_list:
            fig_name += ' ' + data['name']

        fig, axes = plt.subplots(nrows=len(category), ncols=2, num=fig_name, figsize=(15, 8))
        for data in data_list:
            name = data['name']
            for idx in range(len(category)):
                signal_list = get_signal(name, category[idx], data['path'], bounds)
                for i in data['num']:
                    fft = single_signal(signal_list[i])
                    label = name + ': ' + str(i + 1)
                    axis = axes if len(category) == 1 else axes[idx]

                    axis[0].plot(fft[0], fft[1], label=label)
                    axis[0].set_title(category[idx])
                    axis[0].set_xlabel('time (s)')
                    axis[0].legend()
                    axis[1].scatter(fft[2], fft[3], label=label)
                    axis[1].set_title(category[idx] + ' fft')
                    axis[1].set_xlabel('frequency (Hz)')
                    axis[1].legend()
            fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_group = [
        [
            dic('zinc(1)_1107_1', list(range(50, 53))),
            dic('zinc(1)_1107_1', list(range(170, 173))),
            dic('zinc(1)_1107_1', list(range(280, 282))),
            # dic('copper(1)_1031_5', list(range(10, 15))),
            # dic('zinc_1031_1', list(range(80, 82)))
        ],
        # [dic('test_1031_3', list(range(10, 19)))],
        # [dic(signal.name, list(range(10, 15))+list(range(90, 95)))]
    ]
    sensor_list = [
        'pressure',
        'heat'
    ]

    plot(data_group, sensor_list, (-200, 1300))
