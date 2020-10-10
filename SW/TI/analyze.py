import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import List, Tuple

from SW import signal
from SW.signal import DEFAULT_PATH, get_signal


def get_pressure_info(sample: str, path: str = DEFAULT_PATH) -> Tuple[List[int], List[float], List[float]]:
    signals = get_signal(sample, 'pressure', path)
    name = ['idx400', 'idx1400', 'idx2000', 'idx200']
    length = []
    slope = []
    amplitude = []

    for sig in signals:
        step = 0
        idx_max = 0
        idx = {}

        for i, value in enumerate(sig):
            cond = [value >= 400, value >= 1400, value >= 2000, value <= 200, False]
            if value > sig[idx_max]:
                idx_max = i
            if cond[step]:
                idx[name[step]] = (i, value)
                step += 1

        if step < 4:
            raise ValueError

        length.append(idx[name[3]][0] - idx[name[0]][0])
        slope.append((idx[name[2]][1] - idx[name[1]][1]) / (idx[name[2]][0] - idx[name[1]][0]))
        amplitude.append(sig[idx_max])

    return length, slope, amplitude


if __name__ == '__main__':
    len1, slope1, amp1 = get_pressure_info(signal.name)
    len2, slope2, amp2 = get_pressure_info('copper_0919_2')

    t_len, p_len = stats.ttest_ind(len1, len2, equal_var=False)
    t_slope, p_slope = stats.ttest_ind(slope1, slope2, equal_var=False)
    t_amp, p_amp = stats.ttest_ind(amp1, amp2, equal_var=False)

    print('len1 mean: ' + str(np.mean(len1)))
    print('len1 std: ' + str(np.std(len1)))
    print('slope1 mean: ' + str(np.mean(slope1)))
    print('slope1 std: ' + str(np.std(slope1)))
    print('amp1 mean: ' + str(np.mean(amp1)))
    print('amp1 std: ' + str(np.std(amp1)))

    print('len2 mean: ' + str(np.mean(len2)))
    print('len2 std: ' + str(np.std(len2)))
    print('slope2 mean: ' + str(np.mean(slope2)))
    print('slope2 std: ' + str(np.std(slope2)))
    print('amp2 mean: ' + str(np.mean(amp2)))
    print('amp2 std: ' + str(np.std(amp2)))

    print(t_len, p_len)
    print(t_slope, p_slope)
    print(t_amp, p_amp)

    plt.figure('len')
    plt.title('len')
    plt.plot(len1, label=1)
    plt.plot(len2, label=2)
    plt.legend()
    plt.figure('slope')
    plt.title('slope')
    plt.plot(slope1, label=1)
    plt.plot(slope2, label=2)
    plt.legend()
    plt.figure('amp')
    plt.title('amp')
    plt.plot(amp1, label=1)
    plt.plot(amp2, label=2)
    plt.legend()

    plt.figure('len_distribution')
    plt.title('len')
    plt.hist([len1, len2])
    plt.figure('slope_distribution')
    plt.title('slope')
    plt.hist([slope1, slope2])
    plt.figure('amp_distribution')
    plt.title('amp')
    plt.hist([amp1, amp2])

    plt.show()
