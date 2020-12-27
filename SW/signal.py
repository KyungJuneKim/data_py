import csv
from os import getcwd
from typing import Final, List, Tuple


sample_list = ['none', 'aluminium', 'copper', 'fabric14', 'fabric30', 'glass', 'polymer', 'zinc']
NAME: Final[str] = '1107_copper(1)_1'

PATH: Final[str] = getcwd() + '/data/'
BOUNDS: Final[Tuple[int, int]] = (-200, 1300)
print(PATH)


def dic(sample_name: str, numbers: List[int] = None, path: str = PATH) -> dict:
    return {'name': sample_name, 'num': numbers, 'path': path}


def get_signal(sample: str, category: str, path: str = PATH, bounds: Tuple[int, int] = BOUNDS) \
        -> List[List[float]]:
    signal = []
    new_signal = []
    with open(path + sample + '_' + category + '.csv', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            signal.append([float(r) for r in row])

    for s1, s2 in zip(signal[:-1], signal[1:]):
        new_signal.append(s1[bounds[0]:] + s2[:bounds[1]])

    return new_signal
