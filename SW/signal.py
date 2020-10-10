import csv
from os import getcwd
from typing import List, Tuple


sample_list = ['none', 'aluminium', 'copper', 'fabric14', 'fabric30', 'glass', 'polymer', 'zinc']
name = 'copper_0926_1'

DEFAULT_PATH = getcwd() + '/data/'
print(DEFAULT_PATH)


def dic(sample_name: str, number: List[int] = None, path: str = DEFAULT_PATH) -> dict:
    return {'name': sample_name, 'num': number, 'path': path}


def get_signal(sample: str, category: str, path: str = DEFAULT_PATH, bounds: Tuple[int, int] = (-1200, 2600)) \
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
