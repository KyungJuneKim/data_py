import csv
import serial
import os
import time
from typing import List, Tuple

from SW.signal import DEFAULT_PATH


MIN = 3


def measure_all_by_cnt(port: str = 'COM5', baud_rate: int = 2000000, cnt: int = MIN - 1) \
        -> Tuple[List[List[float]], List[List[float]], float, str]:

    ti = serial.Serial(port=port, baudrate=baud_rate)
    err = ''

    if cnt < MIN - 1:
        cnt = MIN - 1
        err += 'Too Little Data. '

    start_sensing = False
    first = True
    start_time = time.time()

    p_sets = []
    t_sets = []
    pressure = []
    temperature = []

    while ti.readable():
        line = ti.readline().decode('utf-8').strip()

        if 'Start' in line:
            p_sets = []
            t_sets = []
            start_sensing = True
            start_time = time.time()
        elif start_sensing:
            if 'S' in line[:2]:
                print(str(len(p_sets) + 1) + ': ' + str(len(pressure)))
                if first and (not pressure or not temperature):
                    first = False
                else:
                    first = False
                    p_sets.append(pressure)
                    pressure = []
                    t_sets.append(temperature)
                    temperature = []
                if len(p_sets) is MIN:
                    print('Enough Data')
                if len(p_sets) > cnt:
                    print('End')
                    break
            else:
                key, msg = line.split(':')[:2]
                if key == 'P':
                    pressure.append(float(msg.strip()))
                elif key == 'T':
                    temperature.append(float(msg.strip()))
                else:
                    print('Error')
                    err += 'Message Error. '
                    break
        else:
            pass
    else:
        err += 'Serial Connection Error. '

    return p_sets, t_sets, time.time() - start_time, err


def measure_by_time(port: str = 'COM5', baud_rate: int = 2000000, running_time: float = 10.0, code: str = 'P') \
        -> Tuple[List[float], float, str]:

    ti = serial.Serial(port=port, baudrate=baud_rate)
    err = ''

    start_sensing = False
    start_time = time.time()

    values = []

    while ti.readable():
        line = ti.readline().decode('utf-8').strip()

        if 'Start' in line:
            values = []
            start_sensing = True
            start_time = time.time()
        elif start_sensing:
            if time.time() - start_time > running_time:
                print('End')
                break
            elif 'S' in line[:2]:
                pass
            else:
                key, msg = line.split(':')[:2]
                if key == code:
                    values.append(float(msg.strip()))
        else:
            pass

    return values, time.time() - start_time, err


def measure_by_cnt(port: str = 'COM5', baud_rate: int = 2000000, cnt: int = MIN - 1, code: str = 'P') \
        -> Tuple[List[List[float]], float, str]:

    ti = serial.Serial(port=port, baudrate=baud_rate)
    err = ''

    if cnt < MIN - 1:
        cnt = MIN - 1
        err += 'Too Little Data. '

    start_sensing = False
    first = True
    start_time = time.time()

    sets = []
    values = []

    while ti.readable():
        line = ti.readline().decode('utf-8').strip()

        if 'Start' in line:
            sets = []
            start_sensing = True
            start_time = time.time()
        elif start_sensing:
            if 'S' in line[:2]:
                print(str(len(sets) + 1) + ': ' + str(len(values)))
                if first and not values:
                    first = False
                else:
                    first = False
                    sets.append(values)
                    values = []
                if len(sets) is MIN:
                    print('Enough Data')
                if len(sets) > cnt:
                    print('End')
                    break
            else:
                key, msg = line.split(':')[:2]
                if key == code:
                    values.append(float(msg.strip()))
                else:
                    print('Error')
                    err += 'Message Error. '
                    break
        else:
            pass
    else:
        err += 'Serial Connection Error. '

    return sets, time.time() - start_time, err


def save(sets: List[List], sample: str, category: str, path: str = DEFAULT_PATH) -> bool:
    if not sets:
        print('No Data')
    elif len(sets) < MIN:
        print('Too Little Data')
    else:
        sets = sets[1:]
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(path + sample + '_' + category + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(sets)
        return True
    return False
