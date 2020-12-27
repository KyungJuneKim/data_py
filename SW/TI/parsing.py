import csv
import serial
import os
import time
from typing import Final, List, Tuple

from SW.signal import PATH


PORT: Final[str] = 'COM11'
BAUD_RATE: Final[int] = 2000000
MIN: Final[int] = 3


def measure_all_by_cnt(port: str = PORT, baud_rate: int = BAUD_RATE, cnt: int = MIN - 1) \
        -> Tuple[List[List[float]], List[List[float]], List[str], float]:

    ti = serial.Serial(port=port, baudrate=baud_rate)
    log = []

    if cnt < MIN - 1:
        cnt = MIN - 1
        tmp = '[ERR] Too Little Data'
        log.append(tmp)
        print(tmp)

    start_sensing = False
    first = True
    start_time = time.time()

    p_sets = []
    t_sets = []
    pressure = []
    temperature = []
    log = []

    try:
        while ti.readable():
            line = ti.readline().decode('utf-8').strip()  # decode Exception

            if 'Start' in line:
                p_sets = []
                t_sets = []
                log = []
                start_sensing = True
                start_time = time.time()
            elif start_sensing:
                if 'S' in line[:2]:
                    tmp = '[NUM] ' + str(len(p_sets) + 1) + ': ' + str(len(pressure))
                    log.append(tmp)
                    print(tmp)
                    if first and (not pressure or not temperature):
                        first = False
                    else:
                        first = False
                        p_sets.append(pressure)
                        pressure = []
                        t_sets.append(temperature)
                        temperature = []
                    if len(p_sets) is MIN:
                        tmp = '[INFO] Enough Data'
                        log.append(tmp)
                        print(tmp)
                    if len(p_sets) > cnt:
                        tmp = '[INFO] End'
                        log.append(tmp)
                        print(tmp)
                        break
                else:
                    key, msg = line.split(':')[:2]  # index Exception
                    if key == 'P':
                        pressure.append(float(msg.strip()))  # transfer Exception
                    elif key == 'T':
                        temperature.append(float(msg.strip()))  # transfer Exception
                    else:
                        tmp = '[ERR] Message Error: ' + line
                        log.append(tmp)
                        print(tmp)
                        break
            else:
                pass
        else:
            tmp = '[ERR] Serial Connection Error'
            log.append(tmp)
            print(tmp)
    except Exception as e:
        tmp = '[EXCEPT] ' + str(e)
        log.append(tmp)
        print(tmp)

    return p_sets, t_sets, log, time.time() - start_time


def measure_by_time(port: str = PORT, baud_rate: int = BAUD_RATE, running_time: float = 10.0, code: str = 'P') \
        -> Tuple[List[float], List[str], float]:

    ti = serial.Serial(port=port, baudrate=baud_rate)
    log = []

    start_sensing = False
    start_time = time.time()

    values = []

    try:
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
                        tmp = '[ERR] Message Error: ' + line
                        log.append(tmp)
                        print(tmp)
                        break
            else:
                pass
        else:
            tmp = '[ERR] Serial Connection Error'
            log.append(tmp)
            print(tmp)
    except Exception as e:
        tmp = '[EXCEPT] ' + str(e)
        log.append(tmp)
        print(tmp)

    return values, log, time.time() - start_time


def measure_by_cnt(port: str = PORT, baud_rate: int = BAUD_RATE, cnt: int = MIN - 1, code: str = 'P') \
        -> Tuple[List[List[float]], List[str], float]:

    ti = serial.Serial(port=port, baudrate=baud_rate)
    log = []

    if cnt < MIN - 1:
        cnt = MIN - 1
        tmp = '[ERR] Too Little Data'
        log.append(tmp)
        print(tmp)

    start_sensing = False
    first = True
    start_time = time.time()

    sets = []
    values = []

    try:
        while ti.readable():
            line = ti.readline().decode('utf-8').strip()

            if 'Start' in line:
                sets = []
                start_sensing = True
                start_time = time.time()
            elif start_sensing:
                if 'S' in line[:2]:
                    tmp = '[NUM] ' + str(len(sets) + 1) + ': ' + str(len(values))
                    log.append(tmp)
                    print(tmp)
                    if first and not values:
                        first = False
                    else:
                        first = False
                        sets.append(values)
                        values = []
                    if len(sets) is MIN:
                        tmp = '[INFO] Enough Data'
                        log.append(tmp)
                        print(tmp)
                    if len(sets) > cnt:
                        tmp = '[INFO] End'
                        log.append(tmp)
                        print(tmp)
                        break
                else:
                    key, msg = line.split(':')[:2]
                    if key == code:
                        values.append(float(msg.strip()))
                    else:
                        tmp = '[ERR] Message Error: ' + line
                        log.append(tmp)
                        print(tmp)
                        break
            else:
                pass
        else:
            tmp = '[ERR] Serial Connection Error'
            log.append(tmp)
            print(tmp)
    except Exception as e:
        tmp = '[EXCEPT] ' + str(e)
        log.append(tmp)
        print(tmp)

    return sets, log, time.time() - start_time


def save(sets: List[List], sample: str, category: str, path: str = PATH) -> bool:
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


def save_log(sample: str, log: List[str], path: str = PATH):
    with open(path + sample + '.txt', 'w') as f:
        f.write("\n".join(log))
