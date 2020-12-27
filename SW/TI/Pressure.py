from SW.signal import NAME
from SW.TI.parsing import measure_by_time, save, save_log

if __name__ == '__main__':
    # pressure_sets, log, _ = measure_by_cnt('COM9', 2000000, 10, 'P')
    # _ = save(pressure_sets, NAME, 'pressure')
    # save_log(NAME, log)
    pressure_sets, log, _ = measure_by_time('COM9', 2000000, 10, 'P')
    _ = save([[], pressure_sets, pressure_sets], NAME, 'pressure')
    save_log(NAME, log)
