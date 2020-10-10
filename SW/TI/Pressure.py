from SW.signal import name
from SW.TI.parsing import measure_by_time, save

if __name__ == '__main__':
    # pressure_sets, _, _ = measure_by_cnt('COM9', 2000000, 10, 'P')
    # _ = save(pressure_sets, name, 'pressure')
    pressure_sets, _, _ = measure_by_time('COM9', 2000000, 10., 'P')
    _ = save([[], pressure_sets, pressure_sets], name, 'pressure')
