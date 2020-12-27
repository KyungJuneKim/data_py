from SW.signal import NAME
from SW.TI.parsing import measure_all_by_cnt, save, save_log

if __name__ == '__main__':
    pressure_sets, temperature_sets, log, _ = measure_all_by_cnt('COM11', 2000000, 300)
    _ = save(pressure_sets, NAME, 'pressure')
    _ = save(temperature_sets, NAME, 'heat')
    save_log(NAME, log)
