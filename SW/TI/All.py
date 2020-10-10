from SW.signal import name
from SW.TI.parsing import measure_all_by_cnt, save

if __name__ == '__main__':
    pressure_sets, temperature_sets, _, _ = measure_all_by_cnt('COM5', 2000000, 20)
    _ = save(pressure_sets, name, 'pressure')
    _ = save(temperature_sets, name, 'heat')
