from SW.signal import name
from SW.TI.parsing import measure_by_cnt, save

if __name__ == '__main__':
    heat_sets, _, _ = measure_by_cnt('COM9', 2000000, 15, 'T')
    _ = save([[], heat_sets, heat_sets], name, 'heat')
