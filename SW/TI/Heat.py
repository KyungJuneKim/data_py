from SW.signal import NAME
from SW.TI.parsing import measure_by_cnt, save, save_log

if __name__ == '__main__':
    heat_sets, log, _ = measure_by_cnt('COM9', 2000000, 15, 'T')
    _ = save([[], heat_sets, heat_sets], NAME, 'heat')
    save_log(NAME, log)
