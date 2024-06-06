from pytracking.evaluation import Tracker, get_dataset, trackerlist, load_stream_setting
def offline_ultimate():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('atom', 'fe240', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('dimp', 'dimp18_fe240', range(1)) +\
                trackerlist('dimp', 'prdimp18', range(1)) +\
                trackerlist('dimp', 'prdimp18_fe240', range(1)) +\
                trackerlist('dimp', 'dimp50', range(1)) + \
                trackerlist('dimp', 'prdimp50', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('kys', 'fe240', range(1)) + \
                trackerlist('rts','rts50', range(1)) + \
                trackerlist('keep_track','default', range(1)) + \
                trackerlist('tomp','tomp50', range(1))
    dataset = get_dataset('esot_500_2','esot_250_2','esot_20_2','esot_500_8','esot_250_8','esot_20_8','esot_500_20','esot_250_20','esot_20_20','esot2_default','esot2_2_20','esot2_5_20','esot2_10_20')
    return trackers, dataset
