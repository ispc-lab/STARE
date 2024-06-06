from pytracking.evaluation import Tracker, get_dataset, trackerlist, load_stream_setting


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset

def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def default_test():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('keep_track','default',range(1)) +\
                trackerlist('dimp', 'prdimp18', range(1))
    dataset = get_dataset('esot500')
    return trackers, dataset

def default_online():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('keep_track','default',range(1)) +\
                trackerlist('dimp', 'prdimp18', range(1))
    dataset = get_dataset('esot500s')
    return trackers, dataset

def esot500_offline():
    trackers =  trackerlist('atom', 'esot500', range(1)) + \
                trackerlist('dimp', 'dimp18_esot500', range(1)) + \
                trackerlist('dimp', 'prdimp18_esot500', range(1))
    dataset = get_dataset('esot500')
    return trackers, dataset

def esotVH_offline():
    trackers =  trackerlist('atom', 'default', 99) + \
                trackerlist('dimp', 'dimp18', 99) + \
                trackerlist('kys', 'default', 99) + \
                trackerlist('rts', 'rts50', 99) + \
                trackerlist('keep_track','default', 99) +\
                trackerlist('dimp', 'prdimp18', 99)
    dataset = get_dataset('esoth')
    return trackers, dataset
def esotVM_offline():
    trackers =  trackerlist('atom', 'default', 66) + \
                trackerlist('dimp', 'dimp18', 66) + \
                trackerlist('kys', 'default', 66) + \
                trackerlist('rts', 'rts50', 66) + \
                trackerlist('keep_track','default',66) +\
                trackerlist('dimp', 'prdimp18', 66)
    dataset = get_dataset('esotm')
    return trackers, dataset

def esotVL_offline():
    trackers =  trackerlist('atom', 'default', 33) + \
                trackerlist('dimp', 'dimp18', 33) + \
                trackerlist('kys', 'default', 33) + \
                trackerlist('rts', 'rts50', 33) + \
                trackerlist('keep_track','default',33) +\
                trackerlist('dimp', 'prdimp18', 33)
    dataset = get_dataset('esotl')
    return trackers, dataset

def esot500_fps_window():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('keep_track','default',range(1)) +\
                trackerlist('dimp', 'prdimp18', range(1))
    dataset = get_dataset('esot_500_2','esot_250_2','esot_20_2','esot_500_8','esot_250_8','esot_20_8','esot_500_20','esot_250_20','esot_20_20')

    return trackers, dataset
def esot2_interp_default():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('keep_track','default',range(1)) +\
                trackerlist('dimp', 'prdimp18', range(1))
    dataset = get_dataset('esot2_2_20','esot2_default','esot2_5_20','esot2_10_20','esot2_2_100','esot2_5_100','esot2_10_100')
    return trackers, dataset

def esot500_fps_window_fe():
    trackers =  trackerlist('atom', 'fe240', range(3)) + \
                trackerlist('dimp', 'dimp18_fe240', range(3)) + \
                trackerlist('dimp', 'prdimp18_fe240', range(3)) +\
                trackerlist('kys', 'fe240', range(3)) + \
                trackerlist('tomp','tomp50_fe240',range(3))
    dataset = get_dataset('esot_500_2','esot_250_2','esot_20_2','esot_500_8','esot_250_8','esot_20_8','esot_500_20','esot_250_20','esot_20_20')

    return trackers, dataset
def esot2_interp_fe():
    trackers =  trackerlist('atom', 'fe240', range(3)) + \
                trackerlist('dimp', 'dimp18_fe240', range(3)) + \
                trackerlist('dimp', 'prdimp18_fe240', range(3)) +\
                trackerlist('kys', 'fe240', range(3)) + \
                trackerlist('tomp','tomp50_fe240',range(3))
    dataset = get_dataset('esot2_2_20','esot2_default','esot2_5_20','esot2_10_20','esot2_2_100','esot2_5_100','esot2_10_100')
    return trackers, dataset

