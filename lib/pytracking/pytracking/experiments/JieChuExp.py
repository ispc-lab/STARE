from pytracking.evaluation import Tracker, get_dataset, trackerlist, load_stream_setting


trackers4improve1st =  trackerlist('atom', 'default',range(5)) + \
            trackerlist('atom', 'fe240',range(5)) + \
            trackerlist('dimp', 'dimp18',range(5)) + \
            trackerlist('dimp', 'dimp18_fe240',range(5)) +\
            trackerlist('dimp', 'prdimp18',range(5)) +\
            trackerlist('dimp', 'dimp50',range(5)) + \
            trackerlist('kys', 'default',range(5)) + \
            trackerlist('kys', 'fe240',range(5)) + \
            trackerlist('rts','rts50',range(5)) + \
            trackerlist('keep_track','default',range(5)) + \
            trackerlist('tomp','tomp50',range(5))

trackers_transferd_JieChu_esot500 =  trackerlist('dimp', 'JieChu_dimp50_esot500') + \
            trackerlist('tomp', 'JieChu_tomp50_esot500') 
            # trackerlist('keeptrack', 'JieChu_esot500')
            # trackerlist('tomp', 'JieChu_tomp101_esot500')

allTrackers_esot500 = trackerlist('atom', 'esot500') + \
            trackerlist('dimp', 'dimp18_esot500') + \
            trackerlist('dimp', 'JieChu_dimp50_esot500') + \
            trackerlist('dimp', 'prdimp18_esot500') + \
            trackerlist('kys', 'esot500') + \
            trackerlist('tomp', 'JieChu_tomp50_esot500') + \
            trackerlist('tomp', 'JieChu_tomp101_esot500')+\
            trackerlist('dimp', 'JieChu_prdimp50_esot500') 


allTrackers_fe240 = trackerlist('atom', 'fe240') + \
            trackerlist('dimp', 'dimp18_fe240') + \
            trackerlist('dimp', 'JieChu_dimp50_fe240') + \
            trackerlist('dimp', 'prdimp18_fe240') + \
            trackerlist('kys', 'fe240') + \
            trackerlist('tomp', 'tomp50_fe240') + \
            trackerlist('tomp', 'JieChu_tomp101_fe240')
            # trackerlist('prdimp', 'JieChu_prdimp50_fe240') + \


allTrackers_default = trackerlist('atom', 'default') + \
            trackerlist('dimp', 'dimp18') + \
            trackerlist('dimp', 'dimp50') + \
            trackerlist('dimp', 'prdimp18') + \
            trackerlist('dimp', 'prdimp50') + \
            trackerlist('keep_track', 'default') + \
            trackerlist('kys', 'default') + \
            trackerlist('tomp', 'tomp50') + \
            trackerlist('tomp', 'tomp101')+\
            trackerlist('rts', 'rts50')#added at 2023/10/19-21:55-Thu, when JieChu implement the exp related to the appendix Tab.2

trackers_added_offline=trackerlist('atom', 'esot500') + \
            trackerlist('dimp', 'dimp18_esot500') + \
            trackerlist('dimp', 'JieChu_dimp50_esot500') + \
            trackerlist('dimp', 'prdimp18_esot500') + \
            trackerlist('kys', 'esot500') + \
            trackerlist('tomp', 'JieChu_tomp50_esot500')+ \
            trackerlist('dimp', 'prdimp50') + \
            trackerlist('tomp', 'tomp101')+\
            trackerlist('tomp', 'JieChu_tomp101_esot500')
            # trackerlist('dimp', 'JieChu_prdimp50_esot500') 

trackers_added_online=trackerlist('dimp', 'prdimp50') + \
            trackerlist('tomp', 'tomp101') + \
            trackerlist('atom', 'esot500') + \
            trackerlist('dimp', 'dimp18_esot500') + \
            trackerlist('dimp', 'JieChu_dimp50_esot500') + \
            trackerlist('dimp', 'prdimp18_esot500') + \
            trackerlist('kys', 'esot500') + \
            trackerlist('tomp', 'JieChu_tomp50_esot500') + \
            trackerlist('tomp', 'JieChu_tomp101_esot500')+\
            trackerlist('dimp', 'JieChu_prdimp50_esot500')

def streaming_trackers4improve1st():
    trackers =  trackers_single_5range
    # dataset = get_dataset('esot500s','esot2s')
    dataset = get_dataset('esot500s')
    stream_setting = load_stream_setting('s18')
    return trackers, dataset, stream_setting

def streaming_transferd_JieChu_esot500():
    trackers=trackers_transferd_JieChu_esot500
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('s14')
    return trackers, dataset, stream_setting


def offline_trackers_added_500fps():
    trackers=trackers_added_offline
    datasets=get_dataset("esot_500_2","esot_500_8","esot_500_20")
    return trackers, datasets

def offline_trackers_added_250fps():
    trackers=trackers_added_offline
    datasets=get_dataset("esot_250_2","esot_250_8","esot_250_20")
    return trackers, datasets

def offline_trackers_added_20fps():
    trackers=trackers_added_offline
    datasets=get_dataset("esot_20_2","esot_20_8","esot_20_20")
    return trackers, datasets

def online_1K_F():
    trackers=allTrackers_esot500+allTrackers_default
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('JieChu_s36_1k')
    return trackers, dataset, stream_setting

def online_5K_F():
    trackers=allTrackers_esot500+allTrackers_default
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('JieChu_s42_5k')
    return trackers, dataset, stream_setting

def online_10K_F():
    trackers=allTrackers_esot500+allTrackers_default
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('JieChu_s37_10k')
    return trackers, dataset, stream_setting

def online_2ms_F():
    trackers=allTrackers_esot500+allTrackers_default
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('JieChu_s38_2msF')
    return trackers, dataset, stream_setting

def online_20ms_F():
    trackers=allTrackers_esot500+allTrackers_default
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('JieChu_s43_20msF')
    return trackers, dataset, stream_setting

def online_100ms_F():
    trackers=allTrackers_esot500+allTrackers_default
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('JieChu_s39_100msF')
    return trackers, dataset, stream_setting

def online_10ms_A():
    trackers=allTrackers_esot500+allTrackers_default
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('JieChu_s40_10msA')
    return trackers, dataset, stream_setting

def online_20ms_A():
    trackers=allTrackers_esot500+allTrackers_default
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('JieChu_s44_20msA')
    return trackers, dataset, stream_setting

def online_50ms_A():
    trackers=allTrackers_esot500+allTrackers_default
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting('JieChu_s41_50msA')
    return trackers, dataset, stream_setting

def online_trackers_added():
    trackers=trackers_added_online
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting("s14")
    return trackers, dataset, stream_setting

def online_atom_esot500():
    trackers=trackerlist('atom', 'esot500')
    dataset=get_dataset('esot500s')
    stream_setting = load_stream_setting("s14")
    return trackers, dataset, stream_setting

def offline_atom_esot500_500fps():
    trackers=trackerlist('atom', 'esot500')
    datasets=get_dataset("esot_500_2","esot_500_8","esot_500_20")
    return trackers, datasets

def offline_atom_esot500_250fps():
    trackers=trackerlist('atom', 'esot500')
    datasets=get_dataset("esot_250_2","esot_250_8","esot_250_20")
    return trackers, datasets

def debug_test():
    trackers=trackerlist('atom', 'esot500')
    datasets=get_dataset("esot_250_2")
    return trackers, datasets