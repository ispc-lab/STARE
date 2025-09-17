from pytracking.evaluation import Tracker, get_dataset, trackerlist, load_stream_setting


# Note:
# Currently, default run_id is None and stream_setting_id=100, the tracking results will eventually
# be saved in pytracking/output/tracking_results_rt_final/{tracker_name}/{tracker_params}/{stream_setting_id}/
# e.g. pytracking/output/tracking_results_rt_final/atom/default/100/
# if you set run_id=0, the tracking results will
# be saved in pytracking/output/tracking_results_rt_final/{tracker_name}/{tracker_params}_{run_id}/{stream_setting_id}/
# e.g. pytracking/output/tracking_results_rt_final/atom/default_000/100/

trackers_fast_test =  trackerlist('atom', 'default') + \
            trackerlist('dimp', 'dimp18') + \
            trackerlist('kys', 'default')

def fast_test_stare():
    trackers = trackers_fast_test
    dataset = get_dataset('esot500s')
    stream_setting_id = 100  # Default streaming setting, for real-time testing on your own hardware.
    stream_setting = load_stream_setting(f's{stream_setting_id}')
    return trackers, dataset, stream_setting


trackers_stare_all = trackerlist('dimp', 'dimp18') + \
                trackerlist('dimp', 'dimp18_esot500') + \
                trackerlist('dimp', 'dimp50') + \
                trackerlist('dimp', 'dimp50_esot500') + \
                trackerlist('kys', 'default') + \
                trackerlist('kys', 'esot500') + \
                trackerlist('atom', 'default') + \
                trackerlist('atom', 'esot500') + \
                trackerlist('tomp', 'tomp50') + \
                trackerlist('tomp', 'tomp50_esot500') + \
                trackerlist('tomp', 'tomp101') + \
                trackerlist('tomp', 'tomp101_esot500') + \
                trackerlist('dimp', 'prdimp18') + \
                trackerlist('dimp', 'prdimp50') + \
                trackerlist('keep_track', 'default') + \
                trackerlist('rts', 'rts50') + \
                trackerlist('egt', 'egt')

trackers_stare_esot500h = trackerlist('kys', 'default') + \
                trackerlist('kys', 'esot500') + \
                trackerlist('dimp', 'prdimp18') + \
                trackerlist('keep_track', 'default') + \
                trackerlist('rts', 'rts50') + \
                trackerlist('egt', 'egt')

# for sim real-time testing on your own hardware.
def esot500_stare_w2ms():
    trackers =  trackers_stare_all
    dataset = get_dataset('esot500s')

    # id:101 is designed to simulate the real-time running of STARE (2ms) to reproduce the results
    stream_setting_id = 101
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500h_stare_w2ms():
    trackers =  trackers_stare_esot500h
    dataset = get_dataset('esot500hs')

    # id:101 is designed to simulate the real-time running of STARE (2ms) to reproduce the results
    stream_setting_id = 101
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500h_stare_w8ms():
    trackers =  trackers_stare_esot500h
    dataset = get_dataset('esot500hs')

    # id:107 is designed to simulate the real-time running of STARE (8ms) to reproduce the results
    stream_setting_id = 107
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_stare_w20ms():
    trackers =  trackers_stare_all
    dataset = get_dataset('esot500s')

    # id:102 is designed to simulate the real-time running of STARE (20ms) to reproduce the results
    stream_setting_id = 102
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500h_stare_w20ms():
    trackers =  trackers_stare_esot500h
    dataset = get_dataset('esot500hs')

    # id:102 is designed to simulate the real-time running of STARE (20ms) to reproduce the results
    stream_setting_id = 102
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_stare_w50ms():
    trackers =  trackers_stare_all
    dataset = get_dataset('esot500s')

    # id:103 is designed to simulate the real-time running of STARE (50ms) to reproduce the results
    stream_setting_id = 103
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500h_stare_w50ms():
    trackers =  trackers_stare_esot500h
    dataset = get_dataset('esot500hs')

    # id:103 is designed to simulate the real-time running of STARE (50ms) to reproduce the results
    stream_setting_id = 103
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_stare_w100ms():
    trackers =  trackers_stare_all
    dataset = get_dataset('esot500s')

    # id:104 is designed to simulate the real-time running of STARE (100ms) to reproduce the results
    stream_setting_id = 104
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_stare_w150ms():
    trackers =  trackers_stare_all
    dataset = get_dataset('esot500s')

    # id:105 is designed to simulate the real-time running of STARE (150ms) to reproduce the results
    stream_setting_id = 105
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_stare_w200ms():
    trackers =  trackers_stare_all
    dataset = get_dataset('esot500s')

    # id:106 is designed to simulate the real-time running of STARE (200ms) to reproduce the results
    stream_setting_id = 106
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting
