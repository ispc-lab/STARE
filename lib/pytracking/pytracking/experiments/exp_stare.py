from pytracking.evaluation import Tracker, get_dataset, trackerlist, load_stream_setting


# Note:
# Currently, default run_id is None and stream_setting_id=0, the tracking results will eventually
# be saved in pytracking/output/tracking_results_rt_final/{tracker_name}/{tracker_params}/{stream_setting_id}/
# e.g. pytracking/output/tracking_results_rt_final/atom/default/0/
# if you set run_id=0, the tracking results will
# be saved in pytracking/output/tracking_results_rt_final/{tracker_name}/{tracker_params}_{run_id}/{stream_setting_id}/
# e.g. pytracking/output/tracking_results_rt_final/atom/default_000/0/

trackers_fast_test =  trackerlist('atom', 'default') + \
            trackerlist('dimp', 'dimp18') + \
            trackerlist('kys', 'default')

def fast_test_stare():
    trackers = trackers_fast_test
    dataset = get_dataset('esot500s')
    stream_setting_id = 0  # Default streaming setting, for real-time testing on your own hardware.
    stream_setting = load_stream_setting(f's{stream_setting_id}')
    return trackers, dataset, stream_setting

def esot500_stare_all():
    trackers =  trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('dimp', 'dimp18_esot500', range(1)) + \
                trackerlist('dimp', 'dimp50', range(1)) + \
                trackerlist('dimp', 'dimp50_esot500', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('kys', 'esot500', range(1)) + \
                trackerlist('atom', 'default', range(1)) + \
                trackerlist('atom', 'esot500', range(1)) + \
                trackerlist('tomp', 'tomp50', range(1)) + \
                trackerlist('tomp', 'tomp50_esot500', range(1)) + \
                trackerlist('tomp', 'tomp101', range(1)) + \
                trackerlist('tomp', 'tomp101_esot500', range(1)) + \
                trackerlist('dimp', 'prdimp18', range(1)) + \
                trackerlist('dimp', 'prdimp50', range(1)) + \
                trackerlist('keep_track','default',range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('egt', 'egt', range(1))

    dataset = get_dataset('esot500s')
    
    stream_setting_id = 14  # for sim real-time testing on your own hardware.
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting
