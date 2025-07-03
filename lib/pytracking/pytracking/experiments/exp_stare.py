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

def fast_test_streaming():
    trackers = trackers_fast_test
    dataset = get_dataset('esot500s')
    stream_setting_id = 0  # Default streaming setting, for real-time testing on your own hardware.
    stream_setting = load_stream_setting(f's{stream_setting_id}')
    return trackers, dataset, stream_setting