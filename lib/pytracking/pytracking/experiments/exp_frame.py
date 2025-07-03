from pytracking.evaluation import Tracker, get_dataset, trackerlist, load_stream_setting


# Note:
# range(1) means run_id=0, the tracking results will
# be saved in pytracking/output/tracking_results/{tracker_name}/{tracker_params}_{run_id}/
# e.g. pytracking/output/tracking_results/atom/default_000/
# if you set run_id=None, the tracking results will
# be saved in pytracking/output/tracking_results/{tracker_name}/{tracker_params}/
# e.g. pytracking/output/tracking_results/atom/default/

def fast_test_offline():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1))
    dataset = get_dataset('esot_20_50')
    return trackers, dataset