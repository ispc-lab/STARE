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

def esot500_frame_all():
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

    dataset = get_dataset(
        'esot_500_2','esot_250_2','esot_20_2',
        'esot_500_50','esot_250_50','esot_20_50',
        'esot_500_100','esot_250_100','esot_20_100',
        'esot_500_150','esot_250_150','esot_20_150',
    )

    return trackers, dataset
