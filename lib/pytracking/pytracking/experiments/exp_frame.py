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
                trackerlist('keep_track', 'default', range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('egt', 'egt', range(1))

    dataset = get_dataset(
        'esot_500_2', 'esot_250_2', 'esot_20_2',
        'esot_500_20', 'esot_250_20', 'esot_20_20',
        'esot_500_50', 'esot_250_50', 'esot_20_50',
        'esot_500_100', 'esot_250_100', 'esot_20_100',
        'esot_500_150', 'esot_250_150', 'esot_20_150',
    )

    return trackers, dataset

def esot500_sim_frame_egt_500_w2ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:201 is designed to simulate the frame-based (500FPS, 2ms) evaluation for egt to reproduce the results
    stream_setting_id = 201
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_500_w20ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:202 is designed to simulate the frame-based (500FPS, 20ms) evaluation for egt to reproduce the results
    stream_setting_id = 202
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_500_w50ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:203 is designed to simulate the frame-based (500FPS, 50ms) evaluation for egt to reproduce the results
    stream_setting_id = 203
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_500_w100ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:204 is designed to simulate the frame-based (500FPS, 100ms) evaluation for egt to reproduce the results
    stream_setting_id = 204
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_500_w150ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:205 is designed to simulate the frame-based (500FPS, 150ms) evaluation for egt to reproduce the results
    stream_setting_id = 205
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_250_w2ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:206 is designed to simulate the frame-based (250FPS, 2ms) evaluation for egt to reproduce the results
    stream_setting_id = 206
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_250_w20ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:207 is designed to simulate the frame-based (250FPS, 20ms) evaluation for egt to reproduce the results
    stream_setting_id = 207
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_250_w50ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:208 is designed to simulate the frame-based (250FPS, 50ms) evaluation for egt to reproduce the results
    stream_setting_id = 208
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_250_w100ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:209 is designed to simulate the frame-based (250FPS, 100ms) evaluation for egt to reproduce the results
    stream_setting_id = 209
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_250_w150ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:210 is designed to simulate the frame-based (250FPS, 150ms) evaluation for egt to reproduce the results
    stream_setting_id = 210
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_20_w2ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:211 is designed to simulate the frame-based (20FPS, 2ms) evaluation for egt to reproduce the results
    stream_setting_id = 211
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_20_w20ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:212 is designed to simulate the frame-based (20FPS, 20ms) evaluation for egt to reproduce the results
    stream_setting_id = 212
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_20_w50ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:213 is designed to simulate the frame-based (20FPS, 50ms) evaluation for egt to reproduce the results
    stream_setting_id = 213
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_20_w100ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:214 is designed to simulate the frame-based (20FPS, 100ms) evaluation for egt to reproduce the results
    stream_setting_id = 214
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting

def esot500_sim_frame_egt_20_w150ms():
    trackers =  trackerlist('egt', 'egt', range(1))
    dataset = get_dataset('esot500s')

    # id:215 is designed to simulate the frame-based (20FPS, 150ms) evaluation for egt to reproduce the results
    stream_setting_id = 215
    stream_setting = load_stream_setting(f's{stream_setting_id}')

    return trackers, dataset, stream_setting
    
