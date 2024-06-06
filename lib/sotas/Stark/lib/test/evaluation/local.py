from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    # settings.got10k_lmdb_path = '/home/test4/code/EventBenchmark/data/got10k_lmdb'
    # settings.got10k_path = '/home/test4/code/EventBenchmark/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    # settings.itb_path = '/home/test4/code/EventBenchmark/data/itb'
    # settings.lasot_extension_subset_path_path = '/home/test4/code/EventBenchmark/data/lasot_extension_subset'
    # settings.lasot_lmdb_path = '/home/test4/code/EventBenchmark/data/lasot_lmdb'
    # settings.lasot_path = '/home/test4/code/EventBenchmark/data/lasot'
    settings.network_path = '/home/test4/code/EventBenchmark/output/test/networks'    # Where tracking networks are stored.
    # settings.nfs_path = '/home/test4/code/EventBenchmark/data/nfs'
    # settings.otb_path = '/home/test4/code/EventBenchmark/data/otb'
    settings.prj_dir = '/home/test4/code/EventBenchmark/lib/sotas/Stark'
    settings.result_plot_path = '/home/test4/code/EventBenchmark/output/test/result_plots'
    settings.results_path = '/home/test4/code/EventBenchmark/output/test/tracking_results'    # Where to store tracking results
    settings.results_path_rt = '/home/test4/code/EventBenchmark/lib/pytracking/pytracking/output/tracking_results_rt_raw/'    # Where to store tracking results
    settings.results_path_rt_final = '/home/test4/code/EventBenchmark/output/test/tracking_results_rt_final'

    settings.save_dir = '/home/test4/code/EventBenchmark/output'
    settings.segmentation_path = '/home/test4/code/EventBenchmark/output/test/segmentation_results'
    settings.tc128_path = '/home/test4/code/EventBenchmark/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/test4/code/EventBenchmark/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/test4/code/EventBenchmark/data/trackingnet'
    settings.uav_path = '/home/test4/code/EventBenchmark/data/uav'
    settings.vot18_path = '/home/test4/code/EventBenchmark/data/vot2018'
    settings.vot22_path = '/home/test4/code/EventBenchmark/data/vot2022'
    settings.vot_path = '/home/test4/code/EventBenchmark/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.eventsot_dir = '/home/test4/code/EventBenchmark/data/EventSOT'
    settings.esot500_dir = '/home/test4/code/EventBenchmark/data/EventSOT500/'
    settings.esot2_dir = '/home/test4/code/EventBenchmark/data/EventSOT2'
    settings.eventcarla_dir = '/home/test4/code/EventBenchmark/data/EventSOT/EventCARLA/VoxelGrid'
    settings.fe240_dir = '/home/test4/code/EventBenchmark/data/FE240'
    settings.visEvent_dir = '/home/test4/code/EventBenchmark/data/VisEvent'
    
    return settings

