from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone, NetWrapper

def parameters():
    params = TrackerParams()
    params.id = 107

    """Standard Setting Part"""
    """ id:101 is designed to simulate the real-time running of STARE (8ms) to reproduce the results"""

    # Hardware
    params.convert_time = False
    params.sim = True
    params.sim_runtime_rt = 33 * 1e3 # default run time, unit:us
    params.sim_runtime = { # unit:us
        # atom
        'atom_default': 14 * 1e3,
        'atom_fe240': 14 * 1e3,
        'atom_esot500': 14 * 1e3,
        # dimp
        'dimp_dimp18': 10 * 1e3,
        'dimp_dimp18_fe240': 10 * 1e3,
        'dimp_dimp18_esot500': 10 * 1e3,
        'dimp_dimp50': 12 * 1e3,
        'dimp_dimp50_esot500': 12 * 1e3,
        # prdimp
        'dimp_prdimp18': 16 * 1e3,
        'dimp_prdimp18_fe240': 16 * 1e3,
        'dimp_prdimp18_esot500': 16 * 1e3,
        'dimp_prdimp50': 19 * 1e3,
        'dimp_prdimp50_esot500': 19 * 1e3,
        # keep_track
        'keep_track_default': 27 * 1e3,
        # kys
        'kys_default': 25 * 1e3,
        'kys_fe240': 25 * 1e3,
        'kys_esot500': 25 * 1e3,
        # rts
        'rts_rts50': 20 * 1e3,
        # tomp
        'tomp_tomp50': 20 * 1e3,
        'tomp_tomp50_esot500': 20 * 1e3,
        'tomp_tomp101': 24 * 1e3,
        'tomp_tomp101_esot500': 24 * 1e3,
        # ostrack
        'ostrack_pred_esot500_bs4': 8.5 * 1e3,
        'ostrack_pred_esot500mix_bs4': 8.5 * 1e3,
        'ostrack_trial4_vitb_256_mae_ce_32x4_aligned': 8.5 * 1e3,
        'ostrack_trial6_ostrack256_aug1': 8.5 * 1e3,
        'ostrack_trial8_ostrack256': 8.5 * 1e3,
        'ostrack_trial9_ostrack256': 8.5 * 1e3,
        'ostrack_baseline': 8.5 * 1e3,
        'ostrack_aug': 8.5 * 1e3,
        'ostrack_esot500mix': 8.5 * 1e3,
        'ostrack_esot500_baseline_10000': 8.5 * 1e3,
        'ostrack_esot500_baseline': 8.5 * 1e3,
        'ostrack_pred_esot500_4step': 9.5 * 1e3,
        # mixformer
        'mixformer_convmae_online_baseline': 15 * 1e3,
        'mixformer_convmae_pred_esot500': 16 * 1e3,
        # stark
        'stark_s_baseline': 16 * 1e3,
        # egt
        'egt_egt': 40 * 1e3,
    }
    
    params.sim_runtime_init = {
        # atom
        'atom_default': 273 * 1e3,
        'atom_fe240': 247 * 1e3,
        'atom_esot500': 247 * 1e3,
        # dimp
        'dimp_dimp18': 56 * 1e3,
        'dimp_dimp18_fe240': 55 * 1e3,
        'dimp_dimp18_esot500': 55 * 1e3,
        'dimp_dimp50': 80 * 1e3,
        'dimp_dimp50_esot500': 80 * 1e3,
        # prdimp
        'dimp_prdimp18': 55 * 1e3,
        'dimp_prdimp18_fe240': 55 * 1e3,
        'dimp_prdimp18_esot500': 55 * 1e3,
        'dimp_prdimp50': 100 * 1e3,
        'dimp_prdimp50_esot500': 100 * 1e3,
        # keep_track
        'keep_track_default': 495 * 1e3,
        # kys
        'kys_default': 81 * 1e3,
        'kys_fe240': 80 * 1e3,
        'kys_esot500': 80 * 1e3,
        # rts
        'rts_rts50': 136 * 1e3,
        # tomp
        'tomp_tomp50': 7 * 1e3,
        'tomp_tomp50_esot500': 7 * 1e3,
        'tomp_tomp101': 14 * 1e3,
        'tomp_tomp101_esot500': 14 * 1e3,
        # ostrack
        'ostrack_pred_esot500_bs4': 1 * 1e3,
        'ostrack_pred_esot500mix_bs4': 1 * 1e3,
        'ostrack_trial4_vitb_256_mae_ce_32x4_aligned': 1 * 1e3,
        'ostrack_trial6_ostrack256_aug1': 1 * 1e3,
        'ostrack_trial8_ostrack256': 1 * 1e3,
        'ostrack_trial9_ostrack256': 1 * 1e3,
        'ostrack_baseline': 1 * 1e3,
        'ostrack_aug': 1 * 1e3,
        'ostrack_esot500mix': 1 * 1e3,
        'ostrack_esot500_baseline_10000': 1 * 1e3,
        'ostrack_esot500_baseline': 1 * 1e3,
        'ostrack_pred_esot500_4step': 1 * 1e3,
        # mixformer
        'mixformer_convmae_online_baseline': 26 * 1e3,
        'mixformer_convmae_pred_esot500': 26 * 1e3,
        # stark
        'stark_s_baseline': 25 * 1e3,
        # egt
        'egt_egt': 1 * 1e3,
    }

    # disturbance around runtime
    params.sim_disturb = 0.0

    # Template
    params.template_ = 'default' # default means using the first slice according 
    # params.template_ = 'augmented'
    # params.template_ = 'seperate'
    params.init_time = True # whether to consider initialization latency

    '''User-define Part'''
    # Slicing
    params.slicing = 'FxTime' # FxTime, FxNum, Adaptive, Last
    params.window_size = 8 * 1e3 # unit:us
    params.window_size_template = 8 * 1e3 # unit:us
    params.num_events = 5000
    params.adaptive_ = '' # Adaptive method, e.g. sampling

    # Representation
    params.representation = 'VoxelGridComplex'

    return params
