from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone, NetWrapper

def parameters():
    params = TrackerParams()
    params.id = 16
    """Standard Setting Part"""
    """ id:16 is designed to simulate the running of slower runtime 10fps and adaptive sampling"""
    # Hardware
    params.convert_time = False
    params.sim = True
    params.sim_runtime_rt = 33*1e3 # unit:us
    params.sim_runtime = {
        'atom_default': 14*1e3,
        'atom_fe240':   14*1e3,
        'dimp_dimp18':  10*1e3,
        'dimp_dimp18_fe240':10*1e3,
        'dimp_prdimp18':16*1e3,
        'dimp_prdimp18_fe240':16*1e3,
        'dimp_dimp50':  12*1e3,
        'keep_track_default':27*1e3,
        'kys_default':25*1e3,
        'kys_fe240':25*1e3,
        'rts_rts50':20*1e3,
        'tomp_tomp50':20*1e3,
        'ostrack_vitb_256_mae_ce_32x4_ep300':16*1e3,
        'ostrack_trial4_vitb_256_mae_ce_32x4_aligned':16*1e3,
        'ostrack_trial6_ostrack256_aug1':16*1e3,
        'ostrack_trial8_ostrack256':16*1e3,
        'ostrack_trial9_ostrack256':16*1e3,
        'ostrack_baseline':100*1e3,
        'ostrack_aug':100*1e3,
        'stark_s_baseline':16*1e3,
                          }
    
    params.sim_runtime_init = {
        'atom_default': 273*1e3,
        'atom_fe240':   247*1e3,
        'dimp_dimp18':  56*1e3,
        'dimp_dimp18_fe240':55*1e3,
        'dimp_prdimp18':55*1e3,
        'dimp_prdimp18_fe240':55*1e3,
        'dimp_dimp50':  80*1e3,
        'keep_track_default':495*1e3,
        'kys_default':81*1e3,
        'kys_fe240':80*1e3,
        'rts_rts50':136*1e3,
        'tomp_tomp50':7*1e3,
        'ostrack_vitb_256_mae_ce_32x4_ep300':10*1e3,
        'ostrack_trial4_vitb_256_mae_ce_32x4_aligned':10*1e3,
        'ostrack_trial6_ostrack256_aug1':10*1e3,
        'ostrack_trial8_ostrack256':10*1e3,
        'ostrack_trial9_ostrack256':10*1e3,
        'ostrack_baseline':100*1e3,
        'ostrack_aug':100*1e3,
        'stark_s_baseline':25*1e3,
                          }
    params.sim_disturb = 0.0 # disturbance around runtime

    # Template
    params.template_ = 'default' # default means using the first slice according 
    # params.template_ = 'augmented'
    # params.template_ = 'seperate'
    params.init_time = True # whether to consider initialization latency

    '''User-define Part'''
    # Slicing
    params.slicing = 'Adaptive' # FxTime, FxNum, Adaptive, Last
    params.window_size = 20*1e3 # unit:us
    params.window_size_template = 20*1e3 # unit:us
    params.num_events = 5000
    params.adaptive_ = '' # Adaptive method, e.g. sampling

    # Representation
    params.representation = 'VoxelGridComplex'

    return params
