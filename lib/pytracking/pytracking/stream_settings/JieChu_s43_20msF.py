from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone, NetWrapper

def parameters():
    params = TrackerParams()
    params.id = 43
    """Standard Setting Part"""
    """ id:14 is designed to simulate the running of id:18 to reproduce the results"""
    # Hardware
    params.convert_time = False
    params.sim = True
    params.sim_runtime_rt = 33*1e3 # unit:us
    params.sim_runtime = {
        'atom_default': 14*1e3,
        'atom_fe240':   14*1e3,
        'atom_esot500':   14*1e3,#added by JieChu
        'dimp_dimp18':  10*1e3,
        'dimp_dimp18_fe240':10*1e3,
        'dimp_dimp18_esot500':10*1e3,#added by JieChu
        'dimp_prdimp18':16*1e3,
        'dimp_prdimp18_fe240':16*1e3,
        'dimp_prdimp18_esot500':16*1e3,#added by JieChu
        'dimp_prdimp50':19*1e3,#added by JieChu, however uncertain...
        'dimp_JieChu_prdimp50_esot500':19*1e3,#added by JieChu, however uncertain...
        'dimp_dimp50':  12*1e3,
        'dimp_JieChu_dimp50_esot500':  12*1e3,#added by JieChu
        'keep_track_default':27*1e3,
        'kys_default':25*1e3,
        'kys_fe240':25*1e3,
        'kys_esot500':25*1e3,#added by JieChu
        'rts_rts50':20*1e3,
        'tomp_tomp50':20*1e3,
        'tomp_JieChu_tomp50_esot500':20*1e3,#added by JieChu
        'tomp_tomp101':24*1e3,#added by JieChu, however uncertain...
        'tomp_JieChu_tomp101_esot500':24*1e3,#added by JieChu, however uncertain...
        'ostrack_vitb_256_mae_ce_32x4_ep300':16*1e3,
        'ostrack_trial4_vitb_256_mae_ce_32x4_aligned':16*1e3,
        'ostrack_trial6_ostrack256_aug1':16*1e3,
        'ostrack_trial8_ostrack256':16*1e3,
        'ostrack_trial9_ostrack256':16*1e3,
        'ostrack_baseline':16*1e3,
        'ostrack_aug':16*1e3,
        'stark_s_baseline':16*1e3,
                          }
    
    params.sim_runtime_init = {
        'atom_default': 273*1e3,
        'atom_fe240':   247*1e3,
        'atom_esot500':   247*1e3,#added by JieChu, uncertain however...
        'dimp_dimp18':  56*1e3,
        'dimp_dimp18_fe240':55*1e3,
        'dimp_dimp18_esot500':55*1e3,#added by JieChu, uncertain however...
        'dimp_prdimp18':55*1e3,
        'dimp_prdimp18_fe240':55*1e3,
        'dimp_prdimp18_esot500':55*1e3,#added by JieChu, uncertain however...
        'dimp_prdimp50':100*1e3,#added by JieChu, however uncertain...
        'dimp_JieChu_prdimp50_esot500':100*1e3,#added by JieChu, however uncertain...
        'dimp_dimp50':  80*1e3,
        'dimp_JieChu_dimp50_esot500':  80*1e3,#added by JieChu
        'keep_track_default':495*1e3,
        'kys_default':81*1e3,
        'kys_fe240':80*1e3,
        'kys_esot500':80*1e3,#added by JieChu
        'rts_rts50':136*1e3,
        'tomp_tomp50':7*1e3,
        'tomp_JieChu_tomp50_esot500':7*1e3,#added by JieChu
        'tomp_tomp101':14*1e3,#added by JieChu, however uncertain...
        'tomp_JieChu_tomp101_esot500':14*1e3,#added by JieChu, however uncertain...
        'ostrack_vitb_256_mae_ce_32x4_ep300':10*1e3,
        'ostrack_trial4_vitb_256_mae_ce_32x4_aligned':10*1e3,
        'ostrack_trial6_ostrack256_aug1':10*1e3,
        'ostrack_trial8_ostrack256':10*1e3,
        'ostrack_trial9_ostrack256':10*1e3,
        'ostrack_baseline':10*1e3,
        'ostrack_aug':10*1e3,
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
    params.slicing = 'FxTime' # FxTime, FxNum, Adaptive, Last
    params.window_size = 20*1e3 # unit:us
    params.window_size_template = 20*1e3 # unit:us
    params.num_events = 5000
    params.adaptive_ = '' # Adaptive method, e.g. sampling

    # Representation
    params.representation = 'VoxelGridComplex'

    return params
