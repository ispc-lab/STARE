from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone, NetWrapper

def parameters():
    """Difference: representation """
    params = TrackerParams()
    params.id = 6
    """Standard Setting Part"""
    # Hardware
    params.convert_time = False
    params.sim = True
    params.sim_runtime_rt = 33*1e3 # unit:us 
    params.sim_runtime = {
        'atom_default':0.025*1e6,
        'atom_fe240':0.025*1e6,
        'dimp_dimp18':0.02*1e6,
        'dimp_dimp18_fe240':0.02*1e6,
        'dimp_prdimp18':0.027*1e6,
        'dimp_prdimp18_fe240':0.027*1e6,
        'dimp_dimp50':0.033*1e6,
        'keep_track_default':0.043*1e6,
        'kys_default':0.04*1e6,
        'kys_fe240':0.04*1e6,
        'rts_rts50':0.033*1e6,
        'tomp_tomp50':0.035*1e6,
                          }
    params.sim_disturb = 0.0 # disturbance around runtime

    # Template
    params.template_ = 'default' # default means using the first slice according 
    # params.template_ = 'augmented'
    # params.template_ = 'seperate'
    params.init_time = False # whether to consider initialization latency

    '''User-define Part'''
    # Slicing
    params.slicing = 'FxTime' # FxTime, FxNum, Adaptive, Last
    params.window_size = 20*1e3 # unit:us
    params.window_size_template = 20*1e3 # unit:us
    params.num_events = 5000
    params.adaptive_ = '' # Adaptive method, e.g. sampling
    
    # Representation
    params.representation = 'FE240'

    return params
