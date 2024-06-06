from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone, NetWrapper

def parameters():
    params = TrackerParams()
    params.id = 10
    """Standard Setting Part"""
    # Hardware
    params.convert_time = False
    params.sim = False
    params.sim_runtime_rt = 33*1e3 # unit:us
    params.sim_runtime = {}
    params.sim_disturb = 0.0 # disturbance around runtime

    # Template
    params.template_ = 'egt' # default means using the first slice according 
    # params.template_ = 'augmented'
    # params.template_ = 'seperate'
    params.init_time = True # whether to consider initialization latency

    '''User-define Part'''
    # Slicing
    params.slicing = 'FxTime' # FxTime, FxNum, Adaptive, Last
    params.slicing_ = 'egt' 
    params.window_size = 50*1e3 # unit:us
    params.window_size_template = 50*1e3 # unit:us
    params.num_events = 5000
    params.adaptive_ = '' # Adaptive method, e.g. sampling

    # Representation
    params.representation = 'Raw'

    return params
