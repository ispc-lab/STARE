from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone, NetWrapper

def parameters():
    params = TrackerParams()
    params.id = 1

    # Hardware
    params.convert_time = False
    params.sim = True
    params.sim_runtime = 10*1e3 # unit:us
    params.sim_disturb = .1 # disturbance around runtime

    # Slicing
    params.slicing = 'FxTime' # FxTime, FxNum, Adaptive
    params.window_size = 20*1e3 # unit:us
    params.num_events = 1e4
    params.adaptive_ = '' # Adaptive method, e.g. sampling

    # Template
    params.template_ = 'default' # default means using the first slice according 
    # params.template_ = 'augmented'
    # params.template_ = 'sparse'
    params.init_time = True # consider initialization latency


    # Representation
    params.representation = 'VoxelGridComplex'

    return params
