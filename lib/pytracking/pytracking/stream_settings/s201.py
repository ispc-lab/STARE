from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone, NetWrapper

def parameters():
    params = TrackerParams()
    params.id = 201 # [500FPS, 2ms] for EGT frame-based evaluation

    """Standard Setting Part"""
    # Hardware
    params.convert_time = False
    params.sim = True

    params.sim_runtime_rt = 40 * 1e3  # unit:us, default run time

    params.sim_runtime = {
        'egt_egt': 2 * 1e3, # unit:us, for the control of FPS setting, now egt is [1000 / 2 = 500] FPS
    }

    params.sim_runtime_init = {
        'egt_egt': 0 * 1e3,
    }

    params.sim_disturb = 0.0  # disturbance around runtime

    # Template
    params.template_ = 'egt'  # default means using the first slice according
    # params.template_ = 'augmented'
    # params.template_ = 'seperate'

    params.init_time = False  # whether to consider initialization latency

    '''User-define Part'''
    # Slicing
    params.slicing = 'FxTime' # FxTime, FxNum, Adaptive, Last = egt
    params.slicing_ = 'egt' # convert the input representation to egt style

    params.window_size = 2 * 1e3  # unit:us, for the control of window size, now it's [2] ms

    params.window_size_template = 50 * 1e3  # unit:us, only used for .template_ = 'separate'

    params.num_events = 5000

    params.adaptive_ = ''  # Adaptive method, e.g. sampling

    # Representation
    params.representation = 'Raw' # make convert_event_img_aedat() return the raw events

    return params
