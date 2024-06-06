import importlib

def load_stream_setting(stream_setting):
    """Get stream_setting."""
    param_module = importlib.import_module('pytracking.stream_settings.{}'.format(stream_setting))
    params = param_module.parameters()
    return params