import os
import sys
import argparse
import importlib

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker, get_dataset, trackerlist
from pytracking.evaluation.running import run_dataset, run_dataset_stream
from pytracking.evaluation import Tracker

def load_stream_setting(stream_setting):
    """Get stream_setting."""
    param_module = importlib.import_module('pytracking.stream_settings.{}'.format(stream_setting))
    params = param_module.parameters()
    return params

def run_experiment_stream(experiment_module: str, experiment_name: str, debug=0, threads=0, run_id=None, visdom_info=None):
    """Run experiment.
    args:
        experiment_module: Name of experiment module in the experiments/ folder.
        experiment_name: Name of the experiment function.
        debug: Debug level.
        threads: Number of threads.
    """
    expr_module = importlib.import_module('pytracking.experiments.{}'.format(experiment_module))
    expr_func = getattr(expr_module, experiment_name)
    trackers, dataset, stream_setting = expr_func()
    # stream_setting = load_stream_setting(stream_setting) # dict

    print('Running:  {}  {}  with Streaming setting {}'.format(experiment_module, experiment_name, stream_setting.id))

    run_dataset_stream(dataset, trackers, stream_setting, debug, threads, visdom_info=visdom_info)


def main():
    parser = argparse.ArgumentParser(description='Run tracker.')
    parser.add_argument('experiment_module', type=str, help='Name of experiment module in the experiments/ folder.')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment function.')
    # parser.add_argument('stream_setting', type=str, help='Name of stream_setting file.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')

    args = parser.parse_args()
    
    run_experiment_stream(args.experiment_module, args.experiment_name, args.debug, args.threads)


if __name__ == '__main__':
    main()
