import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['figure.figsize'] = [14, 8]

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)

# from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
# from pytracking.evaluation import get_dataset, trackerlist

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

# Frame-based evaluation

dataset_esoth = [
    "esoth_500_2", "esoth_250_2", "esoth_20_2",
    "esoth_500_8", "esoth_250_8", "esoth_20_8",
    "esoth_500_20", "esoth_250_20", "esoth_20_20",
    "esoth_500_50", "esoth_250_50", "esoth_20_50",
]

trackers = []
trackers.extend(trackerlist(name='stark_s',
                            parameter_name='baseline',
                            dataset_name=dataset_esoth,
                            run_ids=None,
                            display_name='stark_s'))

for dataset_name in dataset_esoth:
    dataset = get_dataset(dataset_name)
    print_results(trackers, dataset, dataset_name, plot_types=('success', 'prec', 'norm_prec'))

# stare

# dataset_name = "esot500hs"
#
# stream_settings = [101, 107, 102, 103]
#
# for stream_setting_id in stream_settings:
#     print("Processing stream setting:", stream_setting_id)
#     dataset = get_dataset(dataset_name)
#     print_results(
#         trackers_stare_xijing,
#         dataset,
#         dataset_name,
#         plot_types=('success', 'prec', 'norm_prec'),
#         stream_id=stream_setting_id,
#     )


# stream_setting_id = 102
#
# print("Processing stream setting:", stream_setting_id)
# dataset = get_dataset(dataset_name)
# print_results(
#     trackers_autodl,
#     dataset,
#     dataset_name,
#     plot_types=('success', 'prec', 'norm_prec'),
#     stream_id=stream_setting_id,
# )



