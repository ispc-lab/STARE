import os
import sys

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('./')

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results, print_results_per_video
from lib.test.evaluation import get_dataset, trackerlist

dataset_name = 'esot500s'

trackers = []
# trackers.extend(trackerlist(name='ostrack', parameter_name='esot500_baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='esot500_baseline'))

trackers.extend(trackerlist(name='ostrack', parameter_name='pred_esot500_4step', dataset_name=dataset_name,
                            run_ids=None, display_name='pred_esot500_4step'))

# trackers.extend(trackerlist(name='mixformer_convmae', parameter_name='pred_esot500', dataset_name=dataset_name,
#                             run_ids=None, display_name='pred_esot500'))

dataset = get_dataset(dataset_name)

# plot_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'prec', 'norm_prec'))
print_results_per_video(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'prec', 'norm_prec'),
                        per_video=True, force_evaluation=True)
# print_per_sequence_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'prec', 'norm_prec'))
