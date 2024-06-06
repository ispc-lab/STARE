
import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('/home/test4/code/EventBenchmark/lib/pytracking')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results, plot_results_mod
from pytracking.evaluation import get_dataset, trackerlist

# Realtime
# dataset_name = 'esot2_5_20'
# trackers = []
# trackers.extend(trackerlist(name='atom', parameter_name='default',
#                             run_ids=0, display_name='atom'))
# trackers.extend(trackerlist(name='dimp', parameter_name='dimp18', 
#                             run_ids=0, display_name='dimp18'))
# dataset = get_dataset(dataset_name)

trackers = []
run_id = range(3)
trackers.extend(trackerlist(name='atom', parameter_name='default',
                            run_ids=run_id, display_name='ATOM'))
trackers.extend(trackerlist(name='atom', parameter_name='fe240',
                            run_ids=run_id, display_name='ATOM*'))
trackers.extend(trackerlist(name='dimp', parameter_name='dimp18', 
                            run_ids=run_id, display_name='DiMP18'))
trackers.extend(trackerlist(name='dimp', parameter_name='dimp18_fe240', 
                            run_ids=run_id, display_name='DiMP18*'))
trackers.extend(trackerlist(name='dimp', parameter_name='prdimp18', 
                            run_ids=run_id, display_name='PrDiMP18'))
trackers.extend(trackerlist(name='dimp', parameter_name='dimp50', 
                            run_ids=run_id, display_name='DiMP50'))
trackers.extend(trackerlist(name='kys', parameter_name='default', 
                            run_ids=run_id, display_name='KYS'))
trackers.extend(trackerlist(name='kys', parameter_name='fe240', 
                            run_ids=run_id, display_name='KYS*'))
trackers.extend(trackerlist(name='rts', parameter_name='rts50', 
                            run_ids=run_id, display_name='RTS50'))
trackers.extend(trackerlist(name='keep_track', parameter_name='default', 
                            run_ids=run_id, display_name='KeepTrack'))
trackers.extend(trackerlist(name='tomp', parameter_name='tomp50', 
                            run_ids=run_id, display_name='ToMP50'))
trackers.extend(trackerlist(name='ostrack', parameter_name='baseline', 
                            run_ids=run_id, display_name='OSTrack'))
trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', 
                            run_ids=run_id, display_name='STARK_s'))
trackers.extend(trackerlist(name='mixformer_convmae_online', parameter_name='baseline', 
                            run_ids=run_id, display_name='MixFormer'))

stream_setting_id = 18
dataset_name = 'esot2s'
dataset = get_dataset(dataset_name)
# plot_results_mod(trackers, dataset, dataset_name+'_'+str(stream_setting_id),plot_types=('success'),merge_results=True,force_evaluation=False, stream_id=stream_setting_id)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),force_evaluation=True, stream_id=stream_setting_id)
