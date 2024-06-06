# From Frame to Stream: Rethinking Visual Object Tracking Based on Event Streams
<!-- We revisit the frame-based evaluation on event streams and analyze its sensitivity to preprocessing and consistency issues.
We then propose the **stream-based evaluation** framework with a unified protocol that allows direct evaluation on raw event streams. The evaluation progress is time-dependent rather than frame-sequential, and the performance metrics are calculated in a latency-aware manner, taking runtime latency into account. -->

## Abstraction
Visual Object Tracking (VOT) in event-based vision is attracting increasing interest. Existing research typically pre-slices continuous event streams into discrete frames for tracking and evaluation, focusing primarily on offline accuracy in fixed-rate frame sequences. However, this approach often overlooks the critical impact of inference latency on tracker performance, especially in real-world scenarios where downstream applications will ceaselessly request the tracker for current object position without waiting any latency. To address these limitations, we introduce a paradigm shift in evaluation methodology with our novel STream-based lAtency-awaRe Evaluation (STARE) framework. STARE not only precisely reveals a tracker's realistic performance by simulating consecutive downstream requests, but also leverages the tracker's real-time capabilities by scheduling it to automatically sample events as inputs. We complement STARE with ESOT500, a new tracking dataset featuring time-aligned, high-frequency annotations, serving as a comprehensive platform for evaluating trackers, emphasizing the importance of both accuracy and latency. Subsequent experiments reveal a notable performance decline under STARE's stringent real-time criteria, highlighting the necessity to optimize trackers' robustness. In response, we further propose two simple yet effective tracker enhancement methods: Predictive Tracking and Adaptive Sampling Strategy, both distinguished by their utilization of the event stream modality's characteristics.


<!-- TABLE OF CONTENTS -->
<!-- <details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'> -->
  <!-- <summary>Table of Contents</summary> -->
  <ol>
    <!-- <li>
      <a href="#introduction">Introduction</a>
    </li> -->
    <!-- <li>
      <a href="#dataset">Dataset</a>
    </li> -->
    <!-- <li>
      <a href="#benchmark">Benchmark</a>
    </li> -->
    <!-- <li>
      <a href="#citation">Citation</a>
    </li> -->
    <!-- <li>
      <a href="#license">License</a>
    </li> -->
  </ol>
<!-- </details> -->

<!-- ## Introduction
We propose a novel evaluation framework for EVOT, named stream-based evaluation, which reformulates object tracking on event streams as a dynamic process both in spatial and temporal dimensions, highlighting the importance of latency. 
Technically, we start by analyzing the limitation of existing frame-based evaluation regarding the sensitivity to event preprocess and consistency issues. 
Drawing on the analysis, we introduce latency awareness into the evaluation and propose a unified framework for benchmarking on event streams.

The stream-based evaluation framework operates on raw event streams in a **streaming manner** in which the evaluation time and the data time are aligned. 
The tracker is viewed as a program running in a loop, whose runtime is accumulated as the elapsed time of both evaluation and event data. 
At each iteration, the tracker samples and process the currently received events and produces a prediction with a corresponding timestamp. 
This iterative process persists until the completion of the event stream. 
Subsequently, the evaluation results are calculated by querying the most recent prediction for each timestamped ground truth, enabling comprehensive assessment of the tracker's performance in a latency-aware manner. -->

<!-- <img src="img/concept1.png" width=65%> -->

## Dataset
We present ESOT500, a new dataset for event-based VOT, featuring time-aligned and high-frequency annotations, designed to support STARE’s stringent real-time criteria.
<!-- The dataset consists of a high-frequency annotated subset **EventSOT-H** and a more challenging subset **EventSOT-C** labeled at normal frequency, both time-aligned. -->

- Download **EventSOT-H** from [[OneDrive]](https://tongjieducn-my.sharepoint.com/:f:/g/personal/2131522_tongji_edu_cn/Eis8Aq1_rSxOgNbeQfzOSaQBKs7R5A1hAPHjLO7pTXzbcg?e=oeMNfE)

- The aedat4 directory contains the raw event data (event stream and corresponding RGB frames), the [DV](https://inivation.gitlab.io/dv/dv-docs/docs/getting-started.html) and [dv-python](https://gitlab.com/inivation/dv/dv-python) is recommended for visualization and processing in python respectively.

<!-- <img src="img/esot2_examples.png" width=65%> -->

<!-- <img src="img/comparison.png" width=65%> -->

## Benchmark
The key advantages of the proposed stream-based evaluation are three-fold:
1. A unified evaluation regardless of the adopted event representations;
2. Dynamic process depending on time rather than frame-sequential;
3. Comprehensive evaluation of trackers in terms of latency and accuracy;

Different from frame sequence, event streams are asynchronous data flows. 
As shown below, the major difference between stream-based evaluation and frame-based streaming perception is that there is **input at any time** instead of at certain moments.

<!-- <img src="img/evaluations.png" width=65%> -->

<!-- <img src="img/algorithm.png" width=65%> -->

## Usage
The code is based on the [**PyTracking**](https://github.com/visionml/pytracking) and other frameworks.

For detailed installation and configuration please refer to [lib/pytracking/INSTALL.md](lib/pytracking/INSTALL.md) 

```
# first go to pytracking
cd lib/pytracking

# preprare dataset
ln -s /PATH/TO/EventSOT-H ./data/EventSOT500

# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

# modify the dataset path in 
lib/pytracking/ltr/admin/local.py # paths about training
pytracking/evaluation/local.py  # paths about testing

bash install.sh conda_install_path STARE
conda activate STARE

# for frame-based evaluation
python pytracking/run_experiment.py myexperiments esot500_offline

# for stream-based evaluation, stream settings are in folder pytracking/stream_settings
python pytracking/run_experiment_streaming.py exp_streaming streaming_34
python eval/streaming_eval_v3.py exp_streaming streaming_34

# The results are in the folders './pytracking/output/tracking_results' and './pytracking/output/tracking_results_rt_final', then evaluate the results.
pytracking/analysis/stream_eval.ipynb

# For trackers not integrated into pytracking, see 'lib/sotas/[tracker]' for details. Their usage is similar.

# For tracker enhancement, see 'lib/sotas/pred_OSTrack' for details.
```


<!-- ## Citation -->

## License
  The released code and dataset are under [Apache 2.0 license](https://www.apache.org/licenses/LIC).

## Acknowledgments
- The benchmark is built on top of the great [PyTracking](https://github.com/visionml/pytracking) library 
- Thanks for the great works including [Stark](https://github.com/researchmm/Stark), [MixFormer](https://github.com/MCG-NJU/MixFormer), [OSTrack](https://github.com/botaoye/OSTrack) and [Event-tracking](https://github.com/ZHU-Zhiyu/Event-tracking)