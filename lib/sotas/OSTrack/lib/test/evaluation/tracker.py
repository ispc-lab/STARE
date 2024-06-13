import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, use_aas: bool = False, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.use_aas = use_aas
        self.display_name = display_name


        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            self.results_dir_rt = '{}/{}/{}'.format(env.results_path_rt, self.name, self.parameter_name)
            self.results_dir_rt_final = '{}/{}/{}'.format(env.results_path_rt_final, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}'.format(env.segmentation_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.results_dir_rt = '{}/{}/{}_{:03d}'.format(env.results_path_rt, self.name, self.parameter_name,self.run_id)
            self.results_dir_rt_final = '{}/{}/{}'.format(env.results_path_rt_final, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None
        self.visdom = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        tracker.visdom = self.visdom
        return tracker

    def run_sequence(self, seq, stream_setting=None, debug=None, visualization=None, visdom_info=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()
        visualization_ = visualization

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_
        if visualization is None:
            if debug is None:
                visualization_ = getattr(params, 'visualization', False)
            else:
                visualization_ = True if debug else False

        params.visualization = visualization_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)
        if seq.dataset in ['esot500s','esot2s']:
            output = self._track_evstream(tracker, seq, init_info, stream_setting)
        else:
            output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0]) # NOTE: For temporary test

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output
    
    def _track_evstream(self, tracker, seq, init_info, stream_setting, **kwargs):
        """
        Core Function
        stream_setting:
            Option: Fixed time window / #events / adaptive / model-based
            Option: Template Initialization Policy
            Option: Simulated / wall-clock runtime / #event-dependent
            Option: Event representation
            Option: Embedding latency counted / ignored
        """
        from pytracking.utils.visdom import Visdom
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from pytracking.utils.plotting import draw_figure, overlay_mask
        from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
        from ltr.data.bounding_box_utils import masks_to_bboxes
        from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
        from pathlib import Path
        import torch
        from time import perf_counter
        from pytracking.utils.search import interpolation_search
        from pytracking.utils.convert_event_img import convert_event_img_aedat
        from dv import AedatFile
        import torchvision.transforms as T
        import cv2 as cv2
        transform = T.ToPILImage()

        aeFile = seq.events
        with AedatFile(aeFile) as f:
            print('Processing:', aeFile)
            events = np.hstack([packet for packet in f['events'].numpy()])
            events['timestamp'] = events['timestamp'] - events['timestamp'][0]

        timestamps = events['timestamp']
        #TODO: adjust timestamps to seconds

        eval_setting = {}

        #init
        pred_bboxes = []
        in_timestamps = []
        runtime = []
        out_timestamps = []
        t_stream_total = timestamps[-1]

#########################################################################################
        active_state = {
            'out': None,
            'bbox': [],
            'density': 0.,
            'is_last_activated': False,
            'idx_start_last': 0,
            'idx_start_next': 0,
            'length': 0,
        }

        cnt = 0

        def generate_density(bbox, img):
            res = 0
            l, t, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # print(l, t, w, h)
            for i in range(h):
                for j in range(w):
                    res += int(img[t + i][l + j][0] > 0 | img[t + i][l + j][1] > 0 | img[t + i][l + j][2] > 0)

            return res
#########################################################################################

        if not stream_setting.convert_time:
            # Initialize
            # t_start = perf_counter()*1e6
            t_left = 0
            idx_start = 0

            # ============== Initialization ====================
            if stream_setting.template_ == 'default':
                if stream_setting.slicing == 'FxTime':
                    t_template = stream_setting.window_size
                    idx_end = interpolation_search(timestamps,t_template)
                elif stream_setting.slicing == 'FxNum':
                    idx_end = stream_setting.num_events
                    t_template = timestamps[idx_end]
                elif stream_setting.slicing == 'Last':
                    t_template = init_info.get('init_timestamp')*1e6
                    idx_end = interpolation_search(timestamps,t_template)
                elif stream_setting.slicing == 'Adaptive':
                    from pytracking.utils.sampling import EventStreamSampler
                    stream_sampler = EventStreamSampler()
                    idx_end, t_template = stream_sampler.init_with_template(events, init_info, stream_setting)
                    # t_template = stream_setting.window_size_template # TODO: use first timestamp
                    # idx_end = interpolation_search(timestamps,t_template)
                template_events = events[idx_start:idx_end]

            elif stream_setting.template_ == 'seperate':
                    t_template = stream_setting.window_size_template
                    idx_end = interpolation_search(timestamps,t_template)
                    template_events = events[idx_start:idx_end]

            elif stream_setting.template_ == 'egt':
                    from pytracking.utils.sampling import sampling_template_egt, sampling_search_egt
                    t_template = init_info.get('init_timestamp') * 1e6
                    idx_end = interpolation_search(timestamps,t_template)
                    template_events_raw = events[idx_start:idx_end]
                    template_events = sampling_template_egt(events, init_info)
            else:
                raise NotImplementedError

            event_rep = convert_event_img_aedat(template_events, stream_setting.representation)
            # event_img_pil = transform(event_rep)
            # event_img_array = np.array(event_img_pil)
            # event_img = cv2.cvtColor(event_img_array,cv2.COLOR_RGB2BGR)
            if tracker.params.visualization and self.visdom is None:
                if stream_setting.representation in ['VoxelGridComplex']:
                    event_img = event_rep
                    self.visualize(event_img, init_info.get('init_bbox'))
                elif stream_setting.representation in ['Raw']:
                    event_img = convert_event_img_aedat(template_events_raw,'VoxelGridComplex')
                    self.visualize(event_img, init_info.get('init_bbox'))

            torch.cuda.synchronize()

            t1 = t_start = perf_counter() * 1e6
            out = tracker.initialize(event_rep, init_info)
            if out is None:
                out = {}
            prev_output = OrderedDict(out)
            pred_bbox = init_info.get('init_bbox')
            pred_bboxes.append(pred_bbox)
            torch.cuda.synchronize()
            t2 = perf_counter() * 1e6
            t_algo_init = t2 - t1
            t_algo_init = out.get('time') * 1e6 if out.get('time') else t_algo_init
            # t_algo = out.get('time',t_algo)
            if stream_setting.sim:
                sim_runtime = stream_setting.sim_runtime_init.get(self.name+'_'+self.parameter_name,stream_setting.sim_runtime_rt)
                sim_disturb = sim_runtime * stream_setting.sim_disturb
                t_algo_init = np.random.normal(loc=sim_runtime, scale=sim_disturb, size=1)[0]
            if stream_setting.init_time == False:
                sim_runtime = stream_setting.sim_runtime.get(self.name+'_'+self.parameter_name,stream_setting.sim_runtime_rt)
                sim_disturb = sim_runtime * stream_setting.sim_disturb
                t_algo_init = np.random.normal(loc=sim_runtime, scale=sim_disturb, size=1)[0]
            # print(t_algo/1e6)
            runtime.append(t_algo_init)
            in_timestamps.append(t_template)
            out_timestamps.append(t_algo_init + t_template)

            ##############################################################################################
            active_state['out'] = prev_output
            active_state['bbox'] = pred_bbox
            active_state['density'] = generate_density(active_state['bbox'], event_rep)
            active_state['is_last_activated'] = True
            active_state['idx_start_last'] = idx_start
            ###############################################################################################

            # =================== Tracking =====================
            while 1:
                t_algo_total = sum(runtime) + t_template
                if t_algo_total > t_stream_total:
                    break

                t_right = out_timestamps[-1] # Current world-time
                idx_end = interpolation_search(timestamps, t_right)

                if stream_setting.slicing == 'FxTime':
                    t_left = t_right - stream_setting.window_size
                    t_left = t_left if t_left >= timestamps[0] else timestamps[0]
                    idx_start = interpolation_search(timestamps, t_left)
                elif stream_setting.slicing == 'FxNum':
                    idx_start = idx_end - stream_setting.num_events
                    idx_start = max(idx_start, 0)
                elif stream_setting.slicing in ['Last','egt']:
                    t_left = in_timestamps[-1]
                    idx_start = interpolation_search(timestamps, t_left)
                elif stream_setting.slicing == 'Adaptive':
                    idx_start, t_left = stream_sampler.sample(events[:idx_end], pred_bboxes, stream_setting)
                    # sampling = load_sampling_func(stream_setting.adaptive_)
                    # idx_start, t_left  = sampling(events[:idx_end], pred_bboxes, stream_setting)

                ###################################################################################################
                if active_state['is_last_activated']:
                    active_state['idx_start_next'] = idx_start
                    events_search = events[idx_start:idx_end]
                else:
                    events_search = events[active_state['idx_start_next']:idx_end]
                ###################################################################################################

                slicing_ = stream_setting.get('slicing_', None)
                if slicing_ and slicing_ in ['egt']:
                    # convert format for egt
                    event_rep = sampling_search_egt(events_search)
                else:
                    event_rep = convert_event_img_aedat(events_search, stream_setting.representation)
                info = {} # changed

                ##################################################################################################
                if active_state['is_last_activated']:
                    info['previous_output'] = prev_output
                else:
                    info['previous_output'] = active_state['out']

                density_tmp = generate_density(active_state['bbox'], event_rep)
                if density_tmp < active_state['density'] * 0.5 and active_state['length'] <= 50 * 1e3:
                    active_state['is_last_activated'] = False
                elif density_tmp < active_state['density'] * 0.05 and active_state['length'] > 50 * 1e3:
                    active_state['is_last_activated'] = False
                else:
                    active_state['is_last_activated'] = True
                ##################################################################################################

                # event_img_pil = transform(event_rep)
                # event_img_array = np.array(event_img_pil)
                # event_img = cv2.cvtColor(event_img_array,cv2.COLOR_RGB2BGR)
                # cv2.imwrite('debug/test2.jpg',event_img)
                # event_img.save('debug/test.jpg')
                t1 = perf_counter() * 1e6
                out = tracker.track(event_rep, info)

                prev_output = OrderedDict(out)
                torch.cuda.synchronize()
                t2 = perf_counter() * 1e6
                t_algo = t2 - t1
                # t_algo = out.get('time',t_algo)
                t_algo = out.get('time') * 1e6 if out.get('time') else t_algo
                if stream_setting.sim:
                    sim_runtime = stream_setting.sim_runtime.get(self.name+'_'+self.parameter_name,stream_setting.sim_runtime_rt)
                    sim_disturb = sim_runtime * stream_setting.sim_disturb
                    t_algo = np.random.normal(loc=sim_runtime, scale=sim_disturb, size=1)[0]
                runtime.append(t_algo)
                # print(t_algo/1e6)
                in_timestamps.append(out_timestamps[-1])
                out_timestamps.append(out_timestamps[-1] + t_algo)

                ################################################################################
                if not self.use_aas:
                    active_state['is_last_activated'] = True

                # pred_bbox = out['target_bbox']
                # active_state['bbox'] = out['target_bbox']
                # active_state['density'] = generate_density(out['target_bbox'], event_rep)

                if active_state['is_last_activated']:
                    pred_bbox = out['target_bbox']
                    active_state['out'] = prev_output
                    active_state['bbox'] = out['target_bbox']
                    active_state['density'] = generate_density(out['target_bbox'], event_rep)
                    active_state['length'] = 0
                else:
                    pred_bbox = active_state['bbox']
                    active_state['length'] += t_algo
                    cnt = cnt + 1
                # print('target_bbox:', pred_bbox)

                pred_bboxes.append(pred_bbox)  # box [x1, y1, w, h]
                ################################################################################

                bboxes = [out['target_bbox']]
                if 'clf_target_bbox' in out:
                    bboxes.append(out['clf_target_bbox'])
                if 'clf_search_area' in out:
                    bboxes.append(out['clf_search_area'])
                if 'segm_search_area' in out:
                    bboxes.append(out['segm_search_area'])
                segmentation = None

                if stream_setting.representation in ['VoxelGridComplex']:
                    event_img = event_rep
                elif stream_setting.representation =='Raw':
                    event_img = convert_event_img_aedat(events_search, 'VoxelGridComplex')

                if self.visdom is not None:
                    tracker.visdom_draw_tracking(event_img, bboxes, segmentation)
                elif tracker.params.visualization:
                    self.visualize(event_img, bboxes, segmentation)
                    
            output={
                        'results_raw': pred_bboxes,
                        'out_timestamps': out_timestamps,
                        'in_timestamps': in_timestamps,
                        'runtime': runtime,
                        'stream_setting':stream_setting.id,
                    }

        else:
            # TODO: under-construction
            # Absolute wall time, including conversion time
            t_start = perf_counter()*1e6
            t_left = 0
            t_right = stream_setting.window_size
            t1 = perf_counter()*1e6
            # idx_start = interpolation_search(timestamps,t_left)
            idx_start = 0
            idx_end = interpolation_search(timestamps,t_right) # merely zero
            event_rep = convert_event_img_aedat(events[idx_start:idx_end],'VoxelGridComplex')
            # event_img = transform(event_rep)
            event_img_pil = transform(event_rep)
            event_img_array = np.array(event_img_pil)
            event_img = cv2.cvtColor(event_img_array,cv2.COLOR_RGB2BGR)
            if tracker.params.visualization and self.visdom is None:
                self.visualize(event_img, init_info.get('init_bbox'))

            out = tracker.initialize(event_img, init_info)
            if out is None:
                out = {}
            prev_output = OrderedDict(out)
            torch.cuda.synchronize()
            t2 = perf_counter()*1e6
            t_elapsed=t2-t_start
            pred_bbox = init_info.get('init_bbox')
            pred_bboxes.append(pred_bbox)
            # input_fidx.append(fidx)
            in_timestamps.append(stream_setting.window_size)
            out_timestamps.append(t_elapsed+stream_setting.window_size)
            runtime.append(t2-t1)
            while 1:
                if stream_setting.convert_time:
                    t1 = perf_counter()*1e6
                    t_elapsed=t1-t_start
                else:
                    t_elapsed = sum(runtime)
                if t_elapsed>t_stream_total:
                    break
                t_right = t_elapsed
                t_left = t_elapsed - stream_setting.window_size
                idx_start = interpolation_search(timestamps,t_left)
                idx_end = interpolation_search(timestamps,t_right)
                event_rep = convert_event_img_aedat(events[idx_start:idx_end],'VoxelGridComplex')
                info = {} # changed
                info['previous_output'] = prev_output
                event_img_pil = transform(event_rep)
                event_img_array = np.array(event_img_pil)
                event_img = cv2.cvtColor(event_img_array,cv2.COLOR_RGB2BGR)
                if not stream_setting.convert_time:
                    t1 = perf_counter()*1e6
                out = tracker.track(event_img, info)
                prev_output = OrderedDict(out)
                torch.cuda.synchronize()
                t2 = perf_counter()*1e6
                t_elapsed=t2-t_start
                in_timestamps.append(t_right)
                out_timestamps.append(t_elapsed)
                runtime.append(t2-t1)
                pred_bbox = out['target_bbox']
                pred_bboxes.append(pred_bbox)
                # input_fidx.append(fidx)
                if t_elapsed>t_stream_total:
                    break
                bboxes = out['target_bbox']
                segmentation = None
                if self.visdom is not None:
                    tracker.visdom_draw_tracking(event_img, bboxes, segmentation)
                elif tracker.params.visualization:
                    self.visualize(event_img, bboxes, segmentation)
            output={
                        'results_raw': pred_bboxes,
                        'out_timestamps': out_timestamps,
                        'in_timestamps': in_timestamps,
                        'runtime': runtime,
                    }

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_video_demo(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        success, frame = cap.read()

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)
        cap.release()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}_ostrack'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



