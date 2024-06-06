import importlib
import os
import numpy as np
from collections import OrderedDict
from pytracking.evaluation.environment import env_settings
import time
import cv2 as cv
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
import importlib
transform = T.ToPILImage()

_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}


def trackerlist(name: str, parameter_name: str, run_ids = None, display_name: str = None):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, run_id, display_name) for run_id in run_ids]

def load_sampling_func(func:str):
    expr_module = importlib.import_module('pytracking.utils.sampling')
    func  = getattr(expr_module, func)
    return func

class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, run_id: int = None, display_name: str = None):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
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
            self.results_dir_rt_final = '{}/{}/{}_{:03d}'.format(env.results_path_rt_final, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name, self.run_id)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', self.name))
        if os.path.isdir(tracker_module_abspath):
            tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.visdom = None


    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)

                # Show help
                help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                            'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                            'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                            'block list.'
                self.visdom.register(help_text, 'text', 1, 'Help')
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True


    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        tracker.visdom = self.visdom
        return tracker

    def run_sequence(self, seq, stream_setting=None, visualization=None, debug=None, visdom_info=None, multiobj_mode=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            visdom_info: Visdom info.
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()
        visualization_ = visualization

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        if visualization is None:
            if debug is None:
                visualization_ = getattr(params, 'visualization', False)
            else:
                visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)
        if visualization_ and self.visdom is None:
            self.init_visualization()

        # Get init information
        '''JieChu:
        This init_info() function is in evaluation/data.py,and as the param is None,
        it will return the bbox and timestamp of the 1st annotation.
        '''
        init_info = seq.init_info()
        is_single_object = not seq.multiobj_mode

        if multiobj_mode is None:
            multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default' or is_single_object:
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

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
        # segmentation[i] is the segmentation mask for frame i (numpy array)

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i
        # segmentation[i] is the multi-label segmentation mask for frame i (numpy array)

        output = {'target_bbox': [],
                  'time': [],
                  'segmentation': [],
                  'object_presence_score': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        if tracker.params.visualization and self.visdom is None:
            self.visualize(image, init_info.get('init_bbox'))

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'clf_target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'segmentation': init_info.get('init_mask'),
                        'object_presence_score': 1.}

        _store_outputs(out, init_default)

        segmentation = out['segmentation'] if 'segmentation' in out else None
        bboxes = [init_default['target_bbox']]
        if 'clf_target_bbox' in out:
            bboxes.append(out['clf_target_bbox'])
        if 'clf_search_area' in out:
            bboxes.append(out['clf_search_area'])
        if 'segm_search_area' in out:
            bboxes.append(out['segm_search_area'])

        if self.visdom is not None:
            tracker.visdom_draw_tracking(image, bboxes, segmentation)
        elif tracker.params.visualization:
            self.visualize(image, bboxes, segmentation)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                else:
                    time.sleep(0.1)

            #我记得就是在这一句报了错
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            segmentation = out['segmentation'] if 'segmentation' in out else None

            bboxes = [out['target_bbox']]
            if 'clf_target_bbox' in out:
                bboxes.append(out['clf_target_bbox'])
            if 'clf_search_area' in out:
                bboxes.append(out['clf_search_area'])
            if 'segm_search_area' in out:
                bboxes.append(out['segm_search_area'])

            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, bboxes, segmentation)
            elif tracker.params.visualization:
                self.visualize(image, bboxes, segmentation)

        for key in ['target_bbox', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        # next two lines are needed for oxuva output format.
        output['image_shape'] = image.shape[:2]
        output['object_presence_score_threshold'] = tracker.params.get('object_presence_score_threshold', 0.55)

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

        aeFile = seq.events
        with AedatFile(aeFile) as f:
            print('Processing:',aeFile)
            events = np.hstack([packet for packet in f['events'].numpy()])
            events['timestamp'] = events['timestamp'] -events['timestamp'][0]

        timestamps = events['timestamp']
        #TODO: adjust timestamps to seconds

        eval_setting = {}

        #init
        pred_bboxes = []
        in_timestamps = []
        runtime = []
        out_timestamps = []
        t_stream_total = timestamps[-1] 

        # Initialize
        # t_start = perf_counter()*1e6
        t_left = 0
        idx_start = 0
        t0 = perf_counter()*1e6
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
                '''JieChu:
                I can figure out how this slicing==Adaptive plays its role.
                Maybe it's another TODO.
                '''          
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
                t_template = init_info.get('init_timestamp')*1e6
                idx_end = interpolation_search(timestamps,t_template)
                template_events_raw = events[idx_start:idx_end]
                template_events = sampling_template_egt(events, init_info)
        else:
            raise NotImplementedError
        event_rep = convert_event_img_aedat(template_events,stream_setting.representation)
        '''JieChu:
        t_convert record the couvert time of the template frame.
        '''
        t_convert = perf_counter()*1e6 - t0
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
        t1 = t_start = perf_counter()*1e6
        out = tracker.initialize(event_rep, init_info)
        if out is None:
            out = {}
        prev_output = OrderedDict(out)
        pred_bbox = init_info.get('init_bbox')
        pred_bboxes.append(pred_bbox)
        torch.cuda.synchronize()
        t2 = perf_counter()*1e6
        t_algo_init=t2-t1
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
        if stream_setting.convert_time:
            t_algo_init += t_convert
        runtime.append(t_algo_init)
        in_timestamps.append(t_template)
        out_timestamps.append(t_algo_init+t_template)
        
        # =================== Tracking =====================
        while 1:
            t_algo_total = sum(runtime) + t_template
            if t_algo_total>t_stream_total:
                break
            t0 = perf_counter()*1e6
            t_right = out_timestamps[-1] # Current world-time
            idx_end = interpolation_search(timestamps,t_right)
            if stream_setting.slicing == 'FxTime':
                t_left = t_right - stream_setting.window_size
                t_left = t_left if t_left >= timestamps[0] else timestamps[0]
                idx_start = interpolation_search(timestamps,t_left)
            elif stream_setting.slicing == 'FxNum':
                idx_start = idx_end - stream_setting.num_events
                idx_start = max(idx_start, 0)
            elif stream_setting.slicing in ['Last','egt']:
                t_left = in_timestamps[-1]
                idx_start = interpolation_search(timestamps,t_left)
            elif stream_setting.slicing == 'Adaptive':
                idx_start, t_left = stream_sampler.sample(events[:idx_end], pred_bboxes, stream_setting)
                # sampling = load_sampling_func(stream_setting.adaptive_)
                # idx_start, t_left  = sampling(events[:idx_end], pred_bboxes, stream_setting)
            events_search = events[idx_start:idx_end]
            slicing_ = stream_setting.get('slicing_', None)
            if slicing_ and slicing_ in ['egt']:
                # convert format for egt
                event_rep = sampling_search_egt(events_search)
            else:
                event_rep = convert_event_img_aedat(events_search,stream_setting.representation)
            info = {} # changed
            info['previous_output'] = prev_output

            # event_img_pil = transform(event_rep)
            # event_img_array = np.array(event_img_pil)
            # event_img = cv2.cvtColor(event_img_array,cv2.COLOR_RGB2BGR)
            # cv2.imwrite('debug/test2.jpg',event_img)
            # event_img.save('debug/test.jpg')
            t1 = perf_counter()*1e6
            t_convert = t1-t0
            out = tracker.track(event_rep, info)

            prev_output = OrderedDict(out)
            torch.cuda.synchronize()
            t2 = perf_counter()*1e6
            t_algo=t2-t1
            # t_algo = out.get('time',t_algo)
            t_algo = out.get('time') * 1e6 if out.get('time') else t_algo
            if stream_setting.sim:
                sim_runtime = stream_setting.sim_runtime.get(self.name+'_'+self.parameter_name,stream_setting.sim_runtime_rt)
                sim_disturb = sim_runtime * stream_setting.sim_disturb
                t_algo = np.random.normal(loc=sim_runtime, scale=sim_disturb, size=1)[0]
            # print(t_algo/1e6)
            if stream_setting.convert_time:
                t_algo += t_convert
            runtime.append(t_algo)
            in_timestamps.append(out_timestamps[-1])
            out_timestamps.append(out_timestamps[-1]+t_algo)
            pred_bbox = out['target_bbox']
            # print('target_bbox:', pred_bbox)
            pred_bboxes.append(pred_bbox) # box [x1, y1, w, h]

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
        return output

    def run_video_generic(self, debug=None, visdom_info=None, videofilepath=None, optional_box=None, save_results=False):
        """Run the tracker with the webcam or a provided video file.
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

        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                tracker.initialize_features()
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.new_init = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'init'
                    self.new_init = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()

        display_name = 'Display: ' + self.name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        frame_number = 0

        if videofilepath is not None:
            assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
            ", videofilepath must be a valid videofile"
            cap = cv.VideoCapture(videofilepath)
            ret, frame = cap.read()
            frame_number += 1
            cv.imshow(display_name, frame)
        else:
            cap = cv.VideoCapture(0)


        next_object_id = 1
        sequence_object_ids = []
        prev_output = OrderedDict()
        output_boxes = OrderedDict()

        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's format is [x,y,w,h]"

            out = tracker.initialize(frame, {'init_bbox': OrderedDict({next_object_id: optional_box}),
                                       'init_object_ids': [next_object_id, ],
                                       'object_ids': [next_object_id, ],
                                       'sequence_object_ids': [next_object_id, ]})

            prev_output = OrderedDict(out)

            output_boxes[next_object_id] = [optional_box, ]
            sequence_object_ids.append(next_object_id)
            next_object_id += 1

        # Wait for initial bounding box if video!
        paused = videofilepath is not None

        while True:

            if not paused:
                # Capture frame-by-frame
                ret, frame = cap.read()
                frame_number += 1
                if frame is None:
                    break

            frame_disp = frame.copy()

            info = OrderedDict()
            info['previous_output'] = prev_output

            if ui_control.new_init:
                ui_control.new_init = False
                init_state = ui_control.get_bb()

                info['init_object_ids'] = [next_object_id, ]
                info['init_bbox'] = OrderedDict({next_object_id: init_state})
                sequence_object_ids.append(next_object_id)
                if save_results:
                    output_boxes[next_object_id] = [init_state, ]
                next_object_id += 1

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)

            if len(sequence_object_ids) > 0:
                info['sequence_object_ids'] = sequence_object_ids
                out = tracker.track(frame, info)
                prev_output = OrderedDict(out)

                if 'segmentation' in out:
                    frame_disp = overlay_mask(frame_disp, out['segmentation'])
                    mask_image = np.zeros(frame_disp.shape, dtype=frame_disp.dtype)

                    if save_results:
                        mask_image = overlay_mask(mask_image, out['segmentation'])
                        if not os.path.exists(self.results_dir):
                            os.makedirs(self.results_dir)
                        cv.imwrite(self.results_dir + f"seg_{frame_number}.jpg", mask_image)

                if 'target_bbox' in out:
                    for obj_id, state in out['target_bbox'].items():
                        state = [int(s) for s in state]
                        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                     _tracker_disp_colors[obj_id], 5)
                        if save_results:
                            output_boxes[obj_id].append(state)

            # Put text
            font_color = (255, 255, 255)
            msg = "Select target(s). Press 'r' to reset or 'q' to quit."
            cv.rectangle(frame_disp, (5, 5), (630, 40), (50, 50, 50), -1)
            cv.putText(frame_disp, msg, (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 2)

            if videofilepath is not None:
                msg = "Press SPACE to pause/resume the video."
                cv.rectangle(frame_disp, (5, 50), (530, 90), (50, 50, 50), -1)
                cv.putText(frame_disp, msg, (10, 75), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 2)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                next_object_id = 1
                sequence_object_ids = []
                prev_output = OrderedDict()

                info = OrderedDict()

                info['object_ids'] = []
                info['init_object_ids'] = []
                info['init_bbox'] = OrderedDict()
                tracker.initialize(frame, info)
                ui_control.mode = 'init'
            # 'Space' to pause video
            elif key == 32 and videofilepath is not None:
                paused = not paused

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = "webcam" if videofilepath is None else Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))
            print(f"Save results to: {base_results_path}")
            for obj_id, bbox in output_boxes.items():
                tracked_bb = np.array(bbox).astype(int)
                bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def run_vot2020(self, debug=None, visdom_info=None):
        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.run_id = self.run_id

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        if debug is None:
            visualization_ = getattr(params, 'visualization', False)
        else:
            visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)

        tracker = self.create_tracker(params)
        tracker.initialize_features()

        output_segmentation = tracker.predicts_segmentation_mask()

        import pytracking.evaluation.vot2020 as vot

        def _convert_anno_to_list(vot_anno):
            vot_anno = [vot_anno[0], vot_anno[1], vot_anno[2], vot_anno[3]]
            return vot_anno

        def _convert_image_path(image_path):
            return image_path

        """Run tracker on VOT."""

        if output_segmentation:
            handle = vot.VOT("mask")
        else:
            handle = vot.VOT("rectangle")

        vot_anno = handle.region()

        image_path = handle.frame()
        if not image_path:
            return
        image_path = _convert_image_path(image_path)

        image = self._read_image(image_path)

        if output_segmentation:
            vot_anno_mask = vot.make_full_size(vot_anno, (image.shape[1], image.shape[0]))
            bbox = masks_to_bboxes(torch.from_numpy(vot_anno_mask), fmt='t').squeeze().tolist()
        else:
            bbox = _convert_anno_to_list(vot_anno)
            vot_anno_mask = None

        out = tracker.initialize(image, {'init_mask': vot_anno_mask, 'init_bbox': bbox})

        if out is None:
            out = {}
        prev_output = OrderedDict(out)

        # Track
        while True:
            image_path = handle.frame()
            if not image_path:
                break
            image_path = _convert_image_path(image_path)

            image = self._read_image(image_path)

            info = OrderedDict()
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)

            if output_segmentation:
                pred = out['segmentation'].astype(np.uint8)
            else:
                state = out['target_bbox']
                pred = vot.Rectangle(*state)
            handle.report(pred, 1.0)

            segmentation = out['segmentation'] if 'segmentation' in out else None
            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            elif tracker.params.visualization:
                self.visualize(image, out['target_bbox'], segmentation)


    def run_vot(self, debug=None, visdom_info=None):
        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.run_id = self.run_id

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        if debug is None:
            visualization_ = getattr(params, 'visualization', False)
        else:
            visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)

        tracker = self.create_tracker(params)
        tracker.initialize_features()

        import pytracking.evaluation.vot as vot

        def _convert_anno_to_list(vot_anno):
            vot_anno = [vot_anno[0][0][0], vot_anno[0][0][1], vot_anno[0][1][0], vot_anno[0][1][1],
                        vot_anno[0][2][0], vot_anno[0][2][1], vot_anno[0][3][0], vot_anno[0][3][1]]
            return vot_anno

        def _convert_image_path(image_path):
            image_path_new = image_path[20:- 2]
            return "".join(image_path_new)

        """Run tracker on VOT."""

        handle = vot.VOT("polygon")

        vot_anno_polygon = handle.region()
        vot_anno_polygon = _convert_anno_to_list(vot_anno_polygon)

        init_state = convert_vot_anno_to_rect(vot_anno_polygon, tracker.params.vot_anno_conversion_type)

        image_path = handle.frame()
        if not image_path:
            return
        image_path = _convert_image_path(image_path)

        image = self._read_image(image_path)
        tracker.initialize(image, {'init_bbox': init_state})

        # Track
        while True:
            image_path = handle.frame()
            if not image_path:
                break
            image_path = _convert_image_path(image_path)

            image = self._read_image(image_path)
            out = tracker.track(image)
            state = out['target_bbox']

            handle.report(vot.Rectangle(state[0], state[1], state[2], state[3]))

            segmentation = out['segmentation'] if 'segmentation' in out else None
            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            elif tracker.params.visualization:
                self.visualize(image, out['target_bbox'], segmentation)

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()
        return params


    def init_visualization(self):
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()


    def visualize(self, image, state, segmentation=None):
        self.ax.cla()
        self.ax.imshow(image)
        if segmentation is not None:
            self.ax.imshow(segmentation, alpha=0.5)

        if isinstance(state, (OrderedDict, dict)):
            boxes = [v for k, v in state.items()]
        elif isinstance(state, list):
            boxes = state
        else:
            boxes = (state,)

        for i, box in enumerate(boxes, start=1):
            col = _tracker_disp_colors[i]
            col = [float(c) / 255.0 for c in col]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=col, facecolor='none')
            self.ax.add_patch(rect)

        if getattr(self, 'gt_state', None) is not None:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g', facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        draw_figure(self.fig)

        if self.pause_mode:
            keypress = False
            while not keypress:
                keypress = plt.waitforbuttonpress()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def _read_image(self, image_file: str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)



