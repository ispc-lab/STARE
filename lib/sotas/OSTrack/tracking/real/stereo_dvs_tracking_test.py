import os
import sys
import time
import datetime
import threading

import dv_processing as dv
import cv2 as cv

from collections import OrderedDict
from readerwriterlock import rwlock


env_path = os.path.join(os.path.dirname(__file__), '..', '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'pytracking')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.utils.convert_event_img import convert_event_img_aedat


BUFFER_HISTORY_MS = 100
SAMPLING_WINDOW_MS = 20


def event_collector(capture: dv.io.CameraCapture, camera_idx: int, events_buffer: list, buffer_rwlock: rwlock.RWLockFair,
                    stop_event: threading.Event, history_ms: int):
    print(f"Event collector for camera [{camera_idx}] started.")
    history_us = history_ms * 1e3

    while not stop_event.is_set():
        events = capture.getNextEventBatch()
        if events is not None:
            with buffer_rwlock.gen_wlock():
                events_buffer[camera_idx].add(events)
                if len(events_buffer[camera_idx]) > 0:
                    latest_ts = events_buffer[camera_idx].getHighestTime()
                    cutoff_ts = int(latest_ts - history_us)
                    events_buffer[camera_idx] = events_buffer[camera_idx].sliceTime(cutoff_ts)

        else:
            time.sleep(0.0001)

    print(f"Event collector for camera [{camera_idx}] stopped.")


def tracking_bbox_collector(ostrack, tracker_idx: int, window: int, events_buffer: list, buffer_rwlock: rwlock.RWLockFair, pred_bbox: list, bbox_rwlock: rwlock.RWLockFair, prev_output: list, stop_event: threading.Event):
    print(f"Tracking bbox collector [{tracker_idx}] thread has started.")

    info = {}
    info['previous_output'] = prev_output[tracker_idx]

    while not stop_event.is_set():
        if events_buffer[tracker_idx] is not None:
            info['previous_output'] = prev_output[tracker_idx]
            window_us = window * 1e3

            with buffer_rwlock.gen_rlock():
                latest_ts = events_buffer[tracker_idx].getHighestTime()
                cutoff_ts = int(latest_ts - window_us)
                events = events_buffer[tracker_idx].sliceTime(cutoff_ts)

            event_rep = convert_event_img_aedat(events.numpy(), 'VoxelGridComplex')
            out = ostrack.track(event_rep, info)

            prev_output[tracker_idx] = OrderedDict(out)

            with bbox_rwlock.gen_wlock():
                pred_bbox[tracker_idx] = out['target_bbox']

        else:
            time.sleep(0.0001)

    print(f"Tracking bbox collector [{tracker_idx}] thread has stopped.")


def main():

    # ------ Camera Initialization ------
    cameras = dv.io.discoverDevices()

    if len(cameras) < 2:
        print("Error: Less than two DVS cameras found. Please connect a stereo pair and try again.")
        sys.exit(1)

    else:
        print("Available DVS cameras:")
        for idx, camera in enumerate(cameras):
            print(f"{idx}: {camera}")

    try:
        capture = dv.io.StereoCapture(cameras[0], cameras[1])
        print("Stereo Camera capture started.")

    except Exception as e:
        print(f"Failed to start camera capture: {e}")
        sys.exit(1)


    # ------ Start Event Stream Recording Thread ------
    events_buffer = [dv.EventStore(), dv.EventStore()]
    buffer_lock = [rwlock.RWLockFair(), rwlock.RWLockFair()]
    stop_event = [threading.Event(), threading.Event()]

    collector_thread_left = threading.Thread(
        target=event_collector,
        args=(capture.left, 0, events_buffer, buffer_lock[0], stop_event[0], BUFFER_HISTORY_MS)
    )
    collector_thread_left.start()

    collector_thread_right = threading.Thread(
        target=event_collector,
        args=(capture.right, 1, events_buffer, buffer_lock[1], stop_event[1], BUFFER_HISTORY_MS)
    )
    collector_thread_right.start()


    # ------ OSTrack Tracker Initialization ------
    tracker = Tracker('ostrack', 'esot500mix', 'esot_500_20')
    params = tracker.get_parameters()
    params.debug = False

    ostrack = [
        tracker.create_tracker(params),
        tracker.create_tracker(params),
    ]

    init_info = [
        {'init_bbox': [175, 117, 33, 35],},
        {'init_bbox': [110, 136, 52, 56],},
    ]

    template = [
        tracker._read_image(os.path.dirname(__file__) + '/init/template/left_1.jpg'),
        tracker._read_image(os.path.dirname(__file__) + '/init/template/right_1.jpg'),
    ]

    outs = [ostrack[i].initialize(template[i], init_info[i]) for i in range(2)]
    outs = [out if out is not None else {} for out in outs]


    # ------ Start Tracking Threads ------
    prev_output = [OrderedDict(out) for out in outs]
    pred_bbox = [init_info[i].get('init_bbox') for i in range(2)]
    bbox_lock = [rwlock.RWLockFair(), rwlock.RWLockFair()]

    tracking_thread_left = threading.Thread(
        target=tracking_bbox_collector,
        args=(ostrack[0], 0, SAMPLING_WINDOW_MS, events_buffer, buffer_lock[0], pred_bbox, bbox_lock[0], prev_output, stop_event[0])
    )
    tracking_thread_left.start()

    tracking_thread_right = threading.Thread(
        target=tracking_bbox_collector,
        args=(ostrack[1], 1, SAMPLING_WINDOW_MS, events_buffer, buffer_lock[1], pred_bbox, bbox_lock[1], prev_output, stop_event[1])
    )
    tracking_thread_right.start()

    # ------ Visualization ------
    resolution = capture.left.getEventResolution()
    visualizer = dv.visualization.EventVisualizer(resolution)

    while capture.left.isRunning() and capture.right.isRunning():

        with buffer_lock[0].gen_rlock():
            latest_ts = events_buffer[0].getHighestTime()
            cutoff_ts = int(latest_ts - SAMPLING_WINDOW_MS * 1e3)
            events_left = events_buffer[0].sliceTime(cutoff_ts)

        if events_left is not None:
            with bbox_lock[0].gen_rlock():
                op_x, op_y, op_w, op_h = pred_bbox[0]

            left_event_frame = visualizer.generateImage(events_left)
            cv.rectangle(
                left_event_frame,
                (int(op_x), int(op_y)),
                (int(op_x + op_w), int(op_y + op_h)),
                (0, 0, 255),
                2,
            )
            cv.imshow("Left-STARE-Tracking", left_event_frame)


        with buffer_lock[1].gen_rlock():
            latest_ts = events_buffer[1].getHighestTime()
            cutoff_ts = int(latest_ts - SAMPLING_WINDOW_MS * 1e3)
            events_right = events_buffer[1].sliceTime(cutoff_ts)

        if events_right is not None:
            with bbox_lock[1].gen_rlock():
                op_x, op_y, op_w, op_h = pred_bbox[1]

            right_event_frame = visualizer.generateImage(events_right)
            cv.rectangle(
                right_event_frame,
                (int(op_x), int(op_y)),
                (int(op_x + op_w), int(op_y + op_h)),
                (0, 0, 255),
                2,
            )
            cv.imshow("Right-STARE-Tracking", right_event_frame)


        time.sleep(0.02)

        if cv.waitKey(2) & 0xFF == ord('q'):
            print("Camera capture stopped.")
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
