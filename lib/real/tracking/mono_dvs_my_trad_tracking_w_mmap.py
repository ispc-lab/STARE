import os
import sys
import time
import datetime
import threading
import mmap
import struct

import numpy as np
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


BUFFER_HISTORY_MS = 20
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

    ##################################################################
    SHM_PATH = "/dev/shm/point2d_shm"
    SIZE = 20  # seq(uint64) + x(float32) + y(float32) + pad(4)

    fd = os.open(SHM_PATH, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
    os.ftruncate(fd, SIZE)
    buf = mmap.mmap(fd, SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
    os.close(fd)

    seq = 0
    pack = struct.Struct("<QffI")  # 8+4+4+4=20，可换 "<Qff" 并改大小
    ##################################################################

    last_time = time.perf_counter()
    last_time_2 = time.perf_counter()
    while not stop_event.is_set():
        if events_buffer[tracker_idx] is not None:

            curr_time = time.perf_counter()
            if curr_time - last_time > window / 1000.0:
                last_time = curr_time
            else:
                time.sleep(0.0001)
                continue

            info['previous_output'] = prev_output[tracker_idx]
            window_us = window * 1e3

            with buffer_rwlock.gen_rlock():
                # latest_ts = events_buffer[tracker_idx].getHighestTime()
                # cutoff_ts = int(latest_ts - window_us)
                # events = events_buffer[tracker_idx].sliceTime(cutoff_ts)
                events = events_buffer[tracker_idx]

            event_rep = convert_event_img_aedat(events.numpy(), 'VoxelGridComplex')
            out = ostrack.track(event_rep, info)

            prev_output[tracker_idx] = OrderedDict(out)
            pred_bbox[tracker_idx] = out['target_bbox']

            op_x, op_y, op_w, op_h = pred_bbox[tracker_idx]
            obj_center = [op_x + op_w / 2, op_y + op_h / 2]

            if obj_center is not None:
                seq += 1
                buf.seek(0)
                buf.write(pack.pack(seq, obj_center[0], obj_center[1], 0))

            # time.sleep(0.005)

            curr_time_2 = time.perf_counter()
            elapsed_time_2 = curr_time_2 - last_time_2
            last_time_2 = curr_time_2
            print(f"[{tracker_idx}] STARE FPS:", 1 / elapsed_time_2)

        else:
            time.sleep(0.0001)

    print(f"Tracking bbox collector [{tracker_idx}] thread has stopped.")


def main():

    # ------ Camera Initialization ------
    cameras = dv.io.discoverDevices()

    if len(cameras) == 0:
        print("No DVS cameras found. Please connect a camera and try again.")
        sys.exit(1)
    else:
        print("Available DVS cameras:")
        for idx, camera in enumerate(cameras):
            print(f"{idx}: {camera}")

        camera_id = int(input("Select a camera by entering its index: "))
        if camera_id < 0 or camera_id >= len(cameras):
            print("Invalid camera index. Exiting.")
            sys.exit(1)

    camera_name = cameras[camera_id]
    window_name = f"DVS Camera: {camera_name}"

    try:
        capture = dv.io.CameraCapture(camera_name)
        print("Camera capture started.")

    except Exception as e:
        print(f"Failed to start camera capture: {e}")
        sys.exit(1)

    capture.setDavisFrameInterval(datetime.timedelta(milliseconds=10))

    # ------ Start Event Stream Recording Thread ------
    events_buffer = [dv.EventStore(), dv.EventStore()]
    buffer_lock = rwlock.RWLockFair()
    stop_event = threading.Event()

    event_collecting_thread = threading.Thread(
        target=event_collector,
        args=(capture, 0, events_buffer, buffer_lock, stop_event, BUFFER_HISTORY_MS)
    )
    event_collecting_thread.start()

    # ------ OSTrack Tracker Initialization ------
    tracker = Tracker('ostrack', 'esot500mix', 'esot_500_20')
    params = tracker.get_parameters()
    params.debug = False

    ostrack = tracker.create_tracker(params)

    init_info = {
        # 'init_bbox': [164, 103, 14, 14],
        'init_bbox': [120, 133, 29, 31],
    }
    template = tracker._read_image(os.path.dirname(__file__) + '/init/template/right_fast_1.jpg')

    out = ostrack.initialize(template, init_info)
    if out is None:
        out = {}

    # ------ Start Tracking Threads ------
    prev_output = [OrderedDict(out)]
    pred_bbox = [init_info.get('init_bbox')]
    bbox_lock = rwlock.RWLockFair()

    tracking_thread = threading.Thread(
        target=tracking_bbox_collector,
        args=(ostrack, 0, SAMPLING_WINDOW_MS, events_buffer, buffer_lock, pred_bbox, bbox_lock, prev_output,
              stop_event)
    )
    tracking_thread.start()


    # ------ Visualization ------
    resolution = capture.getEventResolution()
    visualizer = dv.visualization.EventVisualizer(resolution)

    # Initialize the centers of the bounding boxes
    op_x, op_y, op_w, op_h = init_info['init_bbox']
    # obj_center = [op_x + op_w / 2, op_y + op_h / 2]

    # main loop for tracking and visualization
    while capture.isRunning():

        op_x, op_y, op_w, op_h = pred_bbox[0]
        obj_rect = [
            (int(op_x), int(op_y)),
            (int(op_x + op_w), int(op_y + op_h)),
        ]

        with buffer_lock.gen_rlock():
            latest_ts = events_buffer[0].getHighestTime()
            cutoff_ts = int(latest_ts - SAMPLING_WINDOW_MS * 1e3)
            events = events_buffer[0].sliceTime(cutoff_ts)

        if events is None:
            continue

        event_frame = visualizer.generateImage(events)
        cv.rectangle(
            event_frame,
            obj_rect[0],
            obj_rect[1],
            (0, 0, 255),
            2,
        )
        event_frame = cv.resize(event_frame, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)
        cv.imshow("Mono-Event-Tracking", event_frame)

        time.sleep(0.001)

        if cv.waitKey(2) & 0xFF == ord('q'):
            print("Camera capture stopped.")
            break

    cv.destroyAllWindows()
    print("Program exited.")


if __name__ == '__main__':
    main()