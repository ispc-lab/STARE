import os
import sys
import time
import datetime
import threading
import zmq
import json
import struct

import numpy as np
import dv_processing as dv
import cv2 as cv

from collections import OrderedDict
# from readerwriterlock import rwlock


env_path = os.path.join(os.path.dirname(__file__), '..', '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker


def frame_collector_and_tracking(
        capture: dv.io.CameraCapture, frame_buffer: list,
        ostrack, pred_bbox: list, prev_output: list,
        stop_event: threading.Event
):
    print(f"Frame collector and tracking for Davis346 started.")

    capture.setDavisFrameInterval(datetime.timedelta(milliseconds=10))

    info = {}
    info['previous_output'] = prev_output[0]
    # last_time = time.perf_counter()

    while not stop_event.is_set():
        frame = capture.getNextFrame()

        if frame is not None:
            frame_buffer[0] = frame
            info['previous_output'] = prev_output[0]

            img = cv.cvtColor(frame.image, cv.COLOR_BGR2RGB)
            out = ostrack.track(img, info)

            prev_output[0] = OrderedDict(out)
            pred_bbox[0] = out['target_bbox']

            # curr_time = time.perf_counter()
            # elapsed_time = curr_time - last_time
            # last_time = curr_time
            # print(f" RGB FPS:", 1 / elapsed_time)

        else:
            time.sleep(0.001)

    print(f"Frame collector and tracking for Davis346 stopped.")


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

    # ------ OSTrack Tracker Initialization ------
    tracker_rgb = Tracker('ostrack', 'vitb_256_mae_ce_32x4_ep300', 'esot_500_20')
    params_rgb = tracker_rgb.get_parameters()
    params_rgb.debug = False

    ostrack = tracker_rgb.create_tracker(params_rgb)

    init_info = {
        'init_bbox': [158, 99, 23, 25],
    }
    template = tracker_rgb._read_image(os.path.dirname(__file__) + '/init/template/left_rgb_1.jpg')

    out = ostrack.initialize(template, init_info)
    if out is None:
        out = {}

    # ------ Start Tracking Threads ------
    frame_buffer = [None]
    prev_output = [OrderedDict(out)]
    pred_bbox = [init_info.get('init_bbox')]
    stop_event = threading.Event()

    tracking_thread_rgb = threading.Thread(
        target=frame_collector_and_tracking,
        args=(capture, frame_buffer, ostrack, pred_bbox, prev_output, stop_event)
    )
    tracking_thread_rgb.start()


    # ------ ZeroMQ Initialization ------
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.IMMEDIATE, 1)
    socket.setsockopt(zmq.SNDHWM, 1)  # 仅保留1条待发送
    socket.setsockopt(zmq.CONFLATE, 1)  # 队列永远只有最新帧

    # socket.connect("tcp://192.168.110.7:5555")
    socket.connect("ipc:///tmp/track_pipe.ipc")
    # socket.connect("tcp://192.168.110.249:5555")

    # ------ Visualization ------
    resolution = capture.getEventResolution()
    # visualizer = dv.visualization.EventVisualizer(resolution)

    # Initialize the centers of the bounding boxes
    op_x, op_y, op_w, op_h = init_info['init_bbox']
    obj_center = [op_x + op_w / 2, op_y + op_h / 2]

    # main loop for tracking and visualization
    while capture.isRunning():
        frame = frame_buffer[0]
        if frame is None:
            continue

        op_x, op_y, op_w, op_h = pred_bbox[0]
        obj_center = [op_x + op_w / 2, op_y + op_h / 2]
        obj_rect = [
            (int(op_x), int(op_y)),
            (int(op_x + op_w), int(op_y + op_h)),
        ]

        img = frame.image.copy()
        cv.rectangle(
            img,
            obj_rect[0],
            obj_rect[1],
            (0, 0, 255),
            2,
        )
        # img = cv.resize(img, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)
        cv.imshow("Mono-RGB-Tracking", img)

        if obj_center is not None:
            # send the 2D point to the ZeroMQ socket
            data = struct.pack('<2f', *obj_center)
            try:
                socket.send(data, flags=zmq.NOBLOCK)
            except zmq.Again:
                print("Warning: ZeroMQ send timeout, skipping this point.")

        time.sleep(0.001)

        if cv.waitKey(2) & 0xFF == ord('q'):
            print("Camera capture stopped.")
            break

    cv.destroyAllWindows()
    print("Program exited.")


if __name__ == '__main__':
    main()