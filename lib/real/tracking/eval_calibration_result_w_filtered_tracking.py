import os
import sys
import time
import datetime
import threading

import numpy as np
import dv_processing as dv
import cv2 as cv
import matplotlib.pyplot as plt

from collections import OrderedDict
from readerwriterlock import rwlock
from mpl_toolkits.mplot3d import Axes3D


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


def setup_kalman_filter():
    # 状态量维度为6 (x, y, z, vx, vy, vz)
    # 测量量维度为3 (x, y, z)
    kf = cv.KalmanFilter(6, 3)

    dt = 1.0 # 时间步长，这里简化为1，实际可根据帧率调整

    # 状态转移矩阵 A (匀速模型)
    kf.transitionMatrix = np.array([[1, 0, 0, dt, 0, 0],
                                 [0, 1, 0, 0, dt, 0],
                                 [0, 0, 1, 0, 0, dt],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]], np.float32)

    # 测量矩阵 H
    kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0]], np.float32)

    # 过程噪声协方差 Q (表示我们对运动模型的信任程度，越小越信任)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03

    # 测量噪声协方差 R (表示我们对测量值的信任程度，越小越信任)
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5

    return kf


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


    # ------ Load stereo calibration parameters ------
    try:
        save_path = os.path.dirname(__file__) + '/stereo_calib_rect.npz'
        # save_path = './stereo_calib_rect.npz'
        calib = np.load(save_path)
        K1, D1 = calib['K1'], calib['D1']
        K2, D2 = calib['K2'], calib['D2']
        R, T = calib['R'], calib['T']  # 3x3, 3x1
        R0, t0 = calib['R0'], calib['t0']  # 3x3, 3x1
        scale_mm = float(calib['scale_mm'])

        print("Successfully loaded calibration parameters:")

    except FileNotFoundError:
        print("Error: Calibration file not found. Please run stereo_dvs_calibrate.py first.")
        sys.exit(1)


    # ------ Setup Kalman Filter ------
    kf = setup_kalman_filter()
    first_measurement = True
    max_jump_threshold = 200.0
    last_good_point = np.zeros(3, dtype=np.float32)


    # ------ Visualization ------
    resolution = capture.left.getEventResolution()
    visualizer = dv.visualization.EventVisualizer(resolution)

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter_plot = ax.scatter([], [], [], c='red', s=200, label='Current Position')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Real-time 3D Position')
    ax.legend()

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-1000, 1000)

    op_x, op_y, op_w, op_h = init_info[0]['init_bbox']
    left_center = [op_x + op_w / 2, op_y + op_h / 2]
    op_x, op_y, op_w, op_h = init_info[1]['init_bbox']
    right_center = [op_x + op_w / 2, op_y + op_h / 2]

    while capture.left.isRunning() and capture.right.isRunning():

        with buffer_lock[0].gen_rlock() and buffer_lock[1].gen_rlock():
            latest_ts = events_buffer[0].getHighestTime()
            cutoff_ts = int(latest_ts - SAMPLING_WINDOW_MS * 1e3)
            events_left = events_buffer[0].sliceTime(cutoff_ts)

            latest_ts = events_buffer[1].getHighestTime()
            cutoff_ts = int(latest_ts - SAMPLING_WINDOW_MS * 1e3)
            events_right = events_buffer[1].sliceTime(cutoff_ts)

        if events_left is not None and events_right is not None:
            with bbox_lock[0].gen_rlock() and bbox_lock[1].gen_rlock():
                op_x, op_y, op_w, op_h = pred_bbox[0]
                left_center = [op_x + op_w / 2, op_y + op_h / 2]
                left_pt = [
                    (int(op_x), int(op_y)),
                    (int(op_x + op_w), int(op_y + op_h)),
                ]

                op_x, op_y, op_w, op_h = pred_bbox[1]
                right_center = [op_x + op_w / 2, op_y + op_h / 2]
                right_pt = [
                    (int(op_x), int(op_y)),
                    (int(op_x + op_w), int(op_y + op_h)),
                ]

            left_event_frame = visualizer.generateImage(events_left)
            cv.rectangle(
                left_event_frame,
                left_pt[0],
                left_pt[1],
                (0, 0, 255),
                2,
            )
            cv.imshow("Left-STARE-Tracking", left_event_frame)

            right_event_frame = visualizer.generateImage(events_right)
            cv.rectangle(
                right_event_frame,
                right_pt[0],
                right_pt[1],
                (0, 0, 255),
                2,
            )
            cv.imshow("Right-STARE-Tracking", right_event_frame)

        if left_center is not None and right_center is not None:
            print("Left Feature Center:", left_center)
            print("Right Feature Center:", right_center)

            # get the centroids of the detected features, [x, y] format
            pt_left = np.array(left_center).reshape(1, 1, 2).astype(np.float32)
            pt_right = np.array(right_center).reshape(1, 1, 2).astype(np.float32)

            # remove distortion using the calibration parameters
            pt_left_undistorted = cv.undistortPoints(pt_left, K1, D1)
            pt_right_undistorted = cv.undistortPoints(pt_right, K2, D2)

            # prject the undistorted points to the image plane
            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # P1 = I * [R|T]
            P2 = np.hstack([R, T.reshape(3, 1)])

            # perform triangulation to get the 3D point
            point_4d = cv.triangulatePoints(P1, P2, pt_left_undistorted.T, pt_right_undistorted.T)
            point_3d = point_4d[:3] / point_4d[3]  # convert to 3D coordinates

            # convert to millimeters and apply rectification
            point_3d = R0.T @ (point_3d - t0) * scale_mm  # apply rectification and scale
            point_3d = point_3d.flatten()


            # plausibility check
            measurement = point_3d.astype(np.float32)
            is_outlier = False

            if not first_measurement:
                jump_distance = np.linalg.norm(measurement - last_good_point)
                if jump_distance > max_jump_threshold:
                    is_outlier = True
                    print(f"--- Outlier Detected! Jump distance: {jump_distance:.2f} mm > {max_jump_threshold} mm. Ignoring measurement. ---")

            # Kalman filter prediction and correction
            if first_measurement:
                # 首次测量，初始化滤波器
                kf.statePost = np.array([measurement[0], measurement[1], measurement[2], 0, 0, 0], dtype=np.float32)
                filtered_point_3d = measurement
                first_measurement = False
            else:
                kf.predict()
                if not is_outlier:
                    # 如果不是离群点，用测量值修正预测
                    corrected_state = kf.correct(measurement)
                    filtered_point_3d = corrected_state[:3].flatten()
                else:
                    # 如果是离群点，只信任预测结果，不使用测量值
                    filtered_point_3d = kf.statePost[:3].flatten()

            last_good_point = filtered_point_3d.copy()

            # scatter_plot._offsets3d = ([point_3d[0]], [point_3d[1]], [point_3d[2]])
            scatter_plot._offsets3d = ([filtered_point_3d[0]], [filtered_point_3d[1]], [filtered_point_3d[2]])

            print("Matching features detected, 3D Coordinates (mm): ")
            print(f"RAW: X={point_3d[0]:.2f}, Y={point_3d[1]:.2f}, Z={point_3d[2]:.2f}")
            print(f"FILTERED: X={filtered_point_3d[0]:.2f}, Y={filtered_point_3d[1]:.2f}, Z={filtered_point_3d[2]:.2f}")

        plt.draw()
        plt.pause(0.001)

        time.sleep(0.02)

        if not plt.fignum_exists(fig.number):
            print("Plot window closed. Exiting.")
            break

        if cv.waitKey(2) & 0xFF == ord('q'):
            print("Camera capture stopped.")
            break

    cv.destroyAllWindows()
    plt.ioff()
    plt.show()

    print("Program exited.")


if __name__ == '__main__':
    main()