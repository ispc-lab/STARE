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


def frame_collector_and_tracking(
        capture: dv.io.CameraCapture, camera_idx: int, frame_buffer: list,
        ostrack, tracker_idx: int,
        pred_bbox: list, bbox_rwlock: rwlock.RWLockFair, prev_output: list,
        stop_event: threading.Event
):
    print(f"Frame collector and tracking for camera [{camera_idx}] started.")

    info = {}
    info['previous_output'] = prev_output[tracker_idx]
    # last_time = time.perf_counter()

    while not stop_event.is_set():
        frame = capture.getNextFrame()
        if frame is not None:
            frame_buffer[camera_idx] = frame

            info['previous_output'] = prev_output[tracker_idx]

            img = cv.cvtColor(frame.image, cv.COLOR_BGR2RGB)
            out = ostrack.track(img, info)

            prev_output[tracker_idx] = OrderedDict(out)

            with bbox_rwlock.gen_wlock():
                pred_bbox[tracker_idx] = out['target_bbox']

            # curr_time = time.perf_counter()
            # elapsed_time = curr_time - last_time
            # last_time = curr_time
            # print(f"[{camera_idx}] RGB FPS:", 1 / elapsed_time)

        else:
            time.sleep(0.0001)

    print(f"Frame collector and tracking for camera [{camera_idx}] stopped.")


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

    capture.left.setDavisFrameInterval(datetime.timedelta(milliseconds=10))
    capture.right.setDavisFrameInterval(datetime.timedelta(milliseconds=10))


    # ------ OSTrack Tracker Initialization ------
    tracker_rgb = Tracker('ostrack', 'vitb_256_mae_ce_32x4_ep300', 'esot_500_20')
    params_rgb = tracker_rgb.get_parameters()
    params_rgb.debug = False

    ostrack = [
        tracker_rgb.create_tracker(params_rgb),
        tracker_rgb.create_tracker(params_rgb),
    ]

    init_info = [
        {'init_bbox': [158, 99, 23, 25], },
        {'init_bbox': [141, 121, 28, 28], },
    ]

    template = [
        tracker_rgb._read_image(os.path.dirname(__file__) + '/init/template/left_rgb_1.jpg'),
        tracker_rgb._read_image(os.path.dirname(__file__) + '/init/template/right_rgb_1.jpg'),
    ]

    outs = [ostrack[i].initialize(template[i], init_info[i]) for i in range(2)]
    outs = [out if out is not None else {} for out in outs]


    # ------ Start Tracking Threads ------
    frame_buffer = [None, None]
    prev_output = [OrderedDict(out) for out in outs]
    pred_bbox = [init_info[i].get('init_bbox') for i in range(2)]
    bbox_lock = [rwlock.RWLockFair(), rwlock.RWLockFair()]
    stop_event = [threading.Event(), threading.Event()]

    tracking_thread_rgb_left = threading.Thread(
        target=frame_collector_and_tracking,
        args=(capture.left, 0, frame_buffer, ostrack[0], 0, pred_bbox, bbox_lock[0], prev_output, stop_event[0])
    )
    tracking_thread_rgb_left.start()

    tracking_thread_rgb_right = threading.Thread(
        target=frame_collector_and_tracking,
        args=(capture.right, 1, frame_buffer, ostrack[1], 1, pred_bbox, bbox_lock[1], prev_output, stop_event[1])
    )
    tracking_thread_rgb_right.start()


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
        print("Error: Rectified Calibration file not found. Please run rectify_calibration_result.py first.")
        sys.exit(1)


    # ------ Visualization ------
    resolution = capture.left.getEventResolution()
    visualizer = dv.visualization.EventVisualizer(resolution)

    plt.ion()

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Real-time 3D Tracking and X-Z Plane View', fontsize=16)


    # --- left subplot: 3D view ---
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    scatter_3d = ax_3d.scatter([], [], [], c='red', s=150, label='Current Position')

    ax_3d.set_xlabel('X (mm)')
    ax_3d.set_ylabel('Y (mm)')
    ax_3d.set_zlabel('Z (mm)')
    ax_3d.set_title('3D View')

    ax_3d.set_xlim(-1000, 1000)
    ax_3d.set_ylim(-1000, 1000)
    ax_3d.set_zlim(-1000, 1000)
    ax_3d.legend()

    # --- right subplot: 2D X-Z plane view (front-end view) ---
    ax_2d = fig.add_subplot(1, 2, 2)
    scatter_xz = ax_2d.scatter([], [], c='blue', s=150, label='Current X-Z Position')

    ax_2d.set_xlabel('X (mm)')
    ax_2d.set_ylabel('Z (mm)')
    ax_2d.set_title('Top-Down View (X-Z Plane)')

    ax_2d.set_xlim(-1500, 1500)
    ax_2d.set_ylim(-1500, 1500)

    ax_2d.set_aspect('equal', adjustable='box')
    ax_2d.grid(True)
    ax_2d.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Initialize the centers of the bounding boxes
    op_x, op_y, op_w, op_h = init_info[0]['init_bbox']
    left_center = [op_x + op_w / 2, op_y + op_h / 2]
    op_x, op_y, op_w, op_h = init_info[1]['init_bbox']
    right_center = [op_x + op_w / 2, op_y + op_h / 2]

    # main loop for tracking and visualization
    while capture.left.isRunning() and capture.right.isRunning():
        frame_left = frame_buffer[0]
        frame_right = frame_buffer[1]

        if frame_left is None or frame_right is None:
            continue

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

        img_left = frame_left.image.copy()
        cv.rectangle(
            img_left,
            left_pt[0],
            left_pt[1],
            (0, 0, 255),
            2,
        )
        # img_left = cv.resize(img_left, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)
        cv.imshow("Left-RGB-Tracking", img_left)

        img_right = frame_right.image.copy()
        cv.rectangle(
            img_right,
            right_pt[0],
            right_pt[1],
            (0, 0, 255),
            2,
        )
        # img_right = cv.resize(img_right, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)
        cv.imshow("Right-RGB-Tracking", img_right)

        if left_center is not None and right_center is not None:
            # print("Left Feature Center:", left_center)
            # print("Right Feature Center:", right_center)

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

            # update the scatter plot with the new 3D point
            scatter_3d._offsets3d = ([point_3d[0]], [point_3d[1]], [point_3d[2]])
            scatter_xz.set_offsets([point_3d[0], point_3d[2]])

            print("Matching features detected, 3D Coordinates (mm): ")
            print(f"X={point_3d[0]:.2f}, Y={point_3d[1]:.2f}, Z={point_3d[2]:.2f}")


        plt.draw()
        plt.pause(0.001)

        # time.sleep(0.02)

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