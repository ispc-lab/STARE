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
import matplotlib.pyplot as plt


IP_PREFIX = "tcp://192.168.110.7:555"


def frame_collector_and_sending(
        capture: dv.io.CameraCapture,
        camera_idx: int,
        frame_buffer: list,
        stop_event: threading.Event
):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)  # keep only one frame in the queue
    socket.setsockopt(zmq.CONFLATE, 1)  # keep only the latest frame
    socket.connect(IP_PREFIX + f"{camera_idx}")

    print(f"Frame collector and sending for camera [{camera_idx}] started.")
    # last_time = time.perf_counter()

    while not stop_event.is_set():

        frame = capture.getNextFrame()

        if frame is not None:
            frame_buffer[camera_idx] = frame
            img_bytes = cv.imencode('.jpg', frame.image, [int(cv.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()

            try:
                socket.send(img_bytes, flags=zmq.NOBLOCK)

            except zmq.Again:
                print(f"Socket send failed for camera [{camera_idx}]. Retrying...")
                time.sleep(0.001)
                continue

            # curr_time = time.perf_counter()
            # elapsed_time = curr_time - last_time
            # last_time = curr_time
            # print(f"[{camera_idx}] RGB FPS:", 1 / elapsed_time)

        else:
            time.sleep(0.0001)

    print(f"Frame collector and tracking for camera [{camera_idx}] stopped.")


def bbox_collector(
        camera_idx: int,
        pred_bbox: list,
        stop_event: threading.Event
):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(IP_PREFIX + f"{camera_idx + 2}")

    print(f"bbox collector for camera [{camera_idx}] started.")
    while not stop_event.is_set():
        try:
            data = socket.recv(flags=zmq.NOBLOCK)
            bbox = struct.unpack('<4f', data)
            pred_bbox[camera_idx] = bbox

        except zmq.Again:
            time.sleep(0.001)
            continue

    print(f"bbox collector for camera [{camera_idx}] stopped.")


def point3d_collector(
        point3d_buffer: list,
        stop_event: threading.Event
):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(IP_PREFIX + "5")

    print(f"3D point collector started.")
    while not stop_event.is_set():
        try:
            data = socket.recv(flags=zmq.NOBLOCK)
            point3d = struct.unpack('<3f', data)
            point3d_buffer[0] = point3d

        except zmq.Again:
            time.sleep(0.001)
            continue

    print(f"3D point collector stopped.")


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


    # ------ Start Sending and Receiving Threads ------
    frame_buffer = [None, None]
    stop_event = [threading.Event(), threading.Event()]
    pred_bbox = [None, None]
    point3d_buffer = [None]

    sending_thread_rgb_left = threading.Thread(
        target=frame_collector_and_sending,
        args=(capture.left, 0, frame_buffer, stop_event[0])
    )
    sending_thread_rgb_left.start()

    sending_thread_rgb_right = threading.Thread(
        target=frame_collector_and_sending,
        args=(capture.right, 1, frame_buffer, stop_event[1])
    )
    sending_thread_rgb_right.start()

    receiving_bbox_thread_left = threading.Thread(
        target=bbox_collector,
        args=(0, pred_bbox, stop_event[0])
    )
    receiving_bbox_thread_left.start()

    receiving_bbox_thread_right = threading.Thread(
        target=bbox_collector,
        args=(1, pred_bbox, stop_event[1])
    )
    receiving_bbox_thread_right.start()

    receiving_point3d_thread = threading.Thread(
        target=point3d_collector,
        args=(point3d_buffer, stop_event[0])
    )
    receiving_point3d_thread.start()


    # ------ Visualization ------
    resolution = capture.left.getEventResolution()
    # visualizer = dv.visualization.EventVisualizer(resolution)

    plt.ion()

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Real-time 3D Tracking and X-Z Plane View with Remote Processing', fontsize=16)

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

    # main loop for tracking and visualization
    while capture.left.isRunning() and capture.right.isRunning():
        frame_left = frame_buffer[0]
        frame_right = frame_buffer[1]

        if frame_left is None or frame_right is None:
            continue

        if pred_bbox[0] is None or pred_bbox[1] is None:
            print("Waiting for bounding box predictions...")
            time.sleep(0.01)
            continue

        if point3d_buffer[0] is None:
            print("Waiting for 3D point data...")
            time.sleep(0.01)
            continue

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
        cv.imshow("Left-RGB-Tracking with Remote Processing", img_left)

        img_right = frame_right.image.copy()
        cv.rectangle(
            img_right,
            right_pt[0],
            right_pt[1],
            (0, 0, 255),
            2,
        )
        # img_right = cv.resize(img_right, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)
        cv.imshow("Right-RGB-Tracking with Remote Processing", img_right)

        point_3d = point3d_buffer[0]

        # update the scatter plot with the new 3D point
        scatter_3d._offsets3d = ([point_3d[0]], [point_3d[1]], [point_3d[2]])
        scatter_xz.set_offsets([point_3d[0], point_3d[2]])

        print("Matching features detected, 3D Coordinates (mm): ")
        print(f"X={point_3d[0]:.2f}, Y={point_3d[1]:.2f}, Z={point_3d[2]:.2f}")

        plt.draw()
        plt.pause(0.001)

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