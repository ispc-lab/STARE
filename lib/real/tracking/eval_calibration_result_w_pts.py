import os
import sys
import time
import datetime
import threading

import numpy as np
import dv_processing as dv
import cv2 as cv


def frame_collector(capture: dv.io.CameraCapture, camera_idx: int, frame_buffer: list):
    print(f"Frame collector for camera [{camera_idx}] started.")

    while capture.isRunning():
        frame = capture.getNextFrame()
        if frame is not None:
            frame_buffer[camera_idx] = frame
        else:
            time.sleep(0.0001)

    print(f"Frame collector for camera [{camera_idx}] stopped.")


try:
    cameras = dv.io.discoverDevices()
    if len(cameras) < 2:
        print("Error: Less than two DVS cameras found. Please connect a stereo pair and try again.")
        sys.exit(1)
    else:
        print("Available DVS cameras:")
        for idx, camera in enumerate(cameras):
            print(f"{idx}: {camera}")

    capture = dv.io.StereoCapture(cameras[0], cameras[1])
    print("Stereo Camera capture started.")

    capture.left.setDavisFrameInterval(datetime.timedelta(milliseconds=10))
    capture.right.setDavisFrameInterval(datetime.timedelta(milliseconds=10))
    img_size = capture.left.getFrameResolution()

except Exception as e:
    print(f"Failed to start camera capture: {e}")
    sys.exit(1)


frame_buffer = [None, None]

collector_thread_left = threading.Thread(
    target=frame_collector,
    args=(capture.left, 0, frame_buffer),
    name="Left-Frame-Collector"
)
collector_thread_left.start()

collector_thread_right = threading.Thread(
    target=frame_collector,
    args=(capture.right, 1, frame_buffer),
    name="Right-Frame-Collector"
)
collector_thread_right.start()



try:
    save_path = os.path.dirname(__file__) + '/stereo_calib_rect.npz'
    # save_path = './stereo_calibration_rect.npz'
    calib = np.load(save_path)
    K1, D1 = calib['K1'], calib['D1']
    K2, D2 = calib['K2'], calib['D2']
    R, T = calib['R'], calib['T']  # 3x3, 3x1
    R0, t0 = calib['R0'], calib['t0']  # 3x3, 3x1

    # scale_mm = float(calib['scale_mm'])
    scale_mm = 1

    image_size = calib['image_size']

    print("Successfully loaded calibration parameters:")

except FileNotFoundError:
    print("Error: Calibration file not found. Please run stereo_dvs_calibrate.py first.")
    sys.exit(1)


print("\n>> test mode: select 3D point from stereo images")
print("   - Press  s  to freeze frames and start point selection")
print("   - Press  q  or close the window to exit\n")

clicked_pt = [None, None]


def click_callback(event, x, y, flags, param):
    img, clicked_pt, idx, wdname = param
    if event == cv.EVENT_LBUTTONDOWN:
        clicked_pt[idx] = [x, y]
        print(f"[{wdname}] select: ({x}, {y})")
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv.imshow(wdname, img)


while capture.left.isRunning() and capture.right.isRunning():
    frame_left  = frame_buffer[0]
    frame_right = frame_buffer[1]

    if frame_left is None or frame_right is None:
        continue

    cv.imshow("Left Live",  frame_left.image)
    cv.imshow("Right Live", frame_right.image)

    key = cv.waitKey(1) & 0xFF

    if key == ord('s'):
        print(f"\n's' key pressed!")

        print(f"\nPlease select the coordinates of the reference point in the left image:")
        cv.setMouseCallback(
            "Left Live",
            click_callback,
            param=(frame_left.image.copy(), clicked_pt, 0, "Camera Left")
        )

        print("Waiting for user to select points...")
        while True:
            if cv.waitKey(20) & 0xFF == 27:
                break

        print(f"\nPlease select the coordinates of the reference point in the right image:")
        cv.setMouseCallback(
            "Right Live",
            click_callback,
            param=(frame_right.image.copy(), clicked_pt, 1, "Camera Right")
        )

        print("Waiting for user to select points...")
        while True:
            if cv.waitKey(20) & 0xFF == 27:
                break

        pl = np.array(clicked_pt[0],  dtype=np.float32).reshape(-1, 2)
        pr = np.array(clicked_pt[1],  dtype=np.float32).reshape(-1, 2)

        pl = cv.undistortPoints(pl, K1, D1).reshape(-1, 2)
        pr = cv.undistortPoints(pr, K2, D2).reshape(-1, 2)

        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # P1 = I * [R|T]
        P2 = np.hstack([R, T.reshape(3, 1)])

        X4 = cv.triangulatePoints(P1, P2, pl.T, pr.T)  # 4×N

        X3 = (X4[:3] / X4[3])               # old world coordinates in mm
        X3 = R0.T @ (X3 - t0) * scale_mm    # new coordinate system
        X3 = X3.flatten()

        print(f"\n===> Selected points' 3D coordinates (mm): X={X3[0]:.2f},  Y={X3[1]:.2f},  Z={X3[2]:.2f}\n")

    elif key == ord('q') or key == 27:
        break

cv.destroyAllWindows()
print("Finish.")