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


# ------ Load stereo calibration parameters ------
try:
    save_path = os.path.dirname(__file__) + '/stereo_calib.npz'
    # save_path = './stereo_calibration.npz'
    calib = np.load(save_path)
    K1, D1 = calib['K1'], calib['D1']
    K2, D2 = calib['K2'], calib['D2']
    R, T = calib['R'], calib['T']
    image_size = calib['image_size']
    print("Successfully loaded calibration parameters:")

except FileNotFoundError:
    print("Error: Calibration file not found. Please run stereo_dvs_calibrate.py first.")
    sys.exit(1)


cnt = 0
ref_pt_num = 3

left_pts = []
right_pts = []

def mouse_callback(event, x, y, flags, param):
    img, ref_pts, cnt, wdname = param
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        ref_pts.append([x, y])
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv.imshow(wdname, img)

        print(len(ref_pts), ref_pts)


while cnt < ref_pt_num and capture.left.isRunning() and capture.right.isRunning():
    frame_left = frame_buffer[0]
    frame_right = frame_buffer[1]

    if frame_left is None or frame_right is None:
        continue

    cv.imshow(f"Camera Left", frame_left.image)
    cv.imshow(f"Camera Right", frame_right.image)

    key = cv.waitKey(1) & 0xFF

    if key == ord('s'):
        print(f"\n's' key pressed!")

        print(f"\nPlease select the coordinates of the reference point [{cnt}] in the left image:")
        cv.setMouseCallback(
            "Camera Left",
            mouse_callback,
            param=(frame_left.image.copy(), left_pts, cnt, f"Camera Left [{cnt}]")
        )

        print("Waiting for user to select points...")
        while len(left_pts) <= cnt:
            if cv.waitKey(20) & 0xFF == 27:
                break

        print(f"\nPlease select the coordinates of the reference point [{cnt}] in the right image:")
        cv.setMouseCallback(
            "Camera Right",
            mouse_callback,
            param=(frame_right.image.copy(), right_pts, cnt, f"Camera Right [{cnt}]")
        )

        print("Waiting for user to select points...")
        while len(right_pts) <= cnt:
            if cv.waitKey(20) & 0xFF == 27:
                break

        cnt += 1
        print(f"Successfully saved reference point {cnt}/{ref_pt_num}.")

    elif key == ord('q'):
        print("User quit the rectify process.")
        break


cv.destroyAllWindows()


if cnt >= ref_pt_num:
    print("Rectifying the stereo calibration results in progress...")

    left_pts = np.array(left_pts, dtype=np.float32)
    right_pts = np.array(right_pts, dtype=np.float32)

    # P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    # P2 = K2 @ np.hstack([R, T.reshape(-1, 1)])

    left_pts = cv.undistortPoints(left_pts, K1, D1).reshape(-1, 2)
    right_pts = cv.undistortPoints(right_pts, K2, D2).reshape(-1, 2)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, T.reshape(-1, 1)])

    pts4d = cv.triangulatePoints(P1, P2, left_pts.T, right_pts.T)
    pts3d = (pts4d[:3] / pts4d[3]).T  # (3, 3)

    O_old, Px_old, Pz_old = pts3d

    # print("\nReference points in old coordinate system:")
    # print(f"O (origin): {O_old}")
    # print(f"Px (x-axis point): {Px_old}")
    # print(f"Pz (z-axis point): {Pz_old}")

    x_axis = (Px_old - O_old)
    x_axis /= np.linalg.norm(x_axis)

    z_axis = (Pz_old - O_old)
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)

    R0 = np.vstack([x_axis, y_axis, z_axis]).T
    t0 = O_old.reshape(3, 1)

    # ---------- rectify the unit distance ----------
    measured_dx = np.linalg.norm(Px_old - O_old)  # distance in old coordinate system
    scale_mm = 225.0 / measured_dx  # mm

    # print("Rotation matrix R0:", R0)
    # print("Translation vector t0:", t0)
    print(f"Scale factor to mm: {scale_mm:.4f}")

    # check if the Z distance is approximately 125 mm
    measured_dz = np.linalg.norm(Pz_old - O_old) * scale_mm
    print(f"[Check]  Z distance = {measured_dz:.2f} mm (Target 125 mm)")

    save_path = os.path.join(os.path.dirname(__file__), 'stereo_calib_rect.npz')
    np.savez(
        save_path,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T,
        R0=R0, t0=t0, scale_mm=scale_mm,
        image_size=img_size,
    )
    print(f"Rectified calibration parameters saved to {save_path}")

else:
    print("\nCannot collect enough valid image pairs for stereo calibration. Please check the chessboard movement and camera setup.")
    sys.exit(1)
