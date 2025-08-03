import os
import sys
import time
import datetime
import threading

import numpy as np
import dv_processing as dv
import cv2 as cv


# Parameters for the chessboard calibration
CHESSBOARD_SIZE = (8, 5)
SQUARE_SIZE_MM = 25

# Create 3D object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM


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


objpoints = []
imgpoints_left = []
imgpoints_right = []

num_good_pairs = 0
target_pairs = 10


print(f"\n Please move the chessboard to collect {target_pairs} valid stereo image pairs...")
while num_good_pairs < target_pairs and capture.left.isRunning() and capture.right.isRunning():
    frame_left = frame_buffer[0]
    frame_right = frame_buffer[1]

    if frame_left is None or frame_right is None:
        continue

    cv.imshow(f"Camera Left", frame_left.image)
    cv.imshow(f"camera Right", frame_right.image)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        print("\n's' key pressed, attempting to capture...")

        gray_left = cv.cvtColor(frame_left.image, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(frame_right.image, cv.COLOR_BGR2GRAY)

        ret_left, corners_left = cv.findChessboardCorners(gray_left, CHESSBOARD_SIZE, None)
        ret_right, corners_right = cv.findChessboardCorners(gray_right, CHESSBOARD_SIZE, None)

        if ret_left and ret_right:
            vis_left = frame_left.image.copy()
            vis_right = frame_right.image.copy()
            cv.drawChessboardCorners(vis_left, CHESSBOARD_SIZE, corners_left, ret_left)
            cv.drawChessboardCorners(vis_right, CHESSBOARD_SIZE, corners_right, ret_right)
            cv.putText(vis_left, "VALID PAIR FOUND", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            combined_display = np.hstack((vis_left, vis_right))
            cv.imshow("Stereo Calibration Capture", combined_display)
            cv.waitKey(1)

            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_subpix_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_left.append(corners_subpix_left)
            imgpoints_right.append(corners_subpix_right)

            num_good_pairs += 1
            print(f"Successfully saved pair {num_good_pairs}/{target_pairs}.")

        else:
            print("--- CAPTURE FAILED: Could not detect corners in both views. Please reposition and try again. ---")

    elif key == ord('q'):
        print("User quit the capture process.")
        break


cv.destroyAllWindows()


if len(imgpoints_left) >= target_pairs:
    print("Stereo calibration in progress...")

    mono_left_save_path = os.path.dirname(__file__) + f'/mono_calib_{cameras[0]}.npz'
    mono_left_calib = np.load(mono_left_save_path)
    K1, D1 = mono_left_calib['K'], mono_left_calib['D']

    mono_right_save_path = os.path.dirname(__file__) + f'/mono_calib_{cameras[1]}.npz'
    mono_right_calib = np.load(mono_right_save_path)
    K2, D2 = mono_right_calib['K'], mono_right_calib['D']

    ret, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K1, D1, K2, D2,
        imageSize=img_size,
        flags=cv.CALIB_RATIONAL_MODEL,
        criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )

    if ret:
        print("\nStereo calibration successful!")
        print(f"\n Reprojection Error: {ret:.4f} pixels")

        save_path = os.path.join(os.path.dirname(__file__), 'stereo_calib.npz')
        np.savez(
            save_path,
            K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T,
            image_size=img_size
        )
        print(f"Calibration parameters saved to {save_path}")

        # visualize the calibration results
        print("\nPerforming stereo rectification to visually check calibration...")
        R1, R2, P1_rect, P2_rect, Q, roi1, roi2 = cv.stereoRectify(
            K1, D1, K2, D2, img_size, R, T, alpha=0.9)

        map1_left, map2_left = cv.initUndistortRectifyMap(K1, D1, R1, P1_rect, img_size, cv.CV_32FC1)
        map1_right, map2_right = cv.initUndistortRectifyMap(K2, D2, R2, P2_rect, img_size, cv.CV_32FC1)

        print("Displaying rectified live images. Check if features are on the same horizontal line. Press 'q' to exit.")
        while True:
            frame_left = frame_buffer[0]
            frame_right = frame_buffer[1]

            if frame_left is not None and frame_right is not None:
                rect_left = cv.remap(frame_left.image, map1_left, map2_left, cv.INTER_LINEAR)
                rect_right = cv.remap(frame_right.image, map1_right, map2_right, cv.INTER_LINEAR)

                combined_img = np.hstack((rect_left, rect_right))

                # draw horizontal lines at intervals
                for i in range(20, combined_img.shape[0], 25):
                    cv.line(combined_img, (0, i), (combined_img.shape[1], i), (0, 255, 0), 1)

                cv.imshow("Rectified Stereo Pair", combined_img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        print("\nStereo calibration failed. Please check the camera setup and try again.")

else:
    print("\nCannot collect enough valid image pairs for stereo calibration. Please check the chessboard movement and camera setup.")
    sys.exit(1)