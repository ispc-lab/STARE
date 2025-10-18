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

    last_time = time.perf_counter()

    while capture.isRunning():
        frame = capture.getNextFrame()
        if frame is not None:
            frame_buffer[camera_idx] = frame

            curr_time = time.perf_counter()
            elapsed_time = curr_time - last_time
            last_time = curr_time
            print(f"[{camera_idx}] RGB FPS:", 1 / elapsed_time)

        else:
            time.sleep(0.001)

    print(f"Frame collector for camera [{camera_idx}] stopped.")


try:
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
    img_size = capture.getFrameResolution()

except Exception as e:
    print(f"Failed to start camera capture: {e}")
    sys.exit(1)


frame_buffer = [None, None]

collector_thread = threading.Thread(
    target=frame_collector,
    args=(capture, camera_id, frame_buffer),
    name="RGB-Frame-Collector"
)
collector_thread.start()


objpoints = []
imgpoints = []

num_good_pts = 0
target_pts = 10

print(f"\n Please move the chessboard to collect {target_pts} valid monocular images ...")
while num_good_pts < target_pts and capture.isRunning():

    frame = frame_buffer[camera_id]
    if frame is None:
        continue

    cv.imshow(camera_name, frame.image)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        print("\n's' key pressed, attempting to capture...")

        gray_img = cv.cvtColor(frame.image, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray_img, CHESSBOARD_SIZE, None)

        if ret:
            img = frame.image.copy()
            cv.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
            cv.putText(img, "VALID PAIR FOUND", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv.imshow("Monocular Calibration Capture", img)
            cv.waitKey(1)

            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners_subpix)

            num_good_pts += 1
            print(f"Successfully saved image {num_good_pts}/{target_pts}.")

        else:
            print("--- CAPTURE FAILED: Could not detect corners in the view. Please reposition and try again. ---")

    elif key == ord('q'):
        print("User quit the capture process.")
        break


cv.destroyAllWindows()


if len(imgpoints) >= target_pts:
    print("Monocular calibration in progress...")
    ret, K, D, R, T = cv.calibrateCamera(
        objpoints, imgpoints, img_size,
        None, None,
        flags=cv.CALIB_RATIONAL_MODEL,
        criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )

    if ret:
        print("\nMonocular calibration successful!")
        print(f"\n Reprojection Error: {ret:.4f} pixels")

        save_path = os.path.join(os.path.dirname(__file__), f'mono_calib_{camera_name}.npz')
        np.savez(
            save_path,
            K=K, D=D, R=R, T=T,
            image_size=img_size
        )
        print(f"Calibration parameters saved to {save_path}")

    else:
        print("\nMonocular calibration failed. Please check the camera setup and try again.")

else:
    print("\nCannot collect enough valid images for monocular calibration. Please check the chessboard movement and camera setup.")
    sys.exit(1)