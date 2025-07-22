import os
import sys
import time
import datetime

import numpy as np
import dv_processing as dv
import cv2 as cv


# Parameters for the chessboard calibration
CHESSBOARD_SIZE = (8, 5)
SQUARE_SIZE_MM = 25

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM


def extract_corners_from_live_dvs(dvs_capture, frame_num=100):

    dvs_capture.setDavisFrameInterval(datetime.timedelta(milliseconds=33))
    img_size = dvs_capture.getFrameResolution()

    cnt = 0
    img_points = []
    while cnt < frame_num and dvs_capture.isRunning():
        rgb_frame = dvs_capture.getNextFrame()
        if rgb_frame is not None:
            gray_frame = cv.cvtColor(rgb_frame.image, cv.COLOR_BGR2GRAY)
            cv.imshow("Live DVS Camera - GREY", gray_frame)

            ret, corners = cv.findChessboardCorners(gray_frame, CHESSBOARD_SIZE, None)
            if ret:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_subpix = cv.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_subpix)
                cnt += 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    return img_points, img_size


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

num_good_views = 20

print("Capturing chessboard corners from [left] live DVS cameras...")
img_points_left, img_size_left = extract_corners_from_live_dvs(capture.left, num_good_views)

print("Capturing chessboard corners from [right] live DVS cameras...")
img_points_right, img_size_right = extract_corners_from_live_dvs(capture.right, num_good_views)

print("Stereo calibration in progress...")
objpoints = [objp] * num_good_views
ret, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(
    objpoints, img_points_left, img_points_right,
    None, None, None, None,
    imageSize=img_size_left,
    flags=cv.CALIB_RATIONAL_MODEL,
    criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
)

if ret:
    print("\nStereo calibration successful!")
    save_path = os.path.dirname(__file__) + '/stereo_calibration.npz'
    np.savez(
        save_path,
        K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T,
        image_size=img_size_left
    )
    print(f"Calibration parameters saved to {save_path}")

else:
    print("\nStereo calibration failed. Please check the camera setup and try again.")
