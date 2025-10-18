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


def main():
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

    frame_buffer = [None, None]

    collector_thread = threading.Thread(
        target=frame_collector,
        args=(capture, camera_id, frame_buffer),
        name="RGB-Frame-Collector"
    )
    collector_thread.start()

    output_rgb_file = os.path.dirname(__file__) + f'/init/template/pingpang_{camera_name}_rgb.jpg'

    while capture.isRunning():
        frame = frame_buffer[camera_id]
        if frame is None:
            continue

        cv.imshow(camera_name, frame.image)

        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):
            print("\n's' key pressed, attempting to capture...")
            img = frame.image.copy() # BGR
            cv.imwrite(output_rgb_file, img)
            print(f"Captured and saved RGB image to {output_rgb_file}")

        elif key == ord('q'):
            print("User quit the capture process.")
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()












