import os
import sys
import time
import datetime

import dv_processing as dv
import cv2 as cv


def main():
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

    resolution = capture.left.getEventResolution()
    visualizer = dv.visualization.EventVisualizer(resolution)

    capture.left.setDavisFrameInterval(datetime.timedelta(milliseconds=10))
    capture.right.setDavisFrameInterval(datetime.timedelta(milliseconds=10))

    while capture.left.isRunning() and capture.right.isRunning():
        events_left = capture.left.getNextEventBatch()
        left_rgb_frame = capture.left.getNextFrame()

        events_right = capture.right.getNextEventBatch()
        right_rgb_frame = capture.right.getNextFrame()

        if events_left is not None and events_right is not None:
            left_event_frame = visualizer.generateImage(events_left)
            cv.imshow("Left-Event", left_event_frame)

            right_event_frame = visualizer.generateImage(events_right)
            cv.imshow("Right-Event", right_event_frame)

        else:
            time.sleep(0.0001)

        if left_rgb_frame is not None:
            cv.imshow("Left-RGB", left_rgb_frame.image)
            left_gray_frame = cv.cvtColor(left_rgb_frame.image, cv.COLOR_BGR2GRAY)
            cv.imshow("Left-GREY", left_gray_frame)

        if right_rgb_frame is not None:
            cv.imshow("Right-RGB", right_rgb_frame.image)
            right_gray_frame = cv.cvtColor(right_rgb_frame.image, cv.COLOR_BGR2GRAY)
            cv.imshow("Right-GREY", right_gray_frame)

        if cv.waitKey(2) & 0xFF == ord('q'):
            print("Camera capture stopped.")
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()