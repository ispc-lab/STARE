import os
import sys
import time
import datetime

import dv_processing as dv
import cv2 as cv


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

    resolution = capture.getEventResolution()
    visualizer = dv.visualization.EventVisualizer(resolution)

    output_aedat4_file = os.path.dirname(__file__) + f'/init/aedat4/pingpong_{camera_name}_tracking_init.aedat4'
    config = dv.io.MonoCameraWriter.EventOnlyConfig(
        cameraName=camera_name,
        resolution=resolution,
    )
    writer = dv.io.MonoCameraWriter(output_aedat4_file, config)
    event_store = dv.EventStore()
    cnt, cnt_max = 0, 1000

    def process_time_window(events: dv.EventStore):
        if events is not None:
            frame = visualizer.generateImage(events)
            cv.imshow(window_name, frame)

            event_store.add(events)

            nonlocal cnt, cnt_max
            cnt += 1
            if cnt > cnt_max:
                print(f"Writing events to {output_aedat4_file}")
                writer.writeEvents(event_store)
                sys.exit(0)

    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(
        datetime.timedelta(milliseconds=20),
        process_time_window
    )

    while capture.isRunning():
        events = capture.getNextEventBatch()

        if events is not None:
            slicer.accept(events)
        else:
            time.sleep(0.0001)

        if cv.waitKey(2) & 0xFF == ord('q'):
            print("Camera capture stopped.")
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()












