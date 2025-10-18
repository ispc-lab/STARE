import os
import sys
import time
import datetime

import dv_processing as dv
import cv2 as cv

from collections import OrderedDict

env_path = os.path.join(os.path.dirname(__file__), '..', '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'pytracking')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.utils.convert_event_img import convert_event_img_aedat


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

###################################################################################
    print("Initializing OSTrack tracker...")

    tracker = Tracker('ostrack', 'esot500mix', 'esot_500_20')
    params = tracker.get_parameters()
    params.debug = False
    ostrack = tracker.create_tracker(params)

    init_info = {
        'init_bbox': [175, 117, 33, 35],
    }
    template = tracker._read_image(os.path.dirname(__file__) + '/init/template/left_1.jpg')

    out = ostrack.initialize(template, init_info)
    if out is None:
        out = {}

    prev_output = OrderedDict(out)
    pred_bbox = init_info.get('init_bbox')
###################################################################################

    def process_time_window(events: dv.EventStore):
        if events is not None:
            nonlocal prev_output

            info = {}
            info['previous_output'] = prev_output

            event_rep = convert_event_img_aedat(events.numpy(), 'VoxelGridComplex')

            out = ostrack.track(event_rep, info)

            prev_output = OrderedDict(out)
            pred_bbox = out['target_bbox']

            # Draw output rectangle
            frame = visualizer.generateImage(events)
            op_x, op_y, op_w, op_h = pred_bbox
            cv.rectangle(
                frame,
                (int(op_x), int(op_y)),
                (int(op_x + op_w), int(op_y + op_h)),
                (0, 0, 255),
                2,
            )
            cv.imshow(window_name, frame)

            print("pred_bbox:", pred_bbox)


    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(
        datetime.timedelta(milliseconds=20),
        process_time_window
    )

    print("Start Tracking... Press 'q' to stop.")
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












