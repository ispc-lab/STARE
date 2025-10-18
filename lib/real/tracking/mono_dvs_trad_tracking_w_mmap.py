import os
import sys
import time
import datetime
import threading
import mmap
import struct

import numpy as np
import dv_processing as dv
import cv2 as cv

from collections import OrderedDict
from readerwriterlock import rwlock


env_path = os.path.join(os.path.dirname(__file__), '..', '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'pytracking')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.utils.convert_event_img import convert_event_img_aedat


SAMPLING_WINDOW_MS = 20


# def tracking_res_vis(
#         events_buffer: dv.EventStore,
#         visualizer: dv.visualization.EventVisualizer,
#         pred_bbox: list,
#         stop_event: threading.Event,
# ):
#     print("Tracking result visualization thread has started.")
#
#     while not stop_event.is_set():
#         op_x, op_y, op_w, op_h = pred_bbox[0]
#         obj_rect = [
#             (int(op_x), int(op_y)),
#             (int(op_x + op_w), int(op_y + op_h)),
#         ]
#
#         latest_ts = events_buffer.getHighestTime() if len(events_buffer) > 0 else 0
#         cutoff_ts = int(latest_ts - SAMPLING_WINDOW_MS * 1e3)
#         events_buffer = events_buffer.sliceTime(cutoff_ts)
#
#         event_frame = visualizer.generateImage(events_buffer)
#         cv.rectangle(
#             event_frame,
#             obj_rect[0],
#             obj_rect[1],
#             (0, 0, 255),
#             2,
#         )
#         event_frame = cv.resize(event_frame, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)
#         cv.imshow("DVS Tracking Result", event_frame)
#
#         time.sleep(0.01)
#
#         if cv.waitKey(2) & 0xFF == ord('q'):
#             print("Tracking result visualization stopped.")
#             stop_event.set()
#             break
#
#     print("Tracking result visualization thread has stopped.")
#     cv.destroyAllWindows()


def main():

    # ------ Camera Initialization ------
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
        # 'init_bbox': [164, 103, 14, 14],
        'init_bbox': [120, 133, 29, 31],
    }
    template = tracker._read_image(os.path.dirname(__file__) + '/init/template/right_fast_1.jpg')

    out = ostrack.initialize(template, init_info)
    if out is None:
        out = {}

    prev_output = OrderedDict(out)
    pred_bbox = [init_info.get('init_bbox')]
    op_x, op_y, op_w, op_h = pred_bbox[0]
    # obj_center = [op_x + op_w / 2, op_y + op_h / 2]

###################################################################################

    ##################################################################
    SHM_PATH = "/dev/shm/point2d_shm"
    SIZE = 20  # seq(uint64) + x(float32) + y(float32) + pad(4)

    fd = os.open(SHM_PATH, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
    os.ftruncate(fd, SIZE)
    buf = mmap.mmap(fd, SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
    os.close(fd)

    seq = 0
    pack = struct.Struct("<QffI")  # 8+4+4+4=20，可换 "<Qff" 并改大小
    ##################################################################

    last_time = time.perf_counter()
    events_buffer = dv.EventStore()

    def process_time_window(events: dv.EventStore):
        if events is not None:
            nonlocal prev_output, pred_bbox
            nonlocal last_time
            nonlocal events_buffer
            nonlocal seq
            nonlocal pack

            events_buffer.add(events)

            info = {}
            info['previous_output'] = prev_output

            event_rep = convert_event_img_aedat(events.numpy(), 'VoxelGridComplex')

            out = ostrack.track(event_rep, info)

            prev_output = OrderedDict(out)
            pred_bbox[0] = out['target_bbox']

            curr_time = time.perf_counter()
            elapsed_time = curr_time - last_time
            last_time = curr_time
            print(f"TRAD FPS:", 1 / elapsed_time)

            op_x, op_y, op_w, op_h = pred_bbox[0]
            obj_center = [op_x + op_w / 2, op_y + op_h / 2]
            if obj_center is not None:
                seq += 1
                buf.seek(0)
                buf.write(pack.pack(seq, obj_center[0], obj_center[1], 0))

            # Draw output rectangle
            event_frame = visualizer.generateImage(events)
            cv.rectangle(
                event_frame,
                (int(op_x), int(op_y)),
                (int(op_x + op_w), int(op_y + op_h)),
                (0, 0, 255),
                2,
            )
            # event_frame = cv.resize(event_frame, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)
            cv.imshow(window_name, event_frame)

            # print("pred_bbox:", pred_bbox[0])

    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(
        datetime.timedelta(milliseconds=SAMPLING_WINDOW_MS),
        process_time_window
    )

    # stop_event = threading.Event()
    # tracking_thread = threading.Thread(
    #     target=tracking_res_vis,
    #     args=(events_buffer, visualizer, pred_bbox, stop_event)
    # )
    # tracking_thread.start()

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
    print("Program exited.")


if __name__ == '__main__':
    main()