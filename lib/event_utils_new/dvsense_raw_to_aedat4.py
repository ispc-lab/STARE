import argparse
import time
import numpy as np

from dvsense_driver.raw_file_reader import RawFileReader
import dv_processing as dv


def main():
    parser = argparse.ArgumentParser(description='convert Dvsense events files from .raw to .aedat4 format')

    parser.add_argument('path_to_raw', type=str, help="Path to Dvsense events files in .raw format")
    parser.add_argument('output_aedat4_file', type=str, help='Output file path for the .aedat4 file')
    parser.add_argument('--acc_time', type=int, default=10000, help='Accumulation time when reading events (in microseconds)')

    args = parser.parse_args()

    file_path = args.path_to_raw
    output_aedat4_file = args.output_aedat4_file
    acc_time = args.acc_time

    file_reader = RawFileReader(file_path)
    file_reader.load_file()

    camera_name = 'DVSync'
    width = file_reader.get_width()
    height = file_reader.get_height()
    print(f"{camera_name} Camera resolution - Width: {width}, Height: {height}")

    start_timestamp = file_reader.get_start_timestamp()
    end_timestamp = file_reader.get_end_timestamp()
    print(f"Start timestamp: {start_timestamp}, End timestamp: {end_timestamp}")

    all_timestamps = []
    all_x = []
    all_y = []
    all_p = []

    start_time = time.time()
    while True:
        events = file_reader.get_n_events(acc_time)
        if events.size == 0:
            break

        all_timestamps.append(events['timestamp'])
        all_x.append(events['x'])
        all_y.append(events['y'])
        all_p.append(events['polarity'])

        file_reader.get_start_timestamp()
        current_pos_timestamp = file_reader.get_current_pos_timestamp()
        current_pos_event_num = file_reader.get_current_pos_event_num()
        print(f"Timestamp: {current_pos_timestamp}, Event #: {current_pos_event_num}, This chunk: {events.shape[0]}")

        if current_pos_timestamp + acc_time >= end_timestamp[1]:
            break

    end_time = time.time()
    print(f"All events have been read, took {end_time - start_time:.2f} seconds")

    # concatenate all lists into single arrays
    timestamps = np.concatenate(all_timestamps)
    x_coords = np.concatenate(all_x)
    y_coords = np.concatenate(all_y)
    polarities = np.concatenate(all_p)

    sort_idx = np.argsort(timestamps)
    timestamps = timestamps[sort_idx]
    x_coords = x_coords[sort_idx]
    y_coords = y_coords[sort_idx]
    polarities = polarities[sort_idx]

    num_events = len(timestamps)
    print(f"Total Events Number: {num_events}")

    # save to .aedat4 file
    config = dv.io.MonoCameraWriter.EventOnlyConfig(camera_name, (width, height))
    writer = dv.io.MonoCameraWriter(output_aedat4_file, config)
    event_store = dv.EventStore()

    print(f"Writing to {output_aedat4_file}...")
    for i in range(num_events):
        event_store.push_back(
            int(timestamps[i]),
            int(x_coords[i]),
            int(y_coords[i]),
            bool(polarities[i])
        )

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - end_time
            print(f"\r Having processed {i + 1}/{num_events} events, elapsed time: {elapsed:.2f} seconds", end="")

    writer.writeEvents(event_store)
    print(f"\n Conversion completed, .aedat4 file saved to: {output_aedat4_file}")


if __name__ == '__main__':
    main()
