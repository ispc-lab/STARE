import os

def convert_single_object_to_mot(input_filepath, output_filepath, object_id=1, class_id=1):
    """
    Convert a single object tracking format (x, y, width, height) to MOT format.

    Args:
        input_filepath (str): Input file path containing lines of format: x y width height.
        output_filepath (str): Output file path to save the converted MOT format data.
        object_id (int): Tracking object ID, default is 1 (for single object tracking).
        class_id (int): Tracking class ID, default is 1 (commonly used for pedestrians in MOT).
    """
    mot_lines = []
    frame_id = 1  # Frame ID, Start from 1

    try:
        with open(input_filepath, 'r') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue # Skip empty lines

                parts = line.split()
                if len(parts) != 4:
                    print(f"Note: Skipping line due to incorrect format. Expected 4 values, got {len(parts)}.")
                    continue

                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    width = float(parts[2])
                    height = float(parts[3])

                except ValueError:
                    print(f"Note: Ensure the input file contains valid numeric values for x, y, width, and height.")
                    continue

                # MOT Format: frame_id, object_id, x, y, width, height, conf, class_id, visibility, attribute1, attribute2, attribute3
                # for conf, visibility and attributes: we use default values
                conf = 1.0 # set 1.0 for Ground Truth or certain tracking results
                visibility = 1.0 # Default visible
                attr1, attr2, attr3 = -1, -1, -1 # Default Placeholder

                mot_line = f"{frame_id},{object_id},{x},{y},{width},{height},{conf},{class_id},{visibility},{attr1},{attr2},{attr3}"
                mot_lines.append(mot_line)
                frame_id += 1

        with open(output_filepath, 'w') as outfile:
            for mot_line in mot_lines:
                outfile.write(mot_line + '\n')

        print(f"Successfully converted {len(mot_lines)} lines in '{input_filepath}' to MOT format.")
        print(f"Saved MOT format data to {output_filepath}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found. Please check the path.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # --- set input_file and output_file to your specific paths ---
    # your original annotation file
    input_file = "fan4_500_w2ms.txt"
    # converted MOT format file
    output_file = "output_mot_labels.txt"

    # --- run the conversion ---
    convert_single_object_to_mot(input_file, output_file)
