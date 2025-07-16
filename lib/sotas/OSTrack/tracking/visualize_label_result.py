import argparse
import os
import sys
import cv2
import glob
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.environment import env_settings


def visualize_bboxes(output_file, images_folder, output_folder):
    """
    Visualize ground truth and predicted bounding boxes
    
    Parameters:
    - output_file: Path to the prediction file
    - images_folder: Path to the image folder
    - output_folder: Path to the output image folder
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read output file
    output_bboxes = []
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:  # Ensure at least 4 values (x, y, w, h)
                x, y, w, h = map(int, parts[:4])
                output_bboxes.append((x, y, w, h))
    
    # Get all image files and sort them numerically
    image_files = glob.glob(os.path.join(images_folder, "*.jpg"))
    image_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Ensure the number of images matches the number of bboxes
    if len(image_files) != len(output_bboxes):
        print(f"Warning: Number of images ({len(image_files)}) does not match number of bboxes ({len(output_bboxes)})!")
        return
    
    # Define colors (BGR format)
    teal_color = (180, 180, 0)  # Teal (ground truth)
    red_color = (0, 0, 255)     # Red (output)
    
    # Process each image
    for i, (image_file, output_info) in enumerate(zip(image_files, output_bboxes)):
        # Read image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Unable to read image: {image_file}")
            continue
        
        # Parse output bbox info
        op_x, op_y, op_w, op_h = output_info
        
        # Draw output rectangle
        cv2.rectangle(img, (op_x, op_y), (op_x + op_w, op_y + op_h), red_color, 2)

        # Add "pred" label
        cv2.putText(img, "pred", (op_x, op_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2)
        
        # Save result image
        output_path = os.path.join(output_folder, os.path.basename(image_file))
        cv2.imwrite(output_path, img)
        
        print(f"Processed image {i + 1}/{len(image_files)}: {os.path.basename(image_file)}")
    
    print("Visualization complete! Results saved in:", output_folder)


def main():
    parser = argparse.ArgumentParser(description='Visualize OSTrack labelling results.')

    parser.add_argument('--sequence_name', type=str, default="fan5", help='sequence name')
    parser.add_argument('--tracker_name', type=str, default="ostrack", help='tracker name')
    parser.add_argument('--parameter_name', type=str, default="esot500mix", help='parameter name')

    # parser.add_argument('--images-folder', type=str, default="/media/blank/WD0/Datasets/STARE/data/ESOT500/500/airplane5/VoxelGridComplex", help='Path to the original image folder')
    # parser.add_argument('--save-dir', type=str, default="/media/blank/WD0/Datasets/STARE/lib/pytracking/pytracking/output/vis", help='Directory to save visualization results')

    args = parser.parse_args()

    env_variables = env_settings()

    txt_path = os.path.join(env_variables.results_path, f"{args.tracker_name}/{args.parameter_name}/{args.sequence_name}_500_w2ms.txt")
    images_folder = os.path.join(env_variables.esot500h_dir, f"500_w2ms/{args.sequence_name}/VoxelGridComplex")
    save_dir = os.path.join(env_variables.save_dir, f"vis/{args.tracker_name}/{args.parameter_name}/{args.sequence_name}")

    visualize_bboxes(txt_path, images_folder, save_dir)


if __name__ == "__main__":
    main()
