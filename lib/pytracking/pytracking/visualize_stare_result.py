'''
PKL Conversion and Visualization Tool
Converts tracking results from PKL format to TXT format and visualizes them.
'''

import argparse
import pickle
import os
import sys
import cv2
import glob
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.environment import env_settings

def find_last_pred(gt_t, pred_raw):
    """Find the last prediction for a given timestamp"""
    pred_timestamps = pred_raw['out_timestamps']
    pred_timestamps[0] = 0
    gt_t = gt_t * 1e6
    
    last_pred_idx = np.searchsorted(pred_timestamps, gt_t) - 1
    pred_results = pred_raw['results_raw']
    pred_last_result = pred_results[last_pred_idx]
    pred_last_time = pred_timestamps[last_pred_idx]
    assert pred_last_time <= gt_t
    
    return pred_last_result

def stream_eval(gt_anno_t, raw_result):
    """Convert raw results to final predictions"""
    pred_final = []
    for line_idx, line in enumerate(gt_anno_t):
        gt_t = line[0]
        pred_label = find_last_pred(gt_t, raw_result)
        pred_bbox = pred_label
        pred_final.append(pred_bbox)
    return pred_final

def convert_pkl_to_txt(pkl_path, txt_path, gt_anno_path):
    """
    Convert .pkl file to .txt file
    
    Parameters:
    - pkl_path: Path to the .pkl file
    - txt_path: Path to the output .txt file
    - gt_anno_path: Path to the ground truth timestamps file
    
    Returns:
    - List of predicted bounding boxes
    """
    # Load .pkl file
    print(f"Loading file: {pkl_path}")
    raw_result = pickle.load(open(pkl_path, 'rb'))
    
    # Load ground truth timestamps
    print(f"Loading ground truth: {gt_anno_path}")
    gt_anno_t = np.loadtxt(gt_anno_path)
    
    # Perform evaluation and get results
    pred_final = stream_eval(gt_anno_t, raw_result)
    
    # Save as txt
    print(f"Saving results to: {txt_path}")
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    np.savetxt(txt_path, pred_final, fmt='%d', delimiter='\t')
    print("Conversion complete!")
    
    return pred_final

def visualize_bboxes(gt_file, output_file, images_folder, output_folder):
    """
    Visualize ground truth and predicted bounding boxes
    
    Parameters:
    - gt_file: Path to the ground truth file
    - output_file: Path to the prediction file
    - images_folder: Path to the image folder
    - output_folder: Path to the output image folder
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read ground truth file
    gt_bboxes = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure at least 5 values (time, x, y, w, h)
                time = float(parts[0])
                x, y, w, h = map(int, parts[1:5])
                gt_bboxes.append((time, x, y, w, h))
    
    # Read output file
    output_bboxes = []
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:  # Ensure at least 4 values (x, y, w, h)
                x, y, w, h = map(int, parts[:4])
                output_bboxes.append((x, y, w, h))
    
    # Ensure the number of bounding boxes in both files is equal
    if len(gt_bboxes) != len(output_bboxes):
        print(f"Warning: Number of ground truth bboxes ({len(gt_bboxes)}) does not match number of output bboxes ({len(output_bboxes)})!")
        return
    
    # Get all image files and sort them numerically
    image_files = glob.glob(os.path.join(images_folder, "*.jpg"))
    image_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Ensure the number of images matches the number of bboxes
    if len(image_files) != len(gt_bboxes):
        print(f"Warning: Number of images ({len(image_files)}) does not match number of bboxes ({len(gt_bboxes)})!")
        return
    
    # Define colors (BGR format)
    teal_color = (180, 180, 0)  # Teal (ground truth)
    red_color = (0, 0, 255)     # Red (output)
    
    # Process each image
    for i, (image_file, gt_info, output_info) in enumerate(zip(image_files, gt_bboxes, output_bboxes)):
        # Read image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Unable to read image: {image_file}")
            continue
        
        # Parse ground truth bbox info
        _, gt_x, gt_y, gt_w, gt_h = gt_info
        
        # Parse output bbox info
        op_x, op_y, op_w, op_h = output_info
        
        # Draw ground truth rectangle
        cv2.rectangle(img, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), teal_color, 2)
        
        # Draw output rectangle
        cv2.rectangle(img, (op_x, op_y), (op_x + op_w, op_y + op_h), red_color, 2)
        
        # Add "gt" label
        cv2.putText(img, "gt", (gt_x, gt_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, teal_color, 2)
        
        # Add "pred" label
        cv2.putText(img, "pred", (op_x, op_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2)
        
        # Save result image
        output_path = os.path.join(output_folder, os.path.basename(image_file))
        cv2.imwrite(output_path, img)
        
        print(f"Processed image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
    
    print("Visualization complete! Results saved in:", output_folder)

def main():
    parser = argparse.ArgumentParser(description='Convert PKL file to TXT file and visualize.')
    parser.add_argument('--sequence_name', type=str, default="bag6", help='Environment setting')
    parser.add_argument('--tracker_name', type=str, default="dimp", help='Environment setting')
    parser.add_argument('--parameter_name', type=str, default="dimp50_finetuned", help='Environment setting')
    parser.add_argument('--run_id', type=int, default=14, help='Environment setting')
    parser.add_argument('--images-folder', type=str, default="/root/autodl-tmp/ESOT500/500/airplane5/VoxelGridComplex", help='Path to the original image folder')
    parser.add_argument('--save-dir', type=str, default="/root/autodl-tmp/ESOT500/test_jiechu", help='Directory to save visualization results')
    
    args = parser.parse_args()
    env_variables = env_settings()

    # Generate temporary TXT file path
    txt_path = os.path.join(env_variables.results_path_rt_final, f"{args.tracker_name}/{args.parameter_name}/{args.run_id}/{args.sequence_name}_s.txt")
    pkl_path = os.path.join(env_variables.results_path_rt, f"{args.tracker_name}/{args.parameter_name}/{args.run_id}/{args.sequence_name}_s.pkl")
    gt_path = os.path.join(env_variables.esot500_dir, f"anno_t/{args.sequence_name}.txt")
    images_folder = os.path.join(env_variables.esot500_dir, f"500/{args.sequence_name}/VoxelGridComplex")
    save_dir = os.path.join(args.save_dir, f"{args.tracker_name}/{args.parameter_name}/{args.run_id}/{args.sequence_name}")

    # Step 1: Convert PKL to TXT
    convert_pkl_to_txt(pkl_path, txt_path, gt_path)
    
    # Step 2: Visualization
    visualize_bboxes(gt_path, txt_path, images_folder, save_dir)

if __name__ == "__main__":
    main()