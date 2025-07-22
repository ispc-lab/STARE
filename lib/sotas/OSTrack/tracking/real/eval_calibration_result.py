import os
import sys
import time

import cv2 as cv
import numpy as np
import dv_processing as dv
import open3d as o3d


EVENT_COUNT_THRESHOLD = 10

def find_feature_center(events):
    """
    A simple feature detector:
    when the number of events in the packet exceeds a threshold, compute the centroid
    """
    if events is not None and len(events) > EVENT_COUNT_THRESHOLD:
        coords = events.coordinates()
        centroid = np.mean(coords, axis=0)

        return centroid

    return None

def triangulate(pt1_undistorted, pt2_undistorted, p1_matrix, p2_matrix):
    points_4d_hom = cv.triangulatePoints(p1_matrix, p2_matrix, pt1_undistorted.T, pt2_undistorted.T)
    points_3d = points_4d_hom / points_4d_hom[3]
    return points_3d[:3].flatten()


# --- Load stereo calibration parameters from a file ---
try:
    save_path = os.path.dirname(__file__) + '/stereo_calibration.npz'
    # save_path = './stereo_calibration.npz'
    calib = np.load(save_path)
    K1, D1 = calib['K1'], calib['D1']
    K2, D2 = calib['K2'], calib['D2']
    R, T = calib['R'], calib['T']
    image_size = calib['image_size']
    print("Successfully loaded calibration parameters:")

except FileNotFoundError:
    print("Error: Calibration file not found. Please run stereo_dvs_calibrate.py first.")
    sys.exit(1)

# --- Create projection matrices for both cameras ---
# P1 = K1 * [I|0]
P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
# P2 = K2 * [R|T]
P2 = K2 @ np.hstack([R, T])

# --- Real-time inference setup ---
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

# --- 初始化 Open3D 可视化窗口 ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Trajectory Visualization")

# 创建一个坐标系来表示相机（原点）
# +X为红色, +Y为绿色, +Z为蓝色
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])
vis.add_geometry(coordinate_frame)

# 用于存储轨迹点和绘制线条
trajectory_points = []
line_set = o3d.geometry.LineSet()
vis.add_geometry(line_set) # 将空的线集添加到场景中

# 设置渲染选项，使背景为黑色
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])


# --- Main loop for real-time stereo DVS processing ---
print("\nStarting real-time processing....")
print("Close the 3D window to exit.")
while capture.left.isRunning() and capture.right.isRunning() and vis.poll_events():
    events_left = capture.left.getNextEventBatch()
    events_right = capture.right.getNextEventBatch()

    if events_left is not None and events_right is not None:
        left_event_frame = visualizer.generateImage(events_left)
        left_center = find_feature_center(events_left)

        right_event_frame = visualizer.generateImage(events_right)
        right_center = find_feature_center(events_right)

        if left_center is not None and right_center is not None:
            # get the centroids of the detected features, [x, y] format
            pt_left = left_center.reshape(1, 1, 2).astype(np.float32)
            pt_right = right_center.reshape(1, 1, 2).astype(np.float32)

            # remove distortion using the calibration parameters
            pt_left_undistorted = cv.undistortPoints(pt_left, K1, D1)
            pt_right_undistorted = cv.undistortPoints(pt_right, K2, D2)

            # perform triangulation to get the 3D point
            point_3d = triangulate(pt_left_undistorted, pt_right_undistorted, P1, P2)

            print(f"Matching features detected, 3D Coordinates (mm): X={point_3d[0]:.2f}, Y={point_3d[1]:.2f}, Z={point_3d[2]:.2f}")
            print(f"Left Feature: {left_center}, Right Feature: {right_center}")

            trajectory_points.append(point_3d)
            if len(trajectory_points) > 1:
                # Open3D需要特定的数据类型
                points_o3d = o3d.utility.Vector3dVector(trajectory_points)

                # 创建连接线段
                # line_indices[i] = [i, i+1] 连接第i个点和第i+1个点
                line_indices = [[i, i + 1] for i in range(len(trajectory_points) - 1)]

                # 为线段设置颜色 (例如，从蓝色渐变到红色)
                colors = [[i / len(line_indices), 0, 1 - i / len(line_indices)] for i in range(len(line_indices))]

                # 更新线集的点、线和颜色
                line_set.points = points_o3d
                line_set.lines = o3d.utility.Vector2iVector(line_indices)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                # 通知可视化器几何体已更新
                vis.update_geometry(line_set)

        # Display all
        cv.imshow("Left Event Stream", left_event_frame)
        cv.imshow("Right Event Stream", right_event_frame)
        vis.update_renderer()

    else:
        time.sleep(0.0001)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
print("Program exited.")
