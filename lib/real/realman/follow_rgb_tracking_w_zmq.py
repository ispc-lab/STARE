import os
import struct
import sys
import time
import datetime
import threading
import zmq
import json
import pickle

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Robotic_Arm.rm_robot_interface import *


def point3d_collector(
        point3d_buffer: list,
        stop_event: threading.Event
):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)

    # socket.bind("tcp://192.168.110.7:5555")
    socket.bind("ipc:///tmp/track_pipe.ipc")

    print(f"3D point collector started.")
    while not stop_event.is_set():
        try:
            data = socket.recv(flags=zmq.NOBLOCK)
            point3d = struct.unpack('<3f', data)
            point3d_buffer[0] = point3d

        except zmq.Again:
            time.sleep(0.001)
            continue

    print(f"3D point collector stopped.")


def arm_follow_target(
        point3d_buffer: list,
        stop_event: threading.Event
):
    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = robot.rm_create_robot_arm("192.168.110.118", 8080)
    print("Arm ID：", handle.id)

    # print(robot.rm_set_collision_state(8))
    # print(robot.rm_set_arm_max_line_speed(0.05))
    # print(robot.rm_set_arm_max_line_acc(0.1))
    # print(robot.rm_set_arm_max_angular_speed(0.2))
    # print(robot.rm_set_arm_max_angular_acc(1))

    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E  # RM_65机械臂
    force_type = rm_force_type_e.RM_MODEL_RM_B_E  # 标准版
    algo_handle = Algo(arm_model, force_type)
    algo_handle.rm_algo_set_redundant_parameter_traversal_mode(False)

    # kp = [None] * 4
    # for i in range(4):
    #     with open(f"./pp/kp_r{i}_c0.pkl", 'rb') as f:
    #         kp[i] = pickle.load(f)
    #
    # target_z = np.array([
    #     138.06,
    #     328.43,
    #     538.76,
    #     757.81,
    # ])

    last_time = time.perf_counter()

    print("Follow target thread started.")
    while not stop_event.is_set():
        point3d = point3d_buffer[0]
        if point3d is not None:
            x, y, z = point3d

            # diffs = np.abs(target_z - z)
            # closest_idx = np.argmin(diffs)

            # print(f"Moving to position with target Z={target_z[closest_idx]:.2f} mm")
            # print(robot.rm_movej_canfd(kp[closest_idx][1]["joint"], True, 0, 1, 50))
            # print(robot.rm_movej(kp[closest_idx][1]["joint"], 20, 0, 0, True))
            # print(robot.rm_movej_follow(kp[closest_idx][1]["joint"]))

            z = z / 1000  # Convert mm to m
            y = y / 1000  # Convert mm to m
            x = x / 1000  # Convert mm to m

            curr_state = robot.rm_get_current_arm_state()
            curr_pose = curr_state[1]['pose']
            target_pose = curr_pose.copy()
            target_pose[0] = -0.3
            target_pose[1] = -0.1
            target_pose[2] = 0.2 + 0.05
            target_pose[3:] = [-3.093, 0.047, -0.54]

            # target_pose[2] += z  # Adjust Z position based on the received point
            target_pose[2] += - (y + 0.13)  # Adjust Z position based on the received point
            if target_pose[2] > 0.37:
                target_pose[2] = 0.37
            elif target_pose[2] < 0.22:
                target_pose[2] = 0.22

            # target_pose[1] += x  # Adjust X position based on the received point
            # if target_pose[1] > 0:
            #     target_pose[1] = 0
            # elif target_pose[1] < -0.2:
            #     target_pose[1] = -0.2

            params = rm_inverse_kinematics_params_t(
                q_in=curr_state[1]['joint'],
                q_pose=target_pose,
                flag=1,
            )
            q_out = algo_handle.rm_algo_inverse_kinematics(params)

            # print(robot.rm_movej(q_out[1], 20, 0, 0, True))
            print(robot.rm_movej_follow(q_out[1]))
            # print(robot.rm_movej_canfd(q_out[1], False, 0, 1, 50))

            curr_time = time.perf_counter()
            elapsed_time = curr_time - last_time
            last_time = curr_time
            print(f"3D Points FPS:", 1 / elapsed_time)

            time.sleep(0.005)

        else:
            print("Waiting for 3D point data...")
            time.sleep(0.001)

    print("Follow target thread stopped.")


def main():

    # ------ 3D Points Collecting and Arm Follow Threads ------
    point3d_buffer = [None]
    stop_event = [threading.Event(), threading.Event()]

    receiving_point3d_thread = threading.Thread(
        target=point3d_collector,
        args=(point3d_buffer, stop_event[0])
    )
    receiving_point3d_thread.start()

    arm_follow_target_thread = threading.Thread(
        target=arm_follow_target,
        args=(point3d_buffer, stop_event[0])
    )
    arm_follow_target_thread.start()


    # ------ Visualization ------
    plt.ion()
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Real-time 3D Tracking and X-Z Plane View with Received Data', fontsize=16)

    # --- left subplot: 3D view ---
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    scatter_3d = ax_3d.scatter([], [], [], c='red', s=150, label='Current Position')

    ax_3d.set_xlabel('X (mm)')
    ax_3d.set_ylabel('Y (mm)')
    ax_3d.set_zlabel('Z (mm)')
    ax_3d.set_title('3D View')

    ax_3d.set_xlim(-1000, 1000)
    ax_3d.set_ylim(-1000, 1000)
    ax_3d.set_zlim(-1000, 1000)
    ax_3d.legend()

    # --- right subplot: 2D X-Z plane view (front-end view) ---
    ax_2d = fig.add_subplot(1, 2, 2)
    scatter_xz = ax_2d.scatter([], [], c='blue', s=150, label='Current X-Z Position')

    ax_2d.set_xlabel('X (mm)')
    ax_2d.set_ylabel('Z (mm)')
    ax_2d.set_title('Top-Down View (X-Z Plane)')

    ax_2d.set_xlim(-1500, 1500)
    ax_2d.set_ylim(-1500, 1500)

    ax_2d.set_aspect('equal', adjustable='box')
    ax_2d.grid(True)
    ax_2d.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # main loop for tracking and visualization
    while True:
        point_3d = point3d_buffer[0]

        if point_3d is None:
            print("Waiting for 3D point data...")
            time.sleep(0.001)
            continue

        # update the scatter plot with the new 3D point
        scatter_3d._offsets3d = ([point_3d[0]], [point_3d[1]], [point_3d[2]])
        scatter_xz.set_offsets([point_3d[0], point_3d[2]])

        plt.draw()
        plt.pause(0.001)

        if not plt.fignum_exists(fig.number):
            print("Plot window closed. Exiting.")
            break

    plt.ioff()
    plt.show()

    print("Program exited.")


if __name__ == '__main__':
    main()