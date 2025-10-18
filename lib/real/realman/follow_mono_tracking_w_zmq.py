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
from scipy.interpolate import interp1d
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Robotic_Arm.rm_robot_interface import *


def point2d_collector(
        point2d_buffer: list,
        stop_event: threading.Event
):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 2)  # 2 ms timeout for recv

    # socket.bind("tcp://192.168.110.7:5555")
    socket.bind("ipc:///tmp/track_pipe.ipc")

    print(f"2D point collector started.")
    while not stop_event.is_set():
        try:
            msg = socket.recv(copy=False)
            buf = memoryview(msg.buffer)
            x, y = struct.unpack_from('<2f', buf)
            point2d_buffer[0] = (x, y)

            # print("point2d = ", point2d)

        except zmq.Again:
            # time.sleep(0.001)
            continue

    print(f"2D point collector stopped.")


def arm_follow_target(
        point2d_buffer: list,
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

    # initialize the arm to a known state
    curr_state = robot.rm_get_current_arm_state()
    curr_pose = curr_state[1]['pose']
    target_pose = curr_pose.copy()
    target_pose[0] = -0.3
    target_pose[1] = -0.1
    target_pose[2] = 0.21
    target_pose[3:] = [-3.093, 0.047, -0.54]

    params = rm_inverse_kinematics_params_t(
        q_in=curr_state[1]['joint'],
        q_pose=target_pose,
        flag=1,
    )
    q_out_init = algo_handle.rm_algo_inverse_kinematics(params)
    robot.rm_movej(q_out_init[1], 20, 0, 0, True)

    # retarget the pixel position to the arm's initial position
    z_real = np.array([-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    z_2d = np.array([187, 175, 161, 150, 138, 125, 112, 100, 88, 76, 64])
    f = interp1d(z_2d, z_real, kind='linear')

    # last_time = time.perf_counter()

    print("Follow target thread started.")
    while not stop_event.is_set():

        point2d = point2d_buffer[0]
        if point2d is not None:

            # print("point2d = ", point2d)
            x, z = point2d

            z += 10
            z = np.clip(z, 64, 187)
            z = f(z)  # Interpolate Z value from 2D point
            z = np.clip(z, -0.0, 0.3)

            curr_state = robot.rm_get_current_arm_state()
            curr_pose = curr_state[1]['pose']
            target_pose = curr_pose.copy()
            target_pose[0] = -0.3
            target_pose[1] = -0.1
            target_pose[2] = z
            target_pose[3:] = [-3.093, 0.047, -0.54]

            params = rm_inverse_kinematics_params_t(
                q_in=curr_state[1]['joint'],
                q_pose=target_pose,
                flag=1,
            )
            q_out = algo_handle.rm_algo_inverse_kinematics(params)

            # robot.rm_movej_follow(q_out[1])

            if x > 20:
                # print(robot.rm_movej(q_out[1], 20, 0, 0, True))
                robot.rm_movej_follow(q_out[1])
                # print(robot.rm_movej_canfd(q_out[1], False, 0, 1, 50))
            else:
                # print(robot.rm_movej(q_out_init[1], 20, 0, 0, True))
                robot.rm_movej_follow(q_out_init[1])
                # print(robot.rm_movej_canfd(q_out_init[1], False, 0, 1, 50))

            # if 290 < x < 315:
                # target_joint_angles = q_out[1]
                # print("Target Joint Angles: ", target_joint_angles)
                # target_joint_angles[4] += 20

                # print(robot.rm_movej(target_joint_angles, 20, 0, 0, True))
                # print(robot.rm_movej_follow(target_joint_angles))
                # print(robot.rm_movej_canfd(target_joint_angles, False, 0, 1, 50))

                # time.sleep(0.005)

                # target_joint_angles = q_out[1]
                # print("Target Joint Angles: ", target_joint_angles)
                # target_joint_angles[4] -= 35

                # print(robot.rm_movej(target_joint_angles, 20, 0, 0, True))
                # print(robot.rm_movej_follow(target_joint_angles))
                # print(robot.rm_movej_canfd(target_joint_angles, False, 0, 1, 50))

                # time.sleep(0.005)

                # print(robot.rm_movej(q_out[1], 20, 0, 0, True))
                # print(robot.rm_movej_follow(q_out[1]))
                # print(robot.rm_movej_canfd(q_out[1], False, 0, 1, 50))

                # exit(1)

            # curr_time = time.perf_counter()
            # elapsed_time = curr_time - last_time
            # last_time = curr_time
            # print(f"2D Points FPS:", 1 / elapsed_time)

            time.sleep(0.005)

        else:
            print("Waiting for 2D point data...")
            time.sleep(0.001)

    print("Follow target thread stopped.")


def main():

    # ------ 2D Points Collecting and Arm Follow Threads ------
    point2d_buffer = [None]
    stop_event = threading.Event()

    receiving_point2d_thread = threading.Thread(
        target=point2d_collector,
        args=(point2d_buffer, stop_event)
    )
    receiving_point2d_thread.start()

    arm_follow_target_thread = threading.Thread(
        target=arm_follow_target,
        args=(point2d_buffer, stop_event)
    )
    arm_follow_target_thread.start()

    # main loop for tracking and visualization
    while True:
        point_2d = point2d_buffer[0]

        if point_2d is None:
            print("Waiting for 2D point data...")
            time.sleep(0.001)
            continue

        time.sleep(0.001)
        # print("2D Point:", point_2d[0], point_2d[1])

    print("Program exited.")


if __name__ == '__main__':
    main()