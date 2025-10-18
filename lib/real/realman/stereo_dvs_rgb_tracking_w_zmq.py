import os
import struct
import sys
import time
import datetime
import threading
import zmq
import json

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():

    # ------ ZeroMQ Initialization ------
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1)

    # socket.bind("tcp://192.168.110.7:5555")
    socket.bind("ipc:///tmp/track_pipe.ipc")


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
        try:
            data = socket.recv(flags=zmq.NOBLOCK)
            point_3d = struct.unpack('<3f', data)

            print("Received point_3d:", point_3d)
            print(f"X={point_3d[0]:.2f}, Y={point_3d[1]:.2f}, Z={point_3d[2]:.2f}")

        except zmq.Again:
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