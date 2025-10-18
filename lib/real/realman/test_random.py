import pickle
from time import sleep

from Robotic_Arm.rm_robot_interface import *


robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("192.168.110.119", 8080)
print("Arm ID：", handle.id)

# robot.rm_movej([0,0,0,0,0,0], 20, 0, 0, True)

while True:
    print(robot.rm_get_current_arm_state()[1]['pose'])
    sleep(1)

print(robot.rm_get_current_arm_state()[1]['pose'])

curr_pose = robot.rm_get_current_arm_state()[1]['pose']
curr_pose[3] += 0.2

robot.rm_movej_p(curr_pose, 20, 0, 0, True)

robot.rm_delete_robot_arm()