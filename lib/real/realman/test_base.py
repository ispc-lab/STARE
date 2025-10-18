import pickle

from Robotic_Arm.rm_robot_interface import *


robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("192.168.110.119", 8080)
print("Arm ID：", handle.id)

software_info = robot.rm_get_arm_software_info()
if software_info[0] == 0:
    print("\n================== Arm Software Information ==================")
    print("Arm Model: ", software_info[1]['product_version'])
    print("Algorithm Library Version: ", software_info[1]['algorithm_info']['version'])
    print("Control Layer Software Version: ", software_info[1]['ctrl_info']['version'])
    print("Dynamics Version: ", software_info[1]['dynamic_info']['model_version'])
    print("Planning Layer Software Version: ", software_info[1]['plan_info']['version'])
    print("==============================================================\n")
else:
    print("\nFailed to get arm software information, Error code: ", software_info[0], "\n")


print(robot.rm_change_work_frame("World"))

print("Current Arm State: ")
print(robot.rm_get_current_arm_state())

print("All Work Frames: ")
print(robot.rm_get_total_work_frame())

print("Current Work Frame: ")
print(robot.rm_get_current_work_frame())

print("All Tool Frames: ")
print(robot.rm_get_total_tool_frame())

print("Current Tool Frame: ")
print(robot.rm_get_current_tool_frame())

robot.rm_delete_robot_arm()