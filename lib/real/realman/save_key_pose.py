import pickle

from Robotic_Arm.rm_robot_interface import *


robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("192.168.110.119", 8080)
print("Arm ID：", handle.id)

result = robot.rm_get_current_arm_state()
print(result)

# print(robot.rm_change_tool_frame("test_pingp"))
# print(robot.rm_get_current_tool_frame())
#
# result = robot.rm_get_current_arm_state()
# print(result)

file_name = "./key_poses/kp_9.pkl"
with open(file_name, 'wb') as f:
    pickle.dump(result, f)

print(f"Tuple saved to {file_name}")
robot.rm_delete_robot_arm()