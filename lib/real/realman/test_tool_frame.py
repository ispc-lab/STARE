import pickle

from Robotic_Arm.rm_robot_interface import *


robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("192.168.110.119", 8080)
print("Arm ID：", handle.id)

result = robot.rm_get_current_arm_state()
print(result)

print(robot.rm_change_tool_frame("test_pingp"))
print(robot.rm_get_current_tool_frame())

print(robot.rm_change_work_frame("pp"))
print(robot.rm_get_current_work_frame())

result = robot.rm_get_current_arm_state()
print(result)

# print(robot.rm_set_teach_frame(1))

# while True:
#     user_input = input("Type 'confirm' to continue: ")
#     if user_input == "confirm":
#         print("Confirmation received. Continuing program execution...")
#         break  # Exit the loop and continue with the rest of your program
#     else:
#         print("Invalid input. Please type 'confirm' to proceed.")

# print(robot.rm_set_pos_step(rm_pos_teach_type_e.RM_X_DIR_E, 0.05, 20, 1))

# print(robot.rm_set_pos_step(rm_pos_teach_type_e.RM_Y_DIR_E, -0.05, 20, 1))

# print(robot.rm_set_pos_step(rm_pos_teach_type_e.RM_Z_DIR_E, -0.05, 20, 1))

# curr_pose = result[1]['pose']
# curr_pose[0] += 0.1
#
# print(robot.rm_movej_p(curr_pose, 20, 0, 0, True))

robot.rm_delete_robot_arm()