import pickle

from Robotic_Arm.rm_robot_interface import *


robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("192.168.110.119", 8080)
print("Arm ID：", handle.id)

# print(robot.rm_change_work_frame("world"))

result = robot.rm_get_current_arm_state()
print(result)

while True:
    user_input = input("Type 'confirm' to continue: ")
    if user_input == "confirm":
        print("Confirmation received. Continuing program execution...")
        break  # Exit the loop and continue with the rest of your program
    else:
        print("Invalid input. Please type 'confirm' to proceed.")

file_name = "./key_poses/kp_4.pkl"
with open(file_name, 'rb') as f:
    kp4 = pickle.load(f)

result = kp4

print(robot.rm_movej_p(result[1]["pose"], 20, 0, 0, True))

robot.rm_delete_robot_arm()