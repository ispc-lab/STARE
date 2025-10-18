import pickle

from Robotic_Arm.rm_robot_interface import *


def set_confirm():
    while True:
        user_input = input("Type 'confirm' to continue: ")
        if user_input == "confirm":
            print("Confirmation received. Continuing program execution...")
            break  # Exit the loop and continue with the rest of your program
        else:
            print("Invalid input. Please type 'confirm' to proceed.")


arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

handle = arm.rm_create_robot_arm("192.168.110.119", 8080)
print(handle.id)

with open("./key_poses/kp_7.pkl", 'rb') as f:
    kp7 = pickle.load(f)

with open("./key_poses/kp_8.pkl", 'rb') as f:
    kp8 = pickle.load(f)

with open("./key_poses/kp_9.pkl", 'rb') as f:
    kp9 = pickle.load(f)


print(arm.rm_change_tool_frame("test_pingp"))
print(arm.rm_get_current_tool_frame())


print(arm.rm_change_work_frame("Base"))
print(arm.rm_get_current_work_frame())

# set origin point of work frame
set_confirm()
print(arm.rm_movej_p(kp7[1]['pose'], 20, 0, 0, 1))
print(arm.rm_set_auto_work_frame("pp", 1))

# set point on X axis of work frame
set_confirm()
print(arm.rm_movej_p(kp8[1]["pose"], 20, 0, 0, 1))
print(arm.rm_set_auto_work_frame("pp", 2))

# set point on Y axis of work frame
set_confirm()
print(arm.rm_movej_p(kp9[1]["pose"], 20, 0, 0, 1))
print(arm.rm_set_auto_work_frame("pp", 3))

# generate work frame "test_work"
print(arm.rm_set_auto_work_frame("pp", 4))

print("All Work Frames: ")
print(arm.rm_get_total_work_frame())

print(arm.rm_change_work_frame("pp"))

print("Current Arm State: ")
print(arm.rm_get_current_arm_state())

print("Current Work Frame: ")
print(arm.rm_get_current_work_frame())

print("Current Arm State: ")
print(arm.rm_get_current_arm_state())

arm.rm_delete_robot_arm()