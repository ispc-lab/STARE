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

with open("./key_poses/kp_1.pkl", 'rb') as f:
    kp1 = pickle.load(f)

with open("./key_poses/kp_2.pkl", 'rb') as f:
    kp2 = pickle.load(f)

with open("./key_poses/kp_3.pkl", 'rb') as f:
    kp3 = pickle.load(f)

with open("./key_poses/kp_4.pkl", 'rb') as f:
    kp4 = pickle.load(f)

with open("./key_poses/kp_5.pkl", 'rb') as f:
    kp5 = pickle.load(f)

with open("./key_poses/kp_6.pkl", 'rb') as f:
    kp6 = pickle.load(f)


# print(arm.rm_change_work_frame("Base"))

# change the pose arbitrarily, tool end contacts reference end, calibrate points 1, 2, 3

print(arm.rm_movej_p(kp1[1]["pose"], 20, 0, 0, True))
set_confirm()

print(arm.rm_set_auto_tool_frame(1))    # point 1
print("Point 1 calibration completed")

print(arm.rm_movej_p(kp2[1]["pose"], 20, 0, 0, True))
set_confirm()

print(arm.rm_set_auto_tool_frame(2))    # point 2
print("Point 2 calibration completed")

print(arm.rm_movej_p(kp3[1]["pose"], 20, 0, 0, True))
set_confirm()

print(arm.rm_set_auto_tool_frame(3))    # point 3
print("Point 3 calibration completed")

# tool end vertically down, contact reference end, calibrate point 4
print(arm.rm_movej_p(kp4[1]["pose"], 20, 0, 0, True))
set_confirm()

print(arm.rm_set_auto_tool_frame(4))    # point 4
print("Point 4 calibration completed")

# keep the posture of point 4,
# move from point 4 along the negative X axis of the base coordinate system
# to a position as far as possible from point 4 more than 10cm, and calibrate point 5
print(arm.rm_movej_p(kp5[1]["pose"], 20, 0, 0, True))
set_confirm()

print(arm.rm_set_auto_tool_frame(5))    # point 5
print("Point 5 calibration completed")

# keep the posture of point 4,
# move from point 4 along the positive Z axis of the base coordinate system
# to a position as far as possible from point 4 more than 10cm, and calibrate point 6
print(arm.rm_movej_p(kp6[1]["pose"], 20, 0, 0, True))
set_confirm()

result = arm.rm_set_auto_tool_frame(6)  # point 6
print("Point 6 calibration completed")

# Automatically generate the coordinate system "pingpong" with a load of 2kg
# (Make sure that the number of coordinate systems does not exceed 10
# and that the name does not exist, otherwise the generation will fail)
print(arm.rm_generate_auto_tool_frame("pingpong", 2, 0, 0, 0))
print("Coordinate system has been generated")

print("All Tool Frames: ")
print(arm.rm_get_total_tool_frame())

print(arm.rm_change_tool_frame("pingpong"))

print("Current Tool Frame: ")
print(arm.rm_get_current_tool_frame())

arm.rm_delete_robot_arm()