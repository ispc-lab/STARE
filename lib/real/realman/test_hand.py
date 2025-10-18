from Robotic_Arm.rm_robot_interface import *


arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

handle = arm.rm_create_robot_arm("192.168.110.118", 8080)
print("handle.id: ", handle.id)


# default_angle
# print(arm.rm_set_hand_follow_angle([32767,32767,32767,32767,32767,32767], True))


# loose grasp
# four_finger_angle_raw = [180, 62, 60, 63.5, 66.5]  # 四指的角度，单位为度
# four_finger_angle = [int(angle_raw * 31400 / 180) for angle_raw in four_finger_angle_raw]
#
# print(
#     arm.rm_set_hand_follow_angle(
#         [
#             four_finger_angle[0],
#             four_finger_angle[1],
#             four_finger_angle[2],
#             four_finger_angle[3],
#             four_finger_angle[4],
#             32767
#         ],
#         True,
#     )
# )


# # hard grasp
four_finger_angle_raw = [1, 62, 59.5, 62.5, 64.65]
four_finger_angle = [int(angle_raw * 31400 / 180) for angle_raw in four_finger_angle_raw]

print(
    arm.rm_set_hand_follow_angle(
        [
            four_finger_angle[0],
            four_finger_angle[1],
            four_finger_angle[2],
            four_finger_angle[3],
            four_finger_angle[4],
            32767
        ],
        True,
    )
)

# close hand connection
arm.rm_delete_robot_arm()

