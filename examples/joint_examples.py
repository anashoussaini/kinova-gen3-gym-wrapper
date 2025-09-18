import time
import numpy as np
from gen3_env import make


HOME = [0.0, 0.0, 3.14, -1.57, 0.0, -1.57, 1.57] 

env = make(
    "gen3",
    control_mode="joint_trajectory", 
    joint_states_topic="/joint_states",
    external_rgb_topic="/camera/camera/color/image_raw",
    wrist_rgb_topic="/camera/color/image_raw",
    joint_traj_topic="/joint_trajectory_controller/joint_trajectory",
    twist_cmd_topic="/twist_controller/commands",
    home_on_reset=True,
    home_joint_positions=HOME,      
    robotiq_action_name="/robotiq_gripper_controller/gripper_cmd", 
    gripper_min_pos=0.0, gripper_max_pos=0.8,     
)


obs = env.reset()
print("obs keys:", obs.keys())  # expect rgb_external, rgb_wrist, joint_position, joint_velocity, joint_effort

target = obs["joint_position"].copy()
target[6] += 0.9
env.step(target, seconds=5)

env.close()