# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
### Task parameters

import os

# Resolve the current user without crashing on import. os.getlogin() can raise
# OSError when there is no controlling terminal (e.g. under systemd / uv run),
# and an unknown user must not break `import constants` for the inference path.
try:
    _user = os.getlogin()
except OSError:
    import getpass

    _user = getpass.getuser()

if _user == "guningquan":
    DATA_DIR = "/mnt/ssd1/guningquan/Programs_server/act_dataset_checkpoint/dataset"
elif _user == "ubuntu20":
    DATA_DIR = "/home/robot/Dataset_and_Checkpoint/dataset"
else:
    DATA_DIR = os.path.expanduser("~/Dataset_and_Checkpoint/dataset")

TASK_CONFIGS = {

    'zip_tie_data_collection': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'aloha2_test2': {
        'dataset_dir': DATA_DIR + '/aloha2_test2',
        'episode_len': 600,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

}


### ALOHA fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
# Only the PUPPET pair is still live: it feeds the two PUPPET_GRIPPER_POSITION/VELOCITY
# helpers that real_env.get_qpos/get_qvel use. The MASTER pair has no remaining caller.
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6]) - measured working range of the ALOHA2 grippers.
# The four grippers do not share one range, so every arm is calibrated separately.
# Re-measure with `python3 examples/aloha_real/test_gripper_range.py <robot_name> --release`
# after any gripper is swapped or re-mounted.
MASTER_LEFT_GRIPPER_JOINT_OPEN = 0.7762
MASTER_LEFT_GRIPPER_JOINT_CLOSE = -0.0890
MASTER_RIGHT_GRIPPER_JOINT_OPEN = 0.8038
MASTER_RIGHT_GRIPPER_JOINT_CLOSE = -0.0522
PUPPET_LEFT_GRIPPER_JOINT_OPEN = 0.0767
PUPPET_LEFT_GRIPPER_JOINT_CLOSE = -1.5  # -1.3606
PUPPET_RIGHT_GRIPPER_JOINT_OPEN = 0.0782
PUPPET_RIGHT_GRIPPER_JOINT_CLOSE = -1.5  # -1.3898

# Backward-compatible single-sided aliases (they resolve to the RIGHT arm).
PUPPET_GRIPPER_JOINT_OPEN = PUPPET_RIGHT_GRIPPER_JOINT_OPEN
PUPPET_GRIPPER_JOINT_CLOSE = PUPPET_RIGHT_GRIPPER_JOINT_CLOSE

############################ Helper functions ############################

# --- gripper position (metres) ---
# Used by real_env.get_qpos / get_qvel to normalise the reported finger position.
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)


def _clamp01(x):
    """Clamp to [0, 1] so master over-travel cannot command the puppet past its limit."""
    return min(max(x, 0.0), 1.0)


# --- gripper joint (rad): the live teleop + deployment path ---
# master joint -> normalised action (real_env.get_action)
MASTER_LEFT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_LEFT_GRIPPER_JOINT_CLOSE) / (MASTER_LEFT_GRIPPER_JOINT_OPEN - MASTER_LEFT_GRIPPER_JOINT_CLOSE)
MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_RIGHT_GRIPPER_JOINT_CLOSE) / (MASTER_RIGHT_GRIPPER_JOINT_OPEN - MASTER_RIGHT_GRIPPER_JOINT_CLOSE)
# normalised action -> puppet joint command (real_env.set_gripper_pose)
PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_LEFT_GRIPPER_JOINT_OPEN - PUPPET_LEFT_GRIPPER_JOINT_CLOSE) + PUPPET_LEFT_GRIPPER_JOINT_CLOSE
PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_RIGHT_GRIPPER_JOINT_OPEN - PUPPET_RIGHT_GRIPPER_JOINT_CLOSE) + PUPPET_RIGHT_GRIPPER_JOINT_CLOSE
# master joint -> puppet joint, clamped (one_side_teleop)
MASTER2PUPPET_LEFT_JOINT_FN = lambda x: PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN(_clamp01(MASTER_LEFT_GRIPPER_JOINT_NORMALIZE_FN(x)))
MASTER2PUPPET_RIGHT_JOINT_FN = lambda x: PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(_clamp01(MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN(x)))

# --- master "half open" pose, used to home the grippers and to detect the start squeeze ---
MASTER_LEFT_GRIPPER_JOINT_MID = (MASTER_LEFT_GRIPPER_JOINT_OPEN + MASTER_LEFT_GRIPPER_JOINT_CLOSE) / 2
MASTER_RIGHT_GRIPPER_JOINT_MID = (MASTER_RIGHT_GRIPPER_JOINT_OPEN + MASTER_RIGHT_GRIPPER_JOINT_CLOSE) / 2
