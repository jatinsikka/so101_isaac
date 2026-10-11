"""policy.py -- the trained reach policy, outside Isaac Sim.   [C6, 2026-10-10]

Builds the 25-number observation exactly as training does (tasks/reach_env_cfg.py,
ObservationsCfg.PolicyCfg, same order) and runs the exported ONNX network.
"""

import math

import numpy as np
import onnxruntime as ort

ACTION_SCALE = 0.25  # ActionsCfg: target = current joint pos + 0.25 * action

# The target, in the robot's base frame (x forward, y left, z up; metres).
# Training sampled x 0.20..0.34, y -0.10..0.10, z 0.05..0.25, always pitch -90 deg.
TARGET_XYZ = (0.27, 0.0, 0.15)
TARGET_RPY = (0.0, -1.57, 0.0)


def quat_xyzw(roll, pitch, yaw):
    """Same as Isaac Lab 3.0's quat_from_euler_xyz, which returns (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


class Policy:
    def __init__(self, onnx_path, target_xyz=TARGET_XYZ):
        self.net = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.net.get_inputs()[0].name
        self.command = np.array([*target_xyz, *quat_xyzw(*TARGET_RPY)], dtype=np.float32)

    def observation(self, pos, vel, last_action):
        """25 numbers, in the order ObservationsCfg.PolicyCfg lists its terms."""
        return np.concatenate([
            pos,           # joint_pos       = mdp.joint_pos_rel   (default pose is all zeros)
            vel,           # joint_vel       = mdp.joint_vel_rel   (rad/s)
            self.command,  # pose_command    = mdp.generated_commands("ee_pose"): xyz + quat xyzw
            last_action,   # actions         = mdp.last_action     (raw network output)
        ]).astype(np.float32)

    def act(self, pos, vel, last_action):
        """Raw network output, 6 numbers (one per joint)."""
        obs = self.observation(pos, vel, last_action)[None, :]
        return self.net.run(None, {self.input_name: obs})[0][0]
