from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
)-> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w.torch, asset.data.root_quat_w.torch, des_pos_b)
    curr_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]]  # type: ignore
    error = curr_pos_w - des_pos_w
    return torch.exp(-error.pow(2).sum(dim=1) / std**2)

def orientation_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w.torch, des_quat_b)
    curr_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0]]  # type: ignore
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return torch.exp(-error.pow(2) / std**2)

def action_rate_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    error = env.action_manager.action - env.action_manager.prev_action
    return error.pow(2).sum(dim=1)

def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids].pow(2).sum(dim=1)

def joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids].pow(2).sum(dim=1)

def joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids].pow(2).sum(dim=1)