from dataclasses import MISSING

from isaaclab.envs.common import ViewerCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
import isaaclab_tasks.manager_based.so101_isaac.mdp as task_mdp

from ..assets import SO101_CFG

##
# Scene definition
##


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        # Orientation matters: this asset needs the Y-up -> Z-up flip (w=0) used by
        # Isaac Lab 3.0's own reach task. With the old (0.70711,0,0,0.70711) spin the
        # table's collision top lands at z=+0.796 and is offset +0.5 in y, so the robot
        # at z=0.01 ends up buried inside the table. This puts the top at z~0.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
    )

    # robots
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.34),
            pos_y=(-0.1, 0.1),
            pos_z=(0.05, 0.25),
            roll=(0.0, 0.0),
            pitch=(-1.57, -1.57),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # target = current_pos + clip(scale * action). `clip` acts on the *scaled* offset (rad), and the
    # raw action is otherwise unbounded, so the old clip of +-1.0 rad never engaged. +-scale matches
    # deploy_policy.py, which clips the raw action to [-1, 1] before scaling. Speed is limited by
    # the actuator's velocity_limit_sim, not by shrinking scale: with relative targets, holding
    # torque is stiffness * offset, and a small offset cannot hold the arm up against gravity.
    arm_action: ActionTerm = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, clip={".*": (-0.25, 0.25)}
    )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Offset, not scale: the default pose is all zeros, so reset_joints_by_scale always produced
    # q=0 and the policy never saw any other start. Clamped to the joint limits by the event.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # TODO: weights

    # task terms
    end_effector_position_tracking = RewTerm(
        func=task_mdp.position_command_error,
        weight=1.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
            "std": 0.1,
        },
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=task_mdp.position_command_error,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
            "std": 0.05,
        },
    )
    end_effector_orientation_tracking = RewTerm(
        func=task_mdp.orientation_command_error,
        weight=1.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
            "std": 1.0,
        },
    )

    # penalty terms
    action_rate = RewTerm(
        func=task_mdp.action_rate_l2,
        weight=-0.5
    )
    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2,
        weight=-0.05,
    )
    joint_acc = RewTerm(
        func=task_mdp.joint_acc_l2,
        weight=-0.0075,
    )
    joint_torque = RewTerm(
        func=task_mdp.joint_torques_l2,
        weight=-0.5,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class ReachTaskCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    viewer: ViewerCfg = ViewerCfg(eye=(1.0, -1.0, 0.5), origin_type="asset_root", asset_name="robot")
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = 2
        self.episode_length_s = 12.0
        # simulation settings
        self.sim.dt = 0.01