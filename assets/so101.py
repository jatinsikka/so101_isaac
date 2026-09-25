from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .. import TASK_DIR

##
# Configuration
##

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{TASK_DIR}/assets/so101.urdf",
        fix_base=True,
        merge_fixed_joints=True,
        link_density=1.0e-8,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None, damping=None
                )
        ),
        collision_type="Convex Hull",
        self_collision=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),
        joint_pos={".*": 0.0}, 
        joint_vel={".*": 0.0}, 
    ),
    actuators={
        "sts3215": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            # 12 V STS3215 stall torque is 30 kg.cm = 2.94 Nm, and the real servos run at TorqueLimit
            # 1000 (100%, see sim2real/real_robot_interface.py). The old 10 Nm made the sim arm ~3.4x
            # stronger than the real one.
            effort_limit_sim=2.9,  # ".*_limit_sim" for Implicit, ".*_limit" for Ideal PD
            # Real STS3215s are commanded with GOAL_SPEED=600 ticks/s (sim2real/real_robot_interface.py),
            # i.e. 600 / 651.9 ticks/rad = 0.92 rad/s. At the old 10 rad/s the sim arm was ~10x faster
            # than the real one and the policy oscillated on hardware. Keep these two in sync.
            velocity_limit_sim=0.92,  # ".*_limit_sim" for Implicit, ".*_limit" for Ideal PD
            stiffness=17.8,
            damping=0.0,
            armature=0.028,
            dynamic_friction=0.052,
            friction=0.052,
            viscous_friction=0.6,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of SO-101 arm using implicit actuator model."""