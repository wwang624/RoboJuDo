from __future__ import annotations

from robojudo.policy.policy_cfgs import SelfPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1AmpLocoDoF(DoFConfig):
    joint_names: list[str] = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]

    default_pos: list[float] | None = [
        -0.2,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.42,
        0.42,
        0.35,
        0.35,
        -0.23,
        -0.23,
        0.18,
        -0.18,
        0.0,
        0.0,
        0.0,
        0.0,
        0.87,
        0.87,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    stiffness: list[float] | None = [
        200,
        200,
        200,
        150,
        150,
        200,
        150,
        150,
        200,
        200,
        200,
        100,
        100,
        20,
        20,
        100,
        100,
        20,
        20,
        50,
        50,
        50,
        50,
        40,
        40,
        40,
        40,
        40,
        40,
    ]

    damping: list[float] | None = [
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]


class G1AmpLocoPolicyCfg(SelfPolicyCfg):
    robot: str = "g1"
    model_group: str = "amp"
    policy_name: str = "g1_29dof_walk_14"

    obs_dof: DoFConfig = G1AmpLocoDoF()
    action_dof: DoFConfig = obs_dof

    action_beta: float = 1.0
    action_scales: list[float] = [0.25] * 29


class G1Sim2SimWalkDoF(G1AmpLocoDoF):
    default_pos: list[float] | None = [
        0.01,
        0.01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.24,
        0.24,
        0.20,
        0.20,
        -0.25,
        -0.25,
        0.18,
        -0.18,
        0.0,
        0.0,
        0.0,
        0.0,
        0.95,
        0.95,
        0.15,
        -0.15,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    stiffness: list[float] | None = [
        200,
        200,
        200,
        150,
        150,
        150,
        150,
        150,
        150,
        200,
        200,
        14.25,
        14.25,
        20,
        20,
        14.25,
        14.25,
        20,
        20,
        14.25,
        14.25,
        14.25,
        14.25,
        14.25,
        14.25,
        16.778,
        16.778,
        16.778,
        16.778,
    ]

    damping: list[float] | None = [
        5,
        5,
        5.0,
        5,
        5,
        5.0,
        5,
        5,
        5.0,
        5,
        5,
        0.907,
        0.907,
        2,
        2,
        0.907,
        0.907,
        2,
        2,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
    ]


class G1Sim2SimWalkPolicyCfg(SelfPolicyCfg):
    robot: str = "g1"
    model_group: str = "self"
    policy_name: str = "g1_sim2sim_walk.onnx"

    obs_dof: DoFConfig = G1Sim2SimWalkDoF()
    action_dof: DoFConfig = obs_dof

    action_beta: float = 1.0
    action_scales: list[float] = [0.25] * 29
    single_obs_dim: int = 96
    obs_history_len: int = 10
    history_length: int = 10
    include_phase_obs: bool = False
    max_cmd: list[float] = [1.0, 0.7, 1.57]
    commands_map: list[list[float]] = [
        [-0.8, 0.0, 1.0],
        [0.7, 0.0, -0.7],
        [1.57, 0.0, -1.57],
    ]
