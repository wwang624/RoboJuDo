from __future__ import annotations

from robojudo.config import ASSETS_DIR, cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
    SoccerPerceptionCtrlCfg,  # noqa: F401
    UnitreeCtrlCfg,  # noqa: F401
)
from robojudo.pipeline.pipeline_cfgs import (
    RlLocoMimicPipelineCfg,  # noqa: F401
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)

from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg  # noqa: F401
from .ctrl.g1_motion_ctrl_cfg import (  # noqa: F401
    G1MotionCtrlCfg,
    G1MotionH2HCtrlCfg,
    G1MotionKungfuBotCtrlCfg,
    G1MotionTwistCtrlCfg,
)
from .ctrl.g1_twist_redis_ctrl_cfg import G1TwistRedisCtrlCfg  # noqa: F401
from .env.g1_dummy_env_cfg import G1DummyEnvCfg  # noqa: F401
from .env.g1_mujuco_env_cfg import G1_12MujocoEnvCfg, G1_23MujocoEnvCfg, G1MujocoEnvCfg  # noqa: F401
from .env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg  # noqa: F401
from .pipeline.g1_locomimic_pipeline_cfg import G1RlLocoMimicPipelineCfg  # noqa: F401
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_amp_loco_policy_cfg import G1AmpLocoPolicyCfg  # noqa: F401
from .policy.g1_amp_loco_policy_cfg import G1Sim2SimWalkPolicyCfg
from .policy.g1_asap_policy_cfg import G1AsapLocoPolicyCfg, G1AsapPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg  # noqa: F401
from .policy.g1_dl_policy_cfg import G1dlPolicyCfg  # noqa: F401
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_kungfubot_policy_cfg import G1KungfuBotGeneralPolicyCfg, G1KungfuBotPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_soccer_policy_cfg import G1SoccerPolicyCfg, G1_SOCCER_BALL_WORLD, G1_SOCCER_GOAL_WORLD  # noqa: F401
from .policy.g1_self_policy_cfg import G1SelfPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401

G1_SOCCER_DETECTOR_MODEL = (ASSETS_DIR / "models/g1/soccer/yolov8l.engine").as_posix()

# ================= LocoMotion + MotionMimic Policy Switch Configs ================= #


@cfg_registry.register
class g1_locomimic_beyondmimic(G1RlLocoMimicPipelineCfg):
    """
    Smooth switch between multiple BeyondMimic policies, Sim2Sim.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        ),
        # JoystickCtrlCfg(
        #     combination_init_buttons=[],
        #     triggers={
        #         "A": "[SHUTDOWN]",
        #         "Back": "[POLICY_LOCO]",
        #         "Start": "[POLICY_MIMIC]",
        #         "RB": "[POLICY_SWITCH],NEXT",
        #         "LB": "[POLICY_SWITCH],LAST",
        #     },
        # ),
    ]

    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg()
    # loco_policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()
    # loco_policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    # loco_policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    """Any LocoMotion policy, as init"""

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(policy_name="Dance_wose", without_state_estimator=True),
        G1BeyondMimicPolicyCfg(policy_name="Violin", without_state_estimator=False, max_timestep=500),
        G1BeyondMimicPolicyCfg(policy_name="Waltz", without_state_estimator=False, max_timestep=850),
    ]


@cfg_registry.register
class g1_locomimic_asap(G1RlLocoMimicPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Locomotion + Deepmimic, Sim2Sim.
    Dynamic switch, keyboard control.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=True)

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [  # note: the ranking of controllers matters
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        ),
        # JoystickCtrlCfg(
        #     combination_init_buttons=[],
        #     triggers={
        #         "A": "[SHUTDOWN]",
        #         "Back": "[POLICY_LOCO]",
        #         "Start": "[POLICY_MIMIC]",
        #         "RB": "[POLICY_SWITCH],NEXT",
        #         "LB": "[POLICY_SWITCH],LAST",
        #     },
        # ),
    ]

    loco_policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()

    # fmt: off
    mimic_policies: list[G1AsapPolicyCfg] = [
        G1AsapPolicyCfg(), # default CR7_level1
        G1AsapPolicyCfg(
            policy_name="robomimic",
            relative_path="dance_0605.onnx",
            motion_length_s=18.0,
            start_upper_body_dof_pos = [
                0, 0, 0,
                0.35, 0.18, 0, 0.87, 
                0.35, -0.18, 0, 0.87,
            ],
        ),
        G1KungfuBotPolicyCfg(),
    ]
    # fmt: on


@cfg_registry.register
class g1_locomimic_soccer(G1RlLocoMimicPipelineCfg):
    """
    RoboJuDo sim closed-loop test: ASAP locomotion + HumanoidSoccer ONNX policy.
    Soccer ball and goal markers are placed from configured world coordinates in MuJoCo.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        forward_kinematic=None,
        update_with_fk=False,
        born_place_align=True,
        soccer_objects_enabled=True,
        soccer_ball_pos=G1_SOCCER_BALL_WORLD,
        soccer_goal_marker_pos=G1_SOCCER_GOAL_WORLD,
        soccer_objects_from_local_targets=False,
    )

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        ),
    ]

    # loco_policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    loco_policy: G1Sim2SimWalkPolicyCfg = G1Sim2SimWalkPolicyCfg()
    mimic_policies: list[G1SoccerPolicyCfg] = [
        G1SoccerPolicyCfg(),
    ]
    upper_dof_num: int = 0
    upper_dof_pos_default: list[float] | None = []
    upper_dof_override_indices: list[int] | None = []


@cfg_registry.register
class g1_locomimic_soccer_real(g1_locomimic_soccer):
    """
    Real G1 soccer config. Target observations must come from controller/perception ctrl_data.
    """

    env: G1RealEnvCfg = G1RealEnvCfg(
        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )
    ctrl: list[UnitreeCtrlCfg | SoccerPerceptionCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],
            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "R1": "[POLICY_SWITCH],NEXT",
                "L1": "[POLICY_SWITCH],LAST",
            },
        ),
        SoccerPerceptionCtrlCfg(
            detector_model=G1_SOCCER_DETECTOR_MODEL,
            manual_ball_local=[1.0, 0.0, -0.4],
            debug_window=False,
            threaded=True,
        ),
    ]
    mimic_policies: list[G1SoccerPolicyCfg] = [
        G1SoccerPolicyCfg(soccer_target_source="auto", use_env_soccer_obs=False),
    ]
    do_safety_check: bool = True


@cfg_registry.register
class g1_locomimic_soccer_real_dryrun(g1_locomimic_soccer_real):
    """
    Dry-run soccer perception/control wiring. Reads robot/camera state but does not send motor commands.
    """

    env: G1RealEnvCfg = G1RealEnvCfg(
        act=False,
        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

# ================= LocoMimic Policy Switch Sim2real Configs ================= #


@cfg_registry.register
class g1_locomimic_beyondmimic_real(g1_locomimic_beyondmimic):
    """
    Locomotion + Beyondmimic, Sim2Real.
    Warning: Make sure the policy is stable for real robot before using it.
    """

    env: G1RealEnvCfg = G1RealEnvCfg(
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],
            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "R1": "[POLICY_SWITCH],NEXT",
                "L1": "[POLICY_SWITCH],LAST",
            },
        ),
    ]

    do_safety_check: bool = True  # enable safety check for real robot


@cfg_registry.register
class g1_locomimic_asap_real(g1_locomimic_asap):
    """
    ASAP Locomotion + Deepmimic, Sim2Real.
    Warning: Make sure the policy is stable for real robot before using it.
    """

    # env: G1DummyEnvCfg = G1DummyEnvCfg()
    env: G1RealEnvCfg = G1RealEnvCfg(
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],
            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "R1": "[POLICY_SWITCH],NEXT",
                "L1": "[POLICY_SWITCH],LAST",
            },
        ),
    ]

    do_safety_check: bool = True  # enable safety check for real robot


# ================= ASAP Policy  ================= #
@cfg_registry.register
class g1_locomimic_asap_full(G1RlLocoMimicPipelineCfg):
    """
    Exact reproduce of the original ASAP code.
    You need to download the model files from the official repo and put them in assets/models/g1/asap
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=True)

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [  # note: the ranking of controllers matters
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
                ";": "[POLICY_SWITCH],NEXT",
                "'": "[POLICY_SWITCH],LAST",
            }
        ),
    ]

    loco_policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()

    mimic_policies: list[G1AsapPolicyCfg] = []

    def __init__(self, **data) -> None:
        super().__init__(**data)
        # add all the asap policies in asap.yaml
        from pathlib import Path

        import yaml

        asap_config = yaml.safe_load(open(Path(__file__).parent / "asap.yaml"))
        for plicy_name, relative_path in asap_config["mimic_models"].items():
            start_upper_body_dof_pos = asap_config["start_upper_body_dof_pos"].get(plicy_name, None)
            # remove some joints that are not in the g1 23-dof model
            if start_upper_body_dof_pos is not None:
                start_upper_body_dof_pos = [start_upper_body_dof_pos[i] for i in [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13]]
            motion_length_s = asap_config["motion_length_s"].get(plicy_name, 10.0)
            self.mimic_policies.append(
                G1AsapPolicyCfg(
                    policy_name=plicy_name,
                    relative_path=relative_path,
                    start_upper_body_dof_pos=start_upper_body_dof_pos,
                    motion_length_s=motion_length_s,
                )
            )
