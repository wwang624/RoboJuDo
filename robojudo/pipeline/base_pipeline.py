import logging
from abc import ABC, abstractmethod

from robojudo.tools.debug_log import DebugLogger
from robojudo.tools.soccer_episode_recorder import SoccerEpisodeRecorder

from .pipeline_cfgs import PipelineCfg

logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """
    Base Controller Module
    """

    def __init__(self, cfg: PipelineCfg):
        self.cfg = cfg

        if cfg.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        logger.info(f"Using device: {self.device}")

        if self.cfg.debug.log_obs:
            self.debug_logger = DebugLogger(run_cfg=cfg)
        self.soccer_recorder = None

        self.dt = 1.0 / 50  # default
        self.timestep = 0
        self.do_safety_check = self.cfg.do_safety_check

    def maybe_start_soccer_recorder(self):
        if self.soccer_recorder is not None or not self.cfg.debug.log_soccer:
            return
        self.soccer_recorder = SoccerEpisodeRecorder(
            run_cfg=self.cfg,
            base_path=self.cfg.debug.soccer_log_dir,
            pipeline=self,
            policy=getattr(self, "policy", None),
            env=getattr(self, "env", None),
            flush_every=self.cfg.debug.soccer_flush_every,
            log_hidden_state=self.cfg.debug.soccer_log_hidden_state,
        )

    def log_soccer_frame(self, env_data, ctrl_data, obs, extras, pd_target):
        if self.cfg.debug.log_soccer:
            self.maybe_start_soccer_recorder()
        if self.soccer_recorder is None:
            return
        self.soccer_recorder.log(
            timestep=self.timestep,
            env_data=env_data,
            ctrl_data=ctrl_data,
            obs=obs,
            extras=extras,
            pd_target=pd_target,
            policy=getattr(self, "policy", None),
        )

    def close(self):
        if self.soccer_recorder is not None:
            self.soccer_recorder.close()
        if hasattr(self, "debug_logger"):
            self.debug_logger.close()

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    def safety_check(self):
        return
