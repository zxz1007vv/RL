from legged_gym.envs.base.legged_robot import LeggedRobot

from .zgwt_config import ZGWTRoughCfg


class Zgwt(LeggedRobot):
    """Zgwt wheel-legged robot environment."""

    cfg: ZGWTRoughCfg
