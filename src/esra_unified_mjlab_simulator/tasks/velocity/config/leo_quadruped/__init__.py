"""CCBR Leo robot velocity task registration."""

from mjlab.tasks.registry import register_mjlab_task

from esra_unified_mjlab_simulator.tasks.velocity.runner import VelocityOnPolicyRunner

from .algo_cfg import ccbr_leo_ppo_runner_cfg
from .env_cfg import make_ccbr_leo_rough_env_cfg

register_mjlab_task(
    task_id="Esra-Velocity-Rough-CCBR-Leo",
    env_cfg=make_ccbr_leo_rough_env_cfg(),
    play_env_cfg=make_ccbr_leo_rough_env_cfg(play=True),
    rl_cfg=ccbr_leo_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
