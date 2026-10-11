"""View an environment on its own, without task registration or any RL algorithm.

The env cfg is built by calling a factory function given as ``module:function``, so
the task does not need to be registered. The environment is stepped with a dummy
policy (zero or random actions), which makes this handy for checking the scene,
sensors, commands and terminations while writing a new env cfg.

Run with:
    uv run view-env <module:function> [OPTIONS]

Example:
    uv run view-env esra_unified_mjlab_simulator.tasks.velocity.config.leo_quadruped.env_cfg:make_ccbr_leo_rough_env_cfg
"""

import importlib
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Literal

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class ViewEnvConfig:
    env_cfg: tyro.conf.Positional[str]
    """Env cfg factory as ``module:function`` (e.g. ``pkg.env_cfg:make_cfg``)."""
    play: bool = True
    """Pass ``play=True`` to the factory if it accepts a ``play`` argument."""
    agent: Literal["zero", "random"] = "zero"
    """Dummy policy used to drive the environment."""
    num_envs: int | None = None
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False
    """Disable all termination conditions."""


def load_env_cfg_from_spec(spec: str, play: bool) -> ManagerBasedRlEnvCfg:
    """Build an env cfg from a ``module:function`` spec."""
    module_name, sep, func_name = spec.partition(":")
    if not sep or not module_name or not func_name:
        raise ValueError(f"Expected 'module:function', got '{spec}'.")
    factory = getattr(importlib.import_module(module_name), func_name)
    if "play" in inspect.signature(factory).parameters:
        return factory(play=play)
    return factory()


def run_view_env(cfg: ViewEnvConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg_from_spec(cfg.env_cfg, cfg.play)

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs

    # The viewers expect a vec-env interface. The wrapper is only an adapter and
    # does not involve any RL algorithm.
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape

    class DummyPolicy:
        def __call__(self, obs) -> torch.Tensor:
            del obs
            if cfg.agent == "zero":
                return torch.zeros(action_shape, device=env.unwrapped.device)
            return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

    policy = DummyPolicy()

    if cfg.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        # The native viewer needs ``mjpython`` on macOS, so it is opt-in there even
        # when DISPLAY is set (e.g. by XQuartz).
        resolved_viewer = (
            "native" if has_display and sys.platform != "darwin" else "viser"
        )
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


def main() -> None:
    cfg = tyro.cli(ViewEnvConfig, config=mjlab.TYRO_FLAGS)
    run_view_env(cfg)


if __name__ == "__main__":
    main()
