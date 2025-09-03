# partA_2d_env/fault_env.py (FINAL)
from __future__ import annotations

import time
from typing import Optional, Any, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FaultInjectEnv(gym.Env):
    """
    Fault-injection wrapper for a discrete-action env (e.g., TwoDEnvHTTP).

    Injected effects (configurable):
      - obs_drop_p:    per-feature Bernoulli mask -> set to 0
      - obs_noise_std: additive Gaussian noise N(0, std^2) on float obs
      - delay_ms:      artificial sleep per step (simulated network lag)
      - sticky_p:      with prob p, reuse previous action (stuck input)
      - reward_flip_p: with prob p, multiply reward by -1
      - reward_offset: constant reward bias added every step

    Info additions per step:
      - fault_sticky_used (0/1)
      - fault_delay_ms
      - fault_obs_drop_p
      - fault_obs_noise_std
      - fault_reward_flipped (0/1)
      - reward_offset

    Notes:
      * Works with spaces.Discrete and spaces.Box actions.
      * Observations that are non-numeric are passed through unchanged.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        env: gym.Env,
        obs_drop_p: float = 0.0,
        obs_noise_std: float = 0.0,
        delay_ms: float = 0.0,
        sticky_p: float = 0.0,
        reward_flip_p: float = 0.0,
        reward_offset: float = 0.0,
    ):
        super().__init__()
        self.env = env
        self.obs_drop_p = float(obs_drop_p)
        self.obs_noise_std = float(obs_noise_std)
        self.delay_ms = float(delay_ms)
        self.sticky_p = float(sticky_p)
        self.reward_flip_p = float(reward_flip_p)
        self.reward_offset = float(reward_offset)

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._last_action = None
        self._rng = np.random.default_rng()

    # -------------- helpers --------------
    def _apply_obs_faults(self, obs):
        try:
            arr = np.asarray(obs, dtype=np.float32)
        except Exception:
            return obs  # non-numeric obs

        if self.obs_drop_p > 0.0:
            mask = self._rng.random(size=arr.shape) >= self.obs_drop_p
            arr = arr * mask.astype(arr.dtype)

        if self.obs_noise_std > 0.0:
            arr = arr + self._rng.normal(0.0, self.obs_noise_std, size=arr.shape).astype(arr.dtype)

        return arr

    def _maybe_sticky(self, action):
        used_sticky = False
        if self._last_action is not None and self.sticky_p > 0.0 and self._rng.random() < self.sticky_p:
            action_to_use = self._last_action
            used_sticky = True
        else:
            action_to_use = action

        # Keep a copy of the chosen action
        if isinstance(action_to_use, np.ndarray):
            self._last_action = action_to_use.copy()
        else:
            self._last_action = action_to_use
        return action_to_use, used_sticky

    # -------------- gym API --------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._apply_obs_faults(obs)
        self._last_action = None
        return obs, info

    def step(self, action):
        # sticky
        action_used, used_sticky = self._maybe_sticky(action)

        # delay
        if self.delay_ms > 0.0:
            time.sleep(self.delay_ms / 1000.0)

        # step underlying env
        obs, reward, terminated, truncated, info = self.env.step(action_used)

        # faults on obs and reward
        obs = self._apply_obs_faults(obs)

        reward_flipped = 0
        if self.reward_flip_p > 0.0 and self._rng.random() < self.reward_flip_p:
            reward = -float(reward)
            reward_flipped = 1
        reward = float(reward) + self.reward_offset

        # annotate info
        info = dict(info or {})
        info.update({
            "fault_sticky_used": int(used_sticky),
            "fault_delay_ms": float(self.delay_ms),
            "fault_obs_drop_p": float(self.obs_drop_p),
            "fault_obs_noise_std": float(self.obs_noise_std),
            "fault_reward_flipped": int(reward_flipped),
            "reward_offset": float(self.reward_offset),
        })
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render() if hasattr(self.env, "render") else None

    def close(self):
        return self.env.close() if hasattr(self.env, "close") else None