# partB_thermal/env_thermal.py (FINAL)
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class ThermalConfig:
    # Simulation timing
    dt: float = 0.1                     # seconds per step
    horizon_steps: int = 200

    # Frequency (DVFS) in normalized [0,1] mapped to [f_min, f_max]
    f_min: float = 0.6                  # relative
    f_max: float = 1.4                  # relative

    # Ambient temperature drift
    T_amb_min: float = 20.0             # °C
    T_amb_max: float = 35.0             # °C
    T_amb_drift_per_step: float = 0.02  # max |ΔT_amb| per step (slow drift)
    T_amb_init: float = 26.0            # °C

    # Thermal limits
    T_safe: float = 85.0                # °C (soft safety)
    T_max: float = 95.0                 # °C (hard cap for normalization)

    # Thermal dynamics
    k_heat: float = 0.8                 # heating factor (power -> ΔT)
    k_cool: float = 0.12                # cooling factor
    fan_cool_gain: float = 0.6          # additional cooling multiplier from fan PWM
    process_heat_gain: float = 1.0      # heat ~ f^2 * load_eff

    # Workload & service (backlog queue, ON-OFF bursts)
    # Arrival in "jobs per step"
    arrival_on: float = 1.5
    arrival_off: float = 0.2
    on_to_off_p: float = 0.01
    off_to_on_p: float = 0.02
    backlog_max: float = 200.0          # drop / miss beyond this
    service_base: float = 1.0           # service per step at f=1.0
    service_load_exp: float = 0.5       # nonlinearity of service vs load

    # Energy model (arbitrary units per step)
    p_base: float = 0.1
    p_cpu_coeff: float = 0.9            # ~ f^2 * utilization
    p_fan_coeff: float = 0.08           # ~ pwm^2

    # Reward weights
    rew_lambda_energy: float = 0.05
    rew_lambda_overtemp: float = 0.5
    rew_lambda_miss: float = 1.0

    # Noise
    temp_sensor_noise: float = 0.1      # °C std
    process_noise: float = 0.01         # small noise on dynamics

    seed: Optional[int] = None


class ThermalEnv(gym.Env):
    """
    Continuous control environment for a simple thermal + queueing system.
    Action (2D, Box [0,1]x[0,1]):
        a[0] : frequency command f_cmd in [0,1] -> maps to [f_min, f_max]
        a[1] : fan PWM in [0,1]
    Observation (float32):
        [ temp_norm,         # (T - T_amb)/(T_max - T_amb) clipped to [0,1]
          freq_norm,         # normalized command in [0,1]
          fan_pwm,           # [0,1]
          load_norm,         # current incoming load rate in [0,1] (scaled)
          backlog_norm,      # backlog/backlog_max in [0,1]
          energy_norm,       # last-step energy / (p_base + p_cpu+ p_fan max) in [0,1]
          amb_norm,          # (T_amb - T_amb_min)/(T_amb_max - T_amb_min)
          throughput_norm ]  # served/backlog_max per step (small)
    Info keys (per step):
        temp_c, energy, throughput, backlog, misses, deadline_missed (bool),
        overtemp (bool), at_cap (bool), reward_terms{...}
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[ThermalConfig] = None):
        super().__init__()
        self.cfg = cfg or ThermalConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        # Action space: 2D continuous (freq command, fan pwm)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: 8D features in [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        # State variables
        self._t = 0
        self.T = float(self.cfg.T_amb_init)  # core temperature
        self.T_amb = float(self.cfg.T_amb_init)
        self.f_cmd = 1.0                     # normalized 0..1
        self.f_real = 1.0
        self.fan_pwm = 0.0
        self.backlog = 0.0
        self.load_rate = self.cfg.arrival_off
        self.mode_on = False
        self.last_energy = 0.0
        self.last_throughput = 0.0
        self.total_miss = 0.0

    # --------- Helpers ---------
    def _map_freq(self, f_norm: float) -> float:
        return float(self.cfg.f_min + f_norm * (self.cfg.f_max - self.cfg.f_min))

    def _obs(self) -> np.ndarray:
        temp_norm = (self.T - self.T_amb) / max(1e-6, (self.cfg.T_max - self.T_amb))
        temp_norm = float(np.clip(temp_norm, 0.0, 1.0))
        energy_max = self.cfg.p_base + self.cfg.p_cpu_coeff * (self.cfg.f_max**2) + self.cfg.p_fan_coeff * (1.0**2)
        obs = np.array([
            temp_norm,
            float(np.clip(self.f_cmd, 0.0, 1.0)),
            float(np.clip(self.fan_pwm, 0.0, 1.0)),
            float(np.clip(self.load_rate / max(1e-6, self.cfg.arrival_on), 0.0, 1.0)),
            float(np.clip(self.backlog / max(1e-6, self.cfg.backlog_max), 0.0, 1.0)),
            float(np.clip(self.last_energy / max(1e-6, energy_max), 0.0, 1.0)),
            float(np.clip((self.T_amb - self.cfg.T_amb_min) / max(1e-6, (self.cfg.T_amb_max - self.cfg.T_amb_min)), 0.0, 1.0)),
            float(np.clip(self.last_throughput / max(1.0, self.cfg.backlog_max), 0.0, 1.0)),
        ], dtype=np.float32)
        return obs

    # --------- Gym API ---------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self.T_amb = float(self.cfg.T_amb_init + self.rng.normal(0, 0.5))
        self.T = float(self.T_amb + self.rng.uniform(5.0, 10.0))
        self.f_cmd = float(self.rng.uniform(0.6, 0.9))
        self.f_real = self._map_freq(self.f_cmd)
        self.fan_pwm = float(self.rng.uniform(0.0, 0.2))
        self.backlog = float(self.rng.uniform(0.0, 10.0))
        self.mode_on = self.rng.random() < 0.5
        self.load_rate = self.cfg.arrival_on if self.mode_on else self.cfg.arrival_off
        self.last_energy = 0.0
        self.last_throughput = 0.0
        self.total_miss = 0.0
        info = {"episode": {"l": 0, "r": 0.0}, "config": asdict(self.cfg)}
        return self._obs(), info

    def step(self, action: np.ndarray):
        # Parse action
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f"Expected action of shape (2,), got {a.shape}")
        self.f_cmd = float(np.clip(a[0], 0.0, 1.0))
        self.f_real = self._map_freq(self.f_cmd)
        self.fan_pwm = float(np.clip(a[1], 0.0, 1.0))

        # Ambient temperature slow drift
        drift = self.rng.uniform(-self.cfg.T_amb_drift_per_step, self.cfg.T_amb_drift_per_step)
        self.T_amb = float(np.clip(self.T_amb + drift, self.cfg.T_amb_min, self.cfg.T_amb_max))

        # ON/OFF burst switching
        if self.mode_on and (self.rng.random() < self.cfg.on_to_off_p):
            self.mode_on = False
        elif (not self.mode_on) and (self.rng.random() < self.cfg.off_to_on_p):
            self.mode_on = True
        self.load_rate = self.cfg.arrival_on if self.mode_on else self.cfg.arrival_off

        # Arrivals this step (Poisson)
        arrivals = self.rng.poisson(self.load_rate)
        self.backlog += float(arrivals)

        # Service capacity this step (depends on frequency and backlog)
        util = float(min(1.0, self.backlog / max(1e-6, self.cfg.service_base)))
        service = self.cfg.service_base * self.f_real * (0.5 + 0.5 * (util ** self.cfg.service_load_exp))
        served = min(self.backlog, service)
        self.backlog -= served
        self.last_throughput = served

        # Miss / drop if backlog exceeds capacity (queue overflow)
        miss_recent = 0.0
        if self.backlog > self.cfg.backlog_max:
            miss_recent = float(self.backlog - self.cfg.backlog_max)
            self.total_miss += miss_recent
            self.backlog = self.cfg.backlog_max  # drop to cap

        # Energy per step
        p_cpu = self.cfg.p_cpu_coeff * (self.f_real ** 2) * min(1.0, util)
        p_fan = self.cfg.p_fan_coeff * (self.fan_pwm ** 2)
        energy_step = self.cfg.p_base + p_cpu + p_fan
        self.last_energy = energy_step

        # Thermal dynamics: dT = (heating - cooling) * dt + noise
        heating = self.cfg.k_heat * self.cfg.process_heat_gain * (self.f_real ** 2) * (0.2 + 0.8 * min(1.0, util))
        cooling = self.cfg.k_cool * (1.0 + self.cfg.fan_cool_gain * self.fan_pwm) * max(0.0, (self.T - self.T_amb))
        dT = (heating - cooling) * self.cfg.dt + self.rng.normal(0.0, self.cfg.process_noise)
        self.T = float(self.T + dT + self.rng.normal(0.0, self.cfg.temp_sensor_noise))

        # Reward: throughput - energy - penalties
        overtemp = float(self.T > self.cfg.T_safe)
        overtemp_pen = overtemp
        reward_terms = {
            "throughput": float(served),
            "energy_pen": float(self.cfg.rew_lambda_energy * energy_step),
            "overtemp_pen": float(self.cfg.rew_lambda_overtemp * overtemp_pen),
            "miss_pen": float(self.cfg.rew_lambda_miss * miss_recent),
        }
        reward = reward_terms["throughput"] - reward_terms["energy_pen"] - reward_terms["overtemp_pen"] - reward_terms["miss_pen"]

        self._t += 1
        terminated = False  # keep episodes time-limited; safety can be monitored via overtemp
        truncated = (self._t >= self.cfg.horizon_steps)

        obs = self._obs()
        info = {
            "misses": self.total_miss,
            "deadline_missed": bool(miss_recent > 0.0),
            "temp_c": float(self.T),
            "throughput": float(served),
            "energy": float(energy_step),
            "backlog": float(self.backlog),
            "overtemp": bool(overtemp > 0.5),
            "at_cap": bool(abs(self.backlog - self.cfg.backlog_max) < 1e-6),
            "reward_terms": reward_terms,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return

    def close(self):
        return