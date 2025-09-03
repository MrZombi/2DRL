# partC_bus_fuzz/bus_fuzz_env.py (FINAL)
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class BusFuzzConfig:
    # Basic link + window
    n_flows: int = 4
    bitrate_bps: int = 500_000
    window_ms: int = 10
    buffer_bits_limit: int = 20_000

    # Defaults per flow
    default_sizes: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    default_periods_ms: List[float] = field(default_factory=lambda: [20.0, 25.0, 40.0, 10.0])
    default_offsets_ms: List[float] = field(default_factory=lambda: [0.0, 5.0, 0.0, 2.0])

    # Arbitration / frame model
    use_priority_arbitration: bool = True        # True => CAN-like (lowest prio number wins)
    flow_priorities: List[int] = field(default_factory=lambda: [3, 2, 1, 0])  # 0=highest
    header_bits: int = 47                        # CAN-ish overhead

    # Physical effects: noise -> BER -> frame errors
    ber_base: float = 1e-7
    noise_to_ber_gain: float = 50.0             # scales with noise_level
    noise_rel_max: float = 0.10                 # Δnoise per step via action

    # Payload scaling (global)
    size_scale_min: float = 0.5
    size_scale_max: float = 2.0
    size_scale_rel_max: float = 0.10

    # Bounds/Scales for per-flow actions
    T_min: float = 5.0
    T_max: float = 200.0
    delta_T_rel_max: float = 0.10
    delta_jitter_ms: float = 2.0
    delta_offset_ms: float = 5.0

    # Background load action (+ stochastic drift)
    delta_bg_rel: float = 0.10
    bg_drift_rel_per_step: float = 0.01

    # Level 1: diagnostics thresholds
    jitter_violation_thresh_rel: float = 0.5    # count if jitter > 0.5 * period

    # Level 2: event/burst traffic (optional)
    event_rate_hz: float = 0.0                  # 0 => off (Poisson arrivals per second)
    event_size_bytes: int = 16
    event_priority: int = 2

    # Episode
    horizon_steps: int = 25
    seed: Optional[int] = None


@dataclass
class Flow:
    size_bytes: int
    period_ms: float
    deadline_factor: float = 1.5
    jitter_ms: float = 0.0
    offset_ms: float = 0.0
    priority: int = 0

    # runtime
    next_release_ms: float = 0.0
    last_latency_ms: float = 0.0
    misses_total: int = 0


@dataclass
class Msg:
    flow_id: int
    bits_left: float
    release_ms: float
    deadline_ms: float
    priority: int


class BusFuzzEnv(gym.Env):
    """
    Continuous bus-fuzz environment with realistic features:
      - Actions: ΔPeriod, ΔJitter, ΔOffset for each flow + ΔBackground, ΔNoise, ΔSizeScale
      - Arbitration: FIFO or priority-based (CAN-like)
      - Physical: noise -> BER -> frame errors -> retransmissions
      - Optional event traffic (Poisson), header overhead, queue limit
      - Reward encourages finding *new* miss flows/combos; diagnostics logged
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[BusFuzzConfig] = None, **legacy_kwargs):
        super().__init__()
        if cfg is None:
            # Legacy kwargs support
            cfg = BusFuzzConfig(
                n_flows=legacy_kwargs.get("n_flows", 4),
                bitrate_bps=legacy_kwargs.get("bitrate_bps", 500_000),
                window_ms=legacy_kwargs.get("window_ms", 10),
                buffer_bits_limit=legacy_kwargs.get("buffer_bits_limit", 20_000),
                default_sizes=legacy_kwargs.get("default_sizes", [64, 64, 64, 64]),
                default_periods_ms=legacy_kwargs.get("default_periods_ms", [20.0, 25.0, 40.0, 10.0]),
                default_offsets_ms=legacy_kwargs.get("default_offsets_ms", [0.0, 5.0, 0.0, 2.0]),
                seed=legacy_kwargs.get("seed", None)
            )
        self.cfg = cfg
        self.rng = np.random.default_rng(self.cfg.seed)

        self.n_flows = int(self.cfg.n_flows)
        self.bps = int(self.cfg.bitrate_bps)
        self.window_ms = int(self.cfg.window_ms)
        self.buf_limit = int(self.cfg.buffer_bits_limit)

        # Flows
        self.flows: List[Flow] = []
        for i in range(self.n_flows):
            prio = self.cfg.flow_priorities[i] if i < len(self.cfg.flow_priorities) else (self.n_flows - 1 - i)
            self.flows.append(Flow(size_bytes=int(self.cfg.default_sizes[i]),
                                   period_ms=float(self.cfg.default_periods_ms[i]),
                                   deadline_factor=1.5,
                                   jitter_ms=0.0,
                                   offset_ms=float(self.cfg.default_offsets_ms[i]),
                                   priority=int(prio)))

        # Globals
        self.bg_bps: float = 0.0
        self.bg_bps_max: float = 0.5 * self.bps
        self.noise_level: float = 0.0
        self.size_scale: float = 1.0

        # Queue
        self.queue: List[Msg] = []
        self.queue_bits: float = 0.0

        # Episode tracking
        self.episode_step: int = 0
        self.missed_flows_set: set[int] = set()
        self.missed_combo_set: set[Tuple[int, ...]] = set()

        # Diagnostics (Level 1)
        self.errors_recent: int = 0
        self.collision_attempts_recent: int = 0
        self.priority_preemptions_recent: int = 0
        self.jitter_violations_recent: int = 0

        # Action: 3*n + 3
        self.action_dim = 3 * self.n_flows + 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        # Observation: 4*n + 5
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4 * self.n_flows + 5,), dtype=np.float32)

    # ------------- Helpers -----------------
    def _make_obs(self, missed_recent_flags: List[int], util: float) -> np.ndarray:
        vec = []
        for i, f in enumerate(self.flows):
            Tn = float(np.clip((f.period_ms - self.cfg.T_min) / (self.cfg.T_max - self.cfg.T_min), 0.0, 1.0))
            jn = float(np.clip((f.jitter_ms) / max(1e-6, 0.5 * f.period_ms), 0.0, 1.0))
            ln = float(np.clip(f.last_latency_ms / (f.deadline_factor * f.period_ms + 1e-6), 0.0, 2.0))
            mr = float(missed_recent_flags[i])
            vec += [Tn, jn, ln, mr]
        queue_norm = float(np.clip(self.queue_bits / max(1.0, self.buf_limit), 0.0, 1.0))
        bg_norm = float(np.clip(self.bg_bps / max(1.0, self.bg_bps_max), 0.0, 1.0))
        noise_norm = float(np.clip(self.noise_level, 0.0, 1.0))
        size_scale_norm = float(
            np.clip((self.size_scale - self.cfg.size_scale_min) / max(1e-6, (self.cfg.size_scale_max - self.cfg.size_scale_min)), 0.0, 1.0)
        )
        vec += [queue_norm, float(np.clip(util, 0.0, 1.5)), bg_norm, noise_norm, size_scale_norm]
        return np.asarray(vec, dtype=np.float32)

    def _miss_recent_flags(self, flows_missed_now: List[int]) -> List[int]:
        flags = [0] * self.n_flows
        for fid in flows_missed_now:
            if 0 <= fid < self.n_flows:
                flags[fid] = 1
        return flags

    def _info_dict(self, misses_step: int, new_flow_misses: int, util: float, errors_step: int) -> Dict[str, Any]:
        ber = max(0.0, self.cfg.ber_base * (1.0 + self.cfg.noise_to_ber_gain * self.noise_level))
        return {
            "misses_step": int(misses_step),
            "new_flow_misses": int(new_flow_misses),
            "uniq_flows_missed": len(self.missed_flows_set),
            "queue_bits": float(self.queue_bits),
            "queue_norm": float(np.clip(self.queue_bits / max(1.0, self.buf_limit), 0.0, 1.0)),
            "util": float(util),
            "bg_bps": float(self.bg_bps),
            "bg_norm": float(np.clip(self.bg_bps / max(1.0, self.bg_bps_max), 0.0, 1.0)),
            "periods_ms": [float(f.period_ms) for f in self.flows],
            "jitters_ms": [float(f.jitter_ms) for f in self.flows],
            "offsets_ms": [float(f.offset_ms) for f in self.flows],
            # physical & diagnostics
            "errors_step": int(errors_step),
            "noise_level": float(self.noise_level),
            "ber": float(ber),
            "header_bits": int(self.cfg.header_bits),
            "size_scale": float(self.size_scale),
            "collision_attempts_step": int(self.collision_attempts_recent),
            "priority_preemptions_step": int(self.priority_preemptions_recent),
            "jitter_violations_step": int(self.jitter_violations_recent),
        }

    # ------------- Gym API -----------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        for f in self.flows:
            f.next_release_ms = f.offset_ms
            f.last_latency_ms = 0.0
            f.misses_total = 0
            f.period_ms = float(np.clip(f.period_ms * (1.0 + self.rng.uniform(-0.05, 0.05)), self.cfg.T_min, self.cfg.T_max))
            f.jitter_ms = float(np.clip(f.jitter_ms + self.rng.uniform(0.0, 1.0), 0.0, 0.5 * f.period_ms))
            f.offset_ms = float(np.mod(f.offset_ms, f.period_ms))

        self.bg_bps = 0.0
        self.noise_level = 0.0
        self.size_scale = 1.0

        self.queue.clear()
        self.queue_bits = 0.0
        self.episode_step = 0
        self.missed_flows_set.clear()
        self.missed_combo_set.clear()

        self.errors_recent = 0
        self.collision_attempts_recent = 0
        self.priority_preemptions_recent = 0
        self.jitter_violations_recent = 0

        obs = self._make_obs([0] * self.n_flows, util=0.0)
        info = self._info_dict(misses_step=0, new_flow_misses=0, util=0.0, errors_step=0)
        info["config"] = asdict(self.cfg)
        return obs, info

    def step(self, action: np.ndarray):
        self.episode_step += 1
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # per-flow mapping
        for i, f in enumerate(self.flows):
            dT_rel = float(a[3 * i + 0]) * self.cfg.delta_T_rel_max
            dJ_ms  = float(a[3 * i + 1]) * self.cfg.delta_jitter_ms
            dO_ms  = float(a[3 * i + 2]) * self.cfg.delta_offset_ms
            f.period_ms = float(np.clip(f.period_ms * (1.0 + dT_rel), self.cfg.T_min, self.cfg.T_max))
            f.jitter_ms = float(np.clip(f.jitter_ms + dJ_ms, 0.0, 0.5 * f.period_ms))
            f.offset_ms = float(np.mod(f.offset_ms + dO_ms, f.period_ms))

        # global mapping
        base = 3 * self.n_flows
        dBg = float(a[base + 0]) * self.cfg.delta_bg_rel * self.bps
        self.bg_bps = float(np.clip(self.bg_bps + dBg, 0.0, self.bg_bps_max))

        dNoise = float(a[base + 1]) * self.cfg.noise_rel_max
        self.noise_level = float(np.clip(self.noise_level + dNoise, 0.0, 1.0))

        dScaleRel = float(a[base + 2]) * self.cfg.size_scale_rel_max
        lo, hi = self.cfg.size_scale_min, self.cfg.size_scale_max
        self.size_scale = float(np.clip(self.size_scale * (1.0 + dScaleRel), lo, hi))

        # stochastic bg drift (optional)
        if self.cfg.bg_drift_rel_per_step > 0.0:
            delta = (self.rng.uniform(-1.0, 1.0) * self.cfg.bg_drift_rel_per_step * self.bps)
            self.bg_bps = float(np.clip(self.bg_bps + delta, 0.0, self.bg_bps_max))

        # simulate
        misses_this_step, missed_flows_now, util_ratio, errors_step = self._simulate_window()

        # reward terms
        new_flow_misses = len([fid for fid in missed_flows_now if fid not in self.missed_flows_set])
        for fid in missed_flows_now:
            self.missed_flows_set.add(fid)
        combo = tuple(sorted(self.missed_flows_set))
        new_combo_bonus = 1 if combo not in self.missed_combo_set and len(combo) >= 2 else 0
        if new_combo_bonus:
            self.missed_combo_set.add(combo)

        queue_penalty = self.queue_bits / max(1.0, self.buf_limit)
        step_penalty = 0.01
        action_cost = 0.001 * float(np.linalg.norm(a))
        error_bonus = 0.2 * errors_step
        collision_bonus = 0.1 * float(self.collision_attempts_recent)

        reward_terms = {
            "new_flow_bonus": 10.0 * new_flow_misses,
            "misses_reward": 1.0 * misses_this_step,
            "combo_bonus": 2.0 * new_combo_bonus,
            "error_bonus": error_bonus,
            "collision_bonus": collision_bonus,
            "step_pen": step_penalty,
            "action_pen": action_cost,
            "queue_pen": 0.1 * queue_penalty,
        }
        reward = (
            reward_terms["new_flow_bonus"]
            + reward_terms["misses_reward"]
            + reward_terms["combo_bonus"]
            + reward_terms["error_bonus"]
            + reward_terms["collision_bonus"]
            - reward_terms["step_pen"]
            - reward_terms["action_pen"]
            - reward_terms["queue_pen"]
        )

        obs = self._make_obs(missed_recent_flags=self._miss_recent_flags(missed_flows_now), util=util_ratio)
        terminated, truncated = False, False
        info = self._info_dict(misses_step=misses_this_step, new_flow_misses=new_flow_misses,
                               util=util_ratio, errors_step=errors_step)
        info["reward_terms"] = reward_terms
        return obs, float(reward), terminated, truncated, info

    # ------------- Simulation -----------------
    def _simulate_window(self) -> Tuple[int, List[int], float, int]:
        """
        Simulate window_ms in 1ms ticks with FIFO/priority arbitration and BER-based frame errors.
        Also counts: collision attempts, priority preemptions, jitter violations.
        """
        misses = 0
        errors = 0
        flows_missed: set[int] = set()
        cap_per_ms_bits = max(0.0, self.bps - self.bg_bps) / 1000.0
        used_bits_total = 0.0

        # reset counters
        self.collision_attempts_recent = 0
        self.priority_preemptions_recent = 0
        self.jitter_violations_recent = 0

        # BER from noise_level
        ber = max(0.0, self.cfg.ber_base * (1.0 + self.cfg.noise_to_ber_gain * self.noise_level))

        # jitter violations (instantaneous check per step)
        for f in self.flows:
            if f.jitter_ms > self.cfg.jitter_violation_thresh_rel * f.period_ms:
                self.jitter_violations_recent += 1

        for tick in range(self.window_ms):
            now = float(tick)

            # Optional event traffic
            if self.cfg.event_rate_hz > 0.0:
                lam_per_ms = self.cfg.event_rate_hz / 1000.0
                n_events = self.rng.poisson(lam_per_ms)
                for _ in range(n_events):
                    payload_bits = float(int(self.cfg.event_size_bytes * self.size_scale) * 8)
                    bits = float(self.cfg.header_bits) + payload_bits
                    prio = int(np.clip(self.cfg.event_priority, 0, max([f.priority for f in self.flows] + [0])))
                    self.queue.append(Msg(flow_id=-1, bits_left=bits, release_ms=now,
                                          deadline_ms=now + 1000.0, priority=prio))
                    self.queue_bits += bits

            # Releases
            releases_this_tick = 0
            for fid, f in enumerate(self.flows):
                guard = 1000
                while f.next_release_ms <= now + 1e-9 and guard > 0:
                    j = self.rng.uniform(-f.jitter_ms, +f.jitter_ms) if f.jitter_ms > 1e-6 else 0.0
                    rel = max(now + j, 0.0)
                    deadline = rel + f.deadline_factor * f.period_ms
                    payload_bits = float(int(f.size_bytes * self.size_scale) * 8)
                    bits = float(self.cfg.header_bits) + payload_bits
                    self.queue.append(Msg(flow_id=fid, bits_left=bits, release_ms=rel,
                                          deadline_ms=deadline, priority=f.priority))
                    self.queue_bits += bits
                    f.next_release_ms += f.period_ms
                    guard -= 1
                    releases_this_tick += 1

            if releases_this_tick >= 2:
                self.collision_attempts_recent += (releases_this_tick - 1)

            # Overflow drop
            if self.queue_bits > self.buf_limit:
                drop_guard = 10000
                while self.queue and self.queue_bits > self.buf_limit and drop_guard > 0:
                    drop = self.queue.pop(0)
                    self.queue_bits -= drop.bits_left
                    misses += 1
                    if drop.flow_id >= 0:
                        flows_missed.add(drop.flow_id)
                    drop_guard -= 1

            # Transmit
            bits_budget = cap_per_ms_bits
            used_bits_total += min(bits_budget, self.queue_bits)
            safety = 100000
            while bits_budget > 1e-9 and self.queue and safety > 0:
                if self.cfg.use_priority_arbitration and len(self.queue) >= 2:
                    priorities = [m.priority for m in self.queue]
                    idx = int(np.argmin(priorities))
                    if idx != 0:
                        self.priority_preemptions_recent += 1
                else:
                    idx = 0 if (not self.cfg.use_priority_arbitration) else int(np.argmin([m.priority for m in self.queue]))
                head = self.queue[idx]
                use = min(bits_budget, head.bits_left)
                if use <= 1e-12:
                    break
                head.bits_left -= use
                self.queue_bits -= use
                bits_budget -= use
                safety -= 1

                if head.bits_left <= 1e-9:
                    total_bits = float(self.cfg.header_bits + (0 if head.flow_id < 0 else int(self.flows[head.flow_id].size_bytes * self.size_scale) * 8))
                    p_err = 1.0 - math.exp(-ber * max(0.0, total_bits))
                    if self.rng.random() < p_err:
                        errors += 1
                        head.bits_left = total_bits
                        self.queue_bits += total_bits
                        self.queue.append(self.queue.pop(idx))  # retransmit -> move to end
                    else:
                        dt_ms = use / max(1e-9, cap_per_ms_bits)
                        finish = now + dt_ms
                        if head.flow_id >= 0:
                            latency = max(0.0, finish - head.release_ms)
                            f = self.flows[head.flow_id]
                            f.last_latency_ms = latency
                            if finish > head.deadline_ms + 1e-9:
                                misses += 1
                                f.misses_total += 1
                                flows_missed.add(head.flow_id)
                        self.queue.pop(idx)

        window_capacity_bits = cap_per_ms_bits * self.window_ms
        util_ratio = float(np.clip(used_bits_total / max(1.0, window_capacity_bits), 0.0, 1.5))
        self.errors_recent = errors
        return misses, sorted(list(flows_missed)), util_ratio, errors
