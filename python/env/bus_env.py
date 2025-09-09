from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, Set, List
from dataclasses import dataclass, field
from sim.bus_core import BusSim, FlowCfg

@dataclass
class EnvConfig:
    n_flows: int = 4
    dt: float = 0.01
    bandwidth_Bps: float = 10000.0
    header_overhead: int = 16
    size_scale: float = 1.0
    arbitration: bool = True
    bit_error_rate: float = 0.01
    queue_limit_bytes: int = 4096
    event_rate_hz: float = 1.0
    episode_len: int = 512
    early_stop_patience: int = 128
    mode: str = "explore"   # explore | repro
    preset: str = "baseline"
    seed: Optional[int] = None
    obs_lag_steps: int = 0
    cooldown_steps: int = 32
    util_low_thresh: float = 0.3
    queue_high_thresh: float = 0.95
    ablate: Optional[List[str]] = field(default_factory=list)

def _make_default_flows(n: int) -> List[FlowCfg]:
    flows: List[FlowCfg] = []
    for i in range(n):
        flows.append(FlowCfg(
            flow_id=i, can_id=0x100 + i,
            period=0.05 + 0.01 * i, jitter_frac=0.05, offset=0.0,
            payload_bytes=8, deadline_frac=1.0
        ))
    return flows

class BusEnv(gym.Env):
    metadata = {"render_fps": 0}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed if cfg.seed is not None else 1234)
        self._ablate = set(cfg.ablate or [])

        # ---- Safety bounds ----
        self.period_min_s = 1e-4
        self.period_max_s = 2.0
        self.offset_min_frac = 0.0
        self.offset_max_frac = 1.0
        self.bandwidth_min = 1_000.0
        self.bandwidth_max = 10_000_000.0
        self.size_scale_min = 0.01
        self.size_scale_max = 10.0
        self.noise_min = 0.0
        self.noise_max = 0.2

        # Counters
        self._clamp_events = 0
        self._nan_fixes = 0

        self.flows = _make_default_flows(cfg.n_flows)
        self.sim = BusSim(
            flows=self.flows, dt=cfg.dt, bandwidth_Bps=float(np.clip(cfg.bandwidth_Bps, self.bandwidth_min, self.bandwidth_max)),
            header_overhead=cfg.header_overhead, size_scale=float(np.clip(cfg.size_scale, self.size_scale_min, self.size_scale_max)),
            arbitration=cfg.arbitration, bit_error_rate=float(np.clip(cfg.bit_error_rate, self.noise_min, self.noise_max)),
            queue_limit_bytes=cfg.queue_limit_bytes, event_rate_hz=max(0.0, cfg.event_rate_hz),
            rng_seed=cfg.seed
        )

        self.n_actions = 3 * cfg.n_flows + 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)
        self.obs_dim = 12
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Tracking
        self._missed_flows: Set[int] = set()
        self._last_miss_flow: Optional[int] = None
        self._seen_bigrams: Set[Tuple[int,int]] = set()
        self._steps_since_novelty: int = 0
        self._ret = 0.0; self._len = 0
        self._collisions_acc = 0; self._jitter_viol_acc = 0
        self._rarity_weight_acc = 0.0; self._uniq_miss_count = 0
        self._first_miss_t = -1.0; self._bigram_cnt = 0
        self._cooldown_hits = 0
        self._util_acc = 0.0; self._queue_acc = 0.0
        self._misses_per_flow: Dict[int, int] = {}
        self._cooldown_until: Dict[int, int] = {}
        self._prev_obs: Optional[np.ndarray] = None
        self._miss_severity_acc = 0.0

        # Reward weights
        if cfg.mode == "explore":
            self.W = dict(
                novelty_flow=1.0, miss_base=0.3, bigram=0.2, collision=0.02,
                time_cost=0.001, delta_action=0.01, rarity=0.1,
                violation_cost=0.01, util_band_cost=0.005, clamp_cost=0.002,
                miss_severity=0.2
            )
        else:
            self.W = dict(
                novelty_flow=0.3, miss_base=0.1, bigram=0.05, collision=0.0,
                time_cost=0.0005, delta_action=0.02, rarity=0.05,
                violation_cost=0.01, util_band_cost=0.005, clamp_cost=0.002,
                miss_severity=0.1
            )

        # per-step action bounds
        self.delta_period_rel_max = 0.05
        self.delta_jitter_rel_max = 0.02
        self.delta_offset_rel_max = 0.05
        self.delta_bg_abs_max = 0.05
        self.delta_noise_abs_max = 0.01
        self.delta_scale_abs_max = 0.05

        # episode accumulators for reward terms
        self._r_terms_acc = {
            "miss_base": 0.0, "novelty_flow": 0.0, "bigram": 0.0, "collision_bonus": 0.0,
            "time_cost": 0.0, "delta_action_cost": 0.0, "rarity_bonus": 0.0, "miss_severity": 0.0,
            "clamp_cost": 0.0, "util_band_cost": 0.0
        }

    def seed(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed if seed is not None else 1234)

    def _is_on(self, term: str) -> bool:
        return term not in (self._ablate or set())

    def _apply_action(self, a: np.ndarray) -> Dict[str, float]:
        n = self.cfg.n_flows
        clamp = bool(np.any(np.abs(a) > 0.98))
        if clamp: self._clamp_events += 1

        # per-flow
        for i in range(n):
            di = a[3*i:3*i+3]
            f = self.flows[i]
            dP = float(np.clip(di[0], -1, 1)) * self.delta_period_rel_max * f.period
            newP = float(np.clip(f.period + dP, self.period_min_s, self.period_max_s))
            if newP != f.period: self.sim.update_flow(f.flow_id, period=newP)
            dJ = float(np.clip(di[1], -1, 1)) * self.delta_jitter_rel_max
            newJ = float(np.clip(f.jitter_frac + dJ, 0.0, 1.0))
            if newJ != f.jitter_frac: self.sim.update_flow(f.flow_id, jitter_frac=newJ)
            dO = float(np.clip(di[2], -1, 1)) * self.delta_offset_rel_max * newP
            max_off = self.offset_max_frac * newP
            newO = float(np.clip(f.offset + dO, self.offset_min_frac * newP, max_off))
            if newO != f.offset: self.sim.update_flow(f.flow_id, offset=newO)

        # globals
        g = a[3*n:3*n+3]
        dBG = float(np.clip(g[0], -1, 1)) * self.delta_bg_abs_max * self.sim.bandwidth_Bps
        dNO = float(np.clip(g[1], -1, 1)) * self.delta_noise_abs_max
        dSC = float(np.clip(g[2], -1, 1)) * self.delta_scale_abs_max
        new_bw = float(np.clip(self.sim.bandwidth_Bps + dBG, self.bandwidth_min, self.bandwidth_max))
        new_ber = float(np.clip(self.sim.bit_error_rate + dNO, self.noise_min, self.noise_max))
        new_scale = float(np.clip(self.sim.size_scale + dSC, self.size_scale_min, self.size_scale_max))
        self.sim.update_globals(bandwidth_Bps=new_bw, bit_error_rate=new_ber, size_scale=new_scale)
        return {"l2": float(np.linalg.norm(a)), "clamped": float(clamp)}

    @staticmethod
    def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return np.divide(a, np.where(np.abs(b) < eps, eps, b))

    def _sanitize_obs(self, x: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(x)):
            self._nan_fixes += 1
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            x = np.clip(x, -1e6, 1e6)
        return x.astype(np.float32, copy=False)

    def _obs_now(self, stats) -> np.ndarray:
        util = float(stats.util); queue_norm = float(stats.queue_norm)
        bg_level = float(np.tanh((10000.0 - float(self.sim.bandwidth_Bps)) / 10000.0) * 0.5 + 0.5)
        noise_level = float(self.sim.bit_error_rate); size_scale = float(self.sim.size_scale)
        retrans_rate = float(self.sim.agg_retrans / max(1, self._len+1))
        collisions_step = float(stats.collisions); jitter_viol_step = float(stats.jitter_violations)

        periods = np.array([float(np.clip(f.period, self.period_min_s, self.period_max_s)) for f in self.flows], dtype=np.float64)
        jitters = np.array([float(np.clip(f.jitter_frac, 0.0, 1.0)) for f in self.flows], dtype=np.float64)
        offsets = np.array([float(np.clip(f.offset, 0.0, float(f.period))) for f in self.flows], dtype=np.float64)

        mean_period_norm = float(np.clip(np.mean(periods) / 0.1, 0.0, 2.0))
        mean_jitter = float(np.mean(jitters))
        mean_offset_norm = float(np.clip(np.mean(self._safe_div(offsets, periods)), 0.0, 1.0))
        miss_count_norm = float(np.clip(len(self._missed_flows) / max(1, self.cfg.n_flows), 0.0, 1.0))

        obs64 = np.array([
            util, queue_norm, bg_level, noise_level, size_scale, retrans_rate, collisions_step, jitter_viol_step,
            mean_period_norm, mean_jitter, mean_offset_norm, miss_count_norm
        ], dtype=np.float64)
        return self._sanitize_obs(obs64)

    def _obs(self, stats) -> np.ndarray:
        cur = self._obs_now(stats)
        if self.cfg.obs_lag_steps > 0:
            if self._prev_obs is None: self._prev_obs = cur.copy()
            out = self._prev_obs; self._prev_obs = cur; return out
        return cur

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None: self.seed(seed)
        self.sim.reset_time(0.0)
        self._missed_flows.clear(); self._last_miss_flow = None; self._seen_bigrams.clear()
        self._steps_since_novelty = 0; self._ret = 0.0; self._len = 0
        self._collisions_acc = 0; self._jitter_viol_acc = 0
        self._rarity_weight_acc = 0.0; self._uniq_miss_count = 0; self._first_miss_t = -1.0; self._bigram_cnt = 0
        self._clamp_events = 0; self._cooldown_hits = 0; self._util_acc = 0.0; self._queue_acc = 0.0
        self._misses_per_flow.clear(); self._cooldown_until.clear(); self._prev_obs = None
        self._miss_severity_acc = 0.0; self._nan_fixes = 0
        for k in self._r_terms_acc: self._r_terms_acc[k] = 0.0
        stats = self.sim.step()
        obs = self._obs(stats)
        info = dict(seed=self.cfg.seed, preset=self.cfg.preset, mode=self.cfg.mode)
        return obs, info

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        deltas = self._apply_action(a)
        stats = self.sim.step()

        # Miss handling + novelty/bigram heuristic
        miss_flow_id = None; cooldown_block = False
        if stats.misses > 0:
            miss_flow_id = int(np.argmin([f.period for f in self.flows]))
            if miss_flow_id in self._cooldown_until and self._len < self._cooldown_until[miss_flow_id]:
                cooldown_block = True; self._cooldown_hits += 1
            else:
                self._cooldown_until[miss_flow_id] = self._len + self.cfg.cooldown_steps
            if miss_flow_id not in self._missed_flows:
                self._missed_flows.add(miss_flow_id); self._uniq_miss_count += 1
            self._misses_per_flow[miss_flow_id] = self._misses_per_flow.get(miss_flow_id, 0) + 1

        bigram_new = 0
        if miss_flow_id is not None and self._last_miss_flow is not None:
            pair = (self._last_miss_flow, miss_flow_id)
            if pair not in self._seen_bigrams:
                self._seen_bigrams.add(pair); bigram_new = 1; self._bigram_cnt += 1
        if miss_flow_id is not None:
            self._last_miss_flow = miss_flow_id
            if self._first_miss_t < 0.0: self._first_miss_t = self._len * self.cfg.dt

        rarity_weight = float((1.0 - min(1.0, self.sim.bit_error_rate*5.0)) * (0.5 + 0.5*(1.0 - min(1.0, abs(10000.0 - self.sim.bandwidth_Bps)/10000.0))))
        self._rarity_weight_acc += rarity_weight
        self._miss_severity_acc += float(getattr(stats, "miss_severity_sum", 0.0))

        # Reward contributions
        r_contrib = {
            "miss_base": (self.W["miss_base"] * float(stats.misses)) if (stats.misses > 0 and not cooldown_block and self._is_on("miss_base")) else 0.0,
            "novelty_flow": (self.W["novelty_flow"] * 1.0) if (miss_flow_id is not None and self._misses_per_flow.get(miss_flow_id,0) == 1 and self._is_on("novelty_flow")) else 0.0,
            "bigram": (self.W["bigram"] * float(bigram_new)) if self._is_on("bigram") else 0.0,
            "collision_bonus": (self.W["collision"] * float(stats.collisions)) if self._is_on("collision_bonus") else 0.0,
            "time_cost": (- self.W["time_cost"] * 1.0) if self._is_on("time_cost") else 0.0,
            "delta_action_cost": (- self.W["delta_action"] * float(deltas["l2"])) if self._is_on("delta_action_cost") else 0.0,
            "rarity_bonus": (self.W["rarity"] * float(stats.misses) * float(rarity_weight)) if self._is_on("rarity_bonus") else 0.0,
            "miss_severity": (self.W["miss_severity"] * float(getattr(stats, "miss_severity_sum", 0.0))) if (self._is_on("miss_severity") and getattr(stats, "miss_severity_sum", 0.0) > 0.0) else 0.0,
            "clamp_cost": (- self.W["clamp_cost"]) if (self._is_on("clamp_cost") and deltas.get("clamped", 0.0) >= 1.0) else 0.0,
            "util_band_cost": (- self.W["util_band_cost"]) if (self._is_on("util_band_cost") and (stats.queue_norm > self.cfg.queue_high_thresh or stats.util < self.cfg.util_low_thresh)) else 0.0
        }
        r = float(sum(r_contrib.values()))
        for k, v in r_contrib.items(): self._r_terms_acc[k] += float(v)

        self._ret += r; self._len += 1
        self._collisions_acc += stats.collisions; self._jitter_viol_acc += stats.jitter_violations
        self._util_acc += stats.util; self._queue_acc += stats.queue_norm

        self._steps_since_novelty = 0 if miss_flow_id is not None else (self._steps_since_novelty + 1)
        early_stop = (self._steps_since_novelty >= self.cfg.early_stop_patience)

        terminated = False
        truncated = (self._len >= self.cfg.episode_len) or early_stop
        obs = self._obs(stats)

        kpis = dict(
            uniq_flows_missed=len(self._missed_flows),
            collisions=stats.collisions,
            jitter_violations=stats.jitter_violations,
            util=stats.util,
            queue_norm=stats.queue_norm,
            miss_severity_mean=self._miss_severity_acc / max(1, self._len),
            rarity_weight_mean=self._rarity_weight_acc / max(1, self._len),
            clamp_events=self._clamp_events,
            cooldown_hits=self._cooldown_hits,
            bigram_cnt=self._bigram_cnt,
            nan_fixes=self._nan_fixes,
        )
        info = dict(kpis=kpis, r_terms=r_contrib)

        if terminated or truncated:
            top = sorted(self._misses_per_flow.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_txt = ", ".join(f"flow{fid}:{cnt}" for fid, cnt in top) if top else "none"
            steps = max(1, self._len)
            r_terms_mean = {f"{k}_mean": (v / steps) for k, v in self._r_terms_acc.items()}
            info["episode_summary"] = dict(
                seed=self.cfg.seed, preset=self.cfg.preset, mode=self.cfg.mode,
                episode=0, ep_len=self._len, ep_return=self._ret,
                uniq_flows_missed=len(self._missed_flows),
                mean_util=self._util_acc / steps,
                mean_queue=self._queue_acc / steps,
                first_miss_t=self._first_miss_t if self._first_miss_t >= 0.0 else -1.0,
                bigram_cnt=self._bigram_cnt,
                collisions=self._collisions_acc,
                jitter_violations=self._jitter_viol_acc,
                rarity_weight_mean=kpis["rarity_weight_mean"],
                clamp_events=self._clamp_events, cooldown_hits=self._cooldown_hits,
                nan_fixes=self._nan_fixes,
                top_miss_flows=top_txt,
                r_terms=r_terms_mean
            )
        return obs, r, False, truncated, info

    def render(self): return None
    def close(self): return None
