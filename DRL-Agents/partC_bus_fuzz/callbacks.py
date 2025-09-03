# partC_bus_fuzz/callbacks.py (FINAL)
from __future__ import annotations
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class FuzzTB(BaseCallback):
    """
    Logs comparable metrics under 'common/*' and algo-specific under 'algo/*'.
    - samples_per_sec
    - action aggregates + per-dim stats
    - bus KPIs: avg_misses_step, uniq_flows_missed, avg_queue_norm, avg_util, avg_bg_norm
    - diagnostics: avg_collision_attempts_step, avg_priority_preemptions_step, avg_jitter_violations_step
    - reward_*_avg aggregation
    """
    def __init__(self, log_every_steps: int = 1000):
        super().__init__()
        self.log_every_steps = int(log_every_steps)
        self._last_wall_t = time.time()

        self._act_abs_sum = 0.0
        self._act_sq_sum = 0.0
        self._act_count = 0
        self._dim_sum = None
        self._dim_abs_sum = None
        self._dim_sq_sum = None

        self._ep_steps = 0
        self._miss_sum = 0.0
        self._q_sum = 0.0
        self._util_sum = 0.0
        self._bg_sum = 0.0
        self._uniq_last = 0

        self._coll_sum = 0.0
        self._preempt_sum = 0.0
        self._jv_sum = 0.0

        self._rt_sums = {}
        self._rt_keys_seen = set()

    def _on_step(self) -> bool:
        # actions
        actions = self.locals.get("actions", None)
        if actions is not None:
            a = np.asarray(actions, dtype=np.float32)
            if a.ndim == 1:
                a = a[None, :]
            self._act_abs_sum += float(np.abs(a).mean())
            self._act_sq_sum += float((a ** 2).mean())
            self._act_count += 1
            if self._dim_sum is None:
                ad = a.shape[-1]
                self._dim_sum = np.zeros(ad, dtype=np.float64)
                self._dim_abs_sum = np.zeros(ad, dtype=np.float64)
                self._dim_sq_sum = np.zeros(ad, dtype=np.float64)
            self._dim_sum += a.mean(axis=0)
            self._dim_abs_sum += np.abs(a).mean(axis=0)
            self._dim_sq_sum += (a ** 2).mean(axis=0)

        # infos
        infos = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            if not isinstance(info, dict):
                continue
            self._ep_steps += 1
            self._miss_sum += float(info.get("misses_step", 0.0))
            self._q_sum += float(info.get("queue_norm", 0.0))
            self._util_sum += float(info.get("util", 0.0))
            self._bg_sum += float(info.get("bg_norm", 0.0))
            self._uniq_last = int(info.get("uniq_flows_missed", self._uniq_last))

            self._coll_sum += float(info.get("collision_attempts_step", 0.0))
            self._preempt_sum += float(info.get("priority_preemptions_step", 0.0))
            self._jv_sum += float(info.get("jitter_violations_step", 0.0))

            rterms = info.get("reward_terms", {})
            for k, v in rterms.items():
                self._rt_sums[k] = self._rt_sums.get(k, 0.0) + float(v)
                self._rt_keys_seen.add(k)

        # periodic throughput
        if (self.num_timesteps % self.log_every_steps) == 0 and self.num_timesteps > 0:
            dt = max(1e-9, time.time() - self._last_wall_t)
            self.logger.record("common/samples_per_sec", float(self.log_every_steps / dt))
            self._last_wall_t = time.time()
            # gaussian policy std param if available (SAC/PPO)
            try:
                if hasattr(self.model.policy, "log_std"):
                    std_mean = float(self.model.policy.log_std.exp().mean().detach().cpu().numpy())
                    self.logger.record("algo/policy_std_param", std_mean)
            except Exception:
                pass

        # episode end?
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                ep_len = max(1, int(info["episode"].get("l", self._ep_steps)))

                self.logger.record("common/action_abs_mean", self._act_abs_sum / max(1, self._act_count))
                mean_sq = self._act_sq_sum / max(1, self._act_count)
                self.logger.record("common/action_rms", float(np.sqrt(max(0.0, mean_sq))))

                if self._dim_sum is not None:
                    dim_mean = self._dim_sum / max(1, self._act_count)
                    dim_abs = self._dim_abs_sum / max(1, self._act_count)
                    dim_rms = np.sqrt(np.maximum(0.0, self._dim_sq_sum / max(1, self._act_count)))
                    for i in range(len(dim_mean)):
                        self.logger.record(f"common/action{i}_mean", float(dim_mean[i]))
                        self.logger.record(f"common/action{i}_abs_mean", float(dim_abs[i]))
                        self.logger.record(f"common/action{i}_rms", float(dim_rms[i]))

                self.logger.record("common/avg_misses_step", self._miss_sum / ep_len)
                self.logger.record("common/uniq_flows_missed", float(self._uniq_last))
                self.logger.record("common/avg_queue_norm", self._q_sum / ep_len)
                self.logger.record("common/avg_util", self._util_sum / ep_len)
                self.logger.record("common/avg_bg_norm", self._bg_sum / ep_len)

                self.logger.record("common/avg_collision_attempts_step", self._coll_sum / ep_len)
                self.logger.record("common/avg_priority_preemptions_step", self._preempt_sum / ep_len)
                self.logger.record("common/avg_jitter_violations_step", self._jv_sum / ep_len)

                for k in sorted(self._rt_keys_seen):
                    self.logger.record(f"common/reward_{k}_avg", self._rt_sums.get(k, 0.0) / ep_len)

                # resets
                self._act_abs_sum = 0.0
                self._act_sq_sum = 0.0
                self._act_count = 0
                self._dim_sum = None
                self._dim_abs_sum = None
                self._dim_sq_sum = None

                self._ep_steps = 0
                self._miss_sum = 0.0
                self._q_sum = 0.0
                self._util_sum = 0.0
                self._bg_sum = 0.0
                self._uniq_last = 0

                self._coll_sum = 0.0
                self._preempt_sum = 0.0
                self._jv_sum = 0.0

                self._rt_sums = {}
                self._rt_keys_seen = set()

        return True
