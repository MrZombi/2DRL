# partB_thermal/callbacks_thermal.py (FINAL)
from __future__ import annotations

import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ThermalTB(BaseCallback):
    """
    Logs comparable metrics for ALL algorithms under 'common/*',
    and algo-specific diagnostics under 'algo/*' (if available).

    Adds:
      - samples_per_sec (wall-clock)
      - per-dimension action means/abs_means/rms
      - episode-averaged temp/energy/throughput/backlog, miss_rate, overtemp_rate
      - reward_terms breakdown
    """
    def __init__(self, log_every_steps: int = 1000):
        super().__init__()
        self.log_every_steps = int(log_every_steps)

        # Rolling accumulators for action stats
        self._act_abs_sum = 0.0
        self._act_sq_sum = 0.0
        self._act_count = 0
        self._dim_sum = None
        self._dim_abs_sum = None
        self._dim_sq_sum = None

        # Episode accumulators
        self._ep_steps = 0
        self._ep_overtemp = 0
        self._ep_miss = 0
        self._temp_sum = 0.0
        self._energy_sum = 0.0
        self._throughput_sum = 0.0
        self._backlog_sum = 0.0

        self._rt_keys = ("throughput", "energy_pen", "overtemp_pen", "miss_pen")
        self._rt_sums = {k: 0.0 for k in self._rt_keys}

        self._last_wall_t = time.time()

    def _on_step(self) -> bool:
        # Actions
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

        # Infos
        infos = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            if not isinstance(info, dict):
                continue
            self._ep_steps += 1
            self._temp_sum += float(info.get("temp_c", 0.0))
            self._energy_sum += float(info.get("energy", 0.0))
            self._throughput_sum += float(info.get("throughput", 0.0))
            self._backlog_sum += float(info.get("backlog", 0.0))
            if info.get("overtemp"):
                self._ep_overtemp += 1
            if info.get("deadline_missed"):
                self._ep_miss += 1
            rterms = info.get("reward_terms", {})
            for k in self._rt_keys:
                self._rt_sums[k] += float(rterms.get(k, 0.0))

        # Periodic throughput
        if (self.num_timesteps % self.log_every_steps) == 0 and self.num_timesteps > 0:
            dt = max(1e-9, time.time() - self._last_wall_t)
            self.logger.record("common/samples_per_sec", float(self.log_every_steps / dt))
            self._last_wall_t = time.time()
            # Std param for Gaussian policies (SAC/PPO)
            try:
                if hasattr(self.model.policy, "log_std"):
                    std_mean = float(self.model.policy.log_std.exp().mean().detach().cpu().numpy())
                    self.logger.record("algo/policy_std_param", std_mean)
            except Exception:
                pass

        # Episode end?
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                ep_len = max(1, int(info["episode"].get("l", self._ep_steps)))
                # Common aggregated metrics
                self.logger.record("common/action_abs_mean", self._act_abs_sum / max(1, self._act_count))
                mean_sq = self._act_sq_sum / max(1, self._act_count)
                self.logger.record("common/action_rms", float(np.sqrt(max(0.0, mean_sq))))

                if self._dim_sum is not None:
                    dim_mean = self._dim_sum / max(1, self._act_count)
                    dim_abs = self._dim_abs_sum / max(1, self._act_count)
                    dim_rms = np.sqrt(np.maximum(0.0, self._dim_sq_sum / max(1, self._act_count)))
                    for i, v in enumerate(dim_mean):
                        self.logger.record(f"common/action{i}_mean", float(v))
                        self.logger.record(f"common/action{i}_abs_mean", float(dim_abs[i]))
                        self.logger.record(f"common/action{i}_rms", float(dim_rms[i]))

                self.logger.record("common/overtemp_rate", self._ep_overtemp / ep_len)
                self.logger.record("common/miss_rate", self._ep_miss / ep_len)
                self.logger.record("common/avg_temp", self._temp_sum / ep_len)
                self.logger.record("common/avg_energy", self._energy_sum / ep_len)
                self.logger.record("common/avg_throughput", self._throughput_sum / ep_len)
                self.logger.record("common/avg_backlog", self._backlog_sum / ep_len)

                for k, v in self._rt_sums.items():
                    self.logger.record(f"common/reward_{k}_avg", v / ep_len)

                # Reset
                self._act_abs_sum = 0.0
                self._act_sq_sum = 0.0
                self._act_count = 0
                self._dim_sum = None
                self._dim_abs_sum = None
                self._dim_sq_sum = None

                self._ep_steps = 0
                self._ep_overtemp = 0
                self._ep_miss = 0
                self._temp_sum = 0.0
                self._energy_sum = 0.0
                self._throughput_sum = 0.0
                self._backlog_sum = 0.0
                self._rt_sums = {k: 0.0 for k in self._rt_keys}

        return True
