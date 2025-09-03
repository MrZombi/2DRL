from __future__ import annotations
import time, numpy as np
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

class DiscreteTB(BaseCallback):
    """
    TensorBoard-Logging für DISKRETE Umgebungen (partA):
      - common/samples_per_sec (Durchsatz)
      - common/action_{i}_freq (Aktionsnutzung pro Episode)
      - common/<info>_avg      (Durchschnitt über numerische info-Keys)
      - common/reward_*_avg    (Reward-Zerlegung, falls info['reward_terms'] existiert)
    """
    def __init__(self, log_every_steps: int = 1000):
        super().__init__()
        self.log_every_steps = int(log_every_steps)
        self._last_t = time.time()
        self._act_counts = None
        self._ep_steps = 0
        self._info_sums: Dict[str, float] = {}
        self._rt_sums: Dict[str, float] = {}
        self._rt_keys = set()

    def _on_step(self) -> bool:
        # Durchsatz
        if (self.num_timesteps % self.log_every_steps) == 0 and self.num_timesteps > 0:
            dt = max(1e-9, time.time() - self._last_t)
            self.logger.record("common/samples_per_sec", float(self.log_every_steps / dt))
            self._last_t = time.time()

        # Aktionen zählen (Discrete)
        actions = self.locals.get("actions", None)
        if actions is not None:
            a = np.asarray(actions).reshape(-1)
            if self._act_counts is None:
                n = int(self.training_env.action_space.n) if hasattr(self.training_env.action_space, "n") else int(a.max()+1)
                self._act_counts = np.zeros(n, dtype=np.int64)
            for v in a:
                iv = int(v)
                if 0 <= iv < self._act_counts.shape[0]:
                    self._act_counts[iv] += 1

        # Infos aggregieren
        infos = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            if not isinstance(info, dict):
                continue
            self._ep_steps += 1
            for k, v in info.items():
                if k == "episode":
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)):
                    self._info_sums[k] = self._info_sums.get(k, 0.0) + float(v)
            rterms = info.get("reward_terms", {})
            for rk, rv in rterms.items():
                self._rt_sums[rk] = self._rt_sums.get(rk, 0.0) + float(rv)
                self._rt_keys.add(rk)

        # Episodenende?
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                ep_len = max(1, int(info["episode"].get("l", self._ep_steps)))
                # Aktionsfrequenzen
                if self._act_counts is not None and self._act_counts.sum() > 0:
                    freqs = self._act_counts / max(1, self._act_counts.sum())
                    for i, f in enumerate(freqs):
                        self.logger.record(f"common/action_{i}_freq", float(f))
                # Info-Mittelwerte
                for k, s in self._info_sums.items():
                    self.logger.record(f"common/{k}_avg", float(s) / ep_len)
                # Reward-Zerlegung
                for rk in sorted(self._rt_keys):
                    self.logger.record(f"common/reward_{rk}_avg", self._rt_sums.get(rk, 0.0) / ep_len)

                # Reset
                self._act_counts = None
                self._ep_steps = 0
                self._info_sums.clear()
                self._rt_sums.clear()
                self._rt_keys.clear()
                break

        return True