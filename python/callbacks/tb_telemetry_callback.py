from __future__ import annotations
from typing import Sequence, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

class TBTelemetryCallback(BaseCallback):
    def __init__(self, keys: Sequence[str] | None = None, prefix: str = "env", dump_every: int = 1000):
        super().__init__()
        self.keys = list(keys) if keys is not None else [
            "uniq_flows_missed","collisions","jitter_violations","util",
            "queue_norm","miss_severity_mean","rarity_weight_mean",
            "clamp_events","cooldown_hits","bigram_cnt","nan_fixes"
        ]
        self.prefix = prefix
        self.dump_every = int(dump_every)

    def _on_training_start(self) -> bool:
        # Erzwingt sofort ein erstes Event-File mit Inhalt
        self.logger.record(f"{self.prefix}/heartbeat", 0.0)
        self.logger.dump(step=0)
        return True

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True
        if not isinstance(infos, (list, tuple)):
            infos = [infos]

        agg: Dict[str, float] = {}; counts: Dict[str, int] = {}
        r_terms_agg: Dict[str, float] = {}; r_terms_n: Dict[str, int] = {}
        for info in infos:
            data: Dict[str, Any] = info.get("kpis", info) or {}
            for k in self.keys:
                if k in data:
                    agg[k] = agg.get(k, 0.0) + float(data[k])
                    counts[k] = counts.get(k, 0) + 1
            r_terms = info.get("r_terms")
            if isinstance(r_terms, dict):
                for rk, rv in r_terms.items():
                    r_terms_agg[rk] = r_terms_agg.get(rk, 0.0) + float(rv)
                    r_terms_n[rk] = r_terms_n.get(rk, 1) + 1

        for k, v in agg.items():
            n = max(counts.get(k, 1), 1)
            self.logger.record(f"{self.prefix}/{k}", v / n)
        for rk, rv in r_terms_agg.items():
            n = max(r_terms_n.get(rk, 1), 1)
            self.logger.record(f"{self.prefix}/r_{rk}", rv / n)

        # Zusatz: sichere ein Dump in festen Abständen, falls SB3s log_interval nicht greift
        if self.dump_every and (self.num_timesteps % self.dump_every == 0):
            self.logger.dump(step=self.num_timesteps)
        return True
