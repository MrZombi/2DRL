from __future__ import annotations
import os, csv, time
from typing import Optional, Sequence, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

_DEFAULT_FIELDS = [
    "run_id","seed","preset","mode","algo","total_timesteps","episode",
    "ep_len","ep_return",
    "uniq_flows_missed","mean_util","mean_queue","first_miss_t","bigram_cnt",
    "collisions","jitter_violations","rarity_weight_mean","clamp_events","cooldown_hits","nan_fixes","top_miss_flows",
    "r_miss_base","r_novelty_flow","r_bigram","r_collision_bonus","r_time_cost",
    "r_delta_action_cost","r_rarity_bonus","r_miss_severity","r_clamp_cost","r_util_band_cost",
    "timestamp"
]

class EpisodeCSVLogger(BaseCallback):
    def __init__(self, save_dir: str = "logs", filename: str = "episodes.csv",
                 run_id: Optional[str] = None, extra_fields: Optional[Sequence[str]] = None):
        super().__init__()
        self.save_dir = save_dir
        self.filename = filename
        self.run_id = run_id or str(int(time.time()))
        self.path = os.path.join(save_dir, filename)
        self.fields = list(_DEFAULT_FIELDS if extra_fields is None else [*_DEFAULT_FIELDS, *extra_fields])
        self._fh = None; self._writer = None; self._episode_counter = 0

    def _on_training_start(self) -> bool:
        os.makedirs(self.save_dir, exist_ok=True)
        fresh = not (os.path.exists(self.path) and os.path.getsize(self.path) > 0)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields)
        if fresh:
            self._writer.writeheader(); self._fh.flush()
        return True

    def _extract_algo_name(self) -> str:
        return getattr(self.model, "__class__", type(self.model)).__name__

    def _on_step(self) -> bool:
        infos = self.locals.get("infos"); dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True
        if not isinstance(infos, (list, tuple)):
            infos = [infos]; dones = [dones]
        for info, done in zip(infos, dones):
            if not done: continue
            summary: Dict[str, Any] = info.get("episode_summary") or {}
            rterms = summary.get("r_terms", {})
            self._episode_counter += 1
            row = {k: "" for k in self.fields}
            row.update({
                "run_id": self.run_id, "seed": summary.get("seed",""), "preset": summary.get("preset",""),
                "mode": summary.get("mode",""), "algo": self._extract_algo_name(),
                "total_timesteps": int(self.num_timesteps), "episode": int(summary.get("episode", self._episode_counter)),
                "ep_len": int(summary.get("ep_len",0)), "ep_return": float(summary.get("ep_return",0.0)),
                "uniq_flows_missed": int(summary.get("uniq_flows_missed",0)),
                "mean_util": float(summary.get("mean_util",0.0)), "mean_queue": float(summary.get("mean_queue",0.0)),
                "first_miss_t": float(summary.get("first_miss_t",-1.0)), "bigram_cnt": int(summary.get("bigram_cnt",0)),
                "collisions": int(summary.get("collisions",0)), "jitter_violations": int(summary.get("jitter_violations",0)),
                "rarity_weight_mean": float(summary.get("rarity_weight_mean",0.0)),
                "clamp_events": int(summary.get("clamp_events",0)), "cooldown_hits": int(summary.get("cooldown_hits",0)),
                "nan_fixes": int(summary.get("nan_fixes",0)), "top_miss_flows": summary.get("top_miss_flows",""),
                "r_miss_base": float(rterms.get("miss_base_mean", 0.0)),
                "r_novelty_flow": float(rterms.get("novelty_flow_mean", 0.0)),
                "r_bigram": float(rterms.get("bigram_mean", 0.0)),
                "r_collision_bonus": float(rterms.get("collision_bonus_mean", 0.0)),
                "r_time_cost": float(rterms.get("time_cost_mean", 0.0)),
                "r_delta_action_cost": float(rterms.get("delta_action_cost_mean", 0.0)),
                "r_rarity_bonus": float(rterms.get("rarity_bonus_mean", 0.0)),
                "r_miss_severity": float(rterms.get("miss_severity_mean", 0.0)),
                "r_clamp_cost": float(rterms.get("clamp_cost_mean", 0.0)),
                "r_util_band_cost": float(rterms.get("util_band_cost_mean", 0.0)),
                "timestamp": int(time.time()),
            })
            self._writer.writerow(row); self._fh.flush()
        return True

    def _on_training_end(self) -> None:
        if self._fh: self._fh.close()
