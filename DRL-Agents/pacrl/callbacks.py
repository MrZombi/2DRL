from __future__ import annotations
import csv, os
from typing import List
from stable_baselines3.common.callbacks import BaseCallback

class ExplorationCSVCallback(BaseCallback):
    """
    Loggt pro Episode die Summen von extrinsischem und intrinsischem Reward aus `info`.
    Schreibt nach runs_py/<run_id>__<agent>.csv
    """
    def __init__(self, agent: str, run_id: str, log_dir: str = "runs_py"):
        super().__init__()
        self.agent, self.run_id = agent, run_id
        self.log_dir = log_dir
        self.file = None
        self.writer = None
        self.ep_ext = []
        self.ep_int = []
        self.ep_steps = []

    def _on_training_start(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"{self.run_id}__{self.agent}.csv")
        self.file = open(path, "a", newline="")
        self.writer = csv.writer(self.file)
        if self.file.tell() == 0:
            self.writer.writerow(["run_id","agent","episode","steps","r_ext_sum","r_int_sum"])
        n_envs = self.model.n_envs
        self.ep_ext = [0.0] * n_envs
        self.ep_int = [0.0] * n_envs
        self.ep_steps = [0] * n_envs

    def _on_step(self) -> bool:
        infos: List[dict] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            self.ep_ext[i] += float(info.get("r_extrinsic", 0.0))
            self.ep_int[i] += float(info.get("r_intrinsic", 0.0))
            self.ep_steps[i] += 1
            if i < len(dones) and bool(dones[i]):
                # Episode fertig → Zeile schreiben
                self.writer.writerow([
                    self.run_id, self.agent,
                    self.num_timesteps,  # globaler Stepzähler als Episode-ID-Proxy
                    self.ep_steps[i], self.ep_ext[i], self.ep_int[i]
                ])
                self.file.flush()
                self.ep_ext[i] = self.ep_int[i] = 0.0
                self.ep_steps[i] = 0
        return True

    def _on_training_end(self) -> None:
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None