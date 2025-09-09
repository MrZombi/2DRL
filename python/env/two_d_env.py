from __future__ import annotations
import time, json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import requests


@dataclass
class EnvConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    seed: int = 42
    render: bool = False
    frame_skip: int = 1
    render_every: int = 1
    agent: str = "sb3"          # deaktiviert C++-CSV-Logger
    run_id: str = "partA"
    http_timeout: float = 10.0


class TwoDEnv(gym.Env):
    """Gymnasium-Env, die den C++-Server (Pacman-2D) via HTTP steuert.

    Erwartete Server-Endpunkte:
      GET  /health -> {"status":"ok"}
      GET  /spec   -> { action_space:{type:"discrete",n:5,...}, observation_space:{type:"box",size:64,"dtype":"float32"}, info_keys:[...], supports:{...} }
      POST /reset  -> { obs:[...], info:{...} }
      POST /set_mode -> { ok:true }
      POST /step   -> { obs:[...], reward:float, done:bool, info:{...} }
    """

    metadata = {"render.modes": ["human"], "name": "TwoDEnv-v0"}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self._session = requests.Session()
        self._base = f"http://{cfg.host}:{cfg.port}"
        self._spec: Dict[str, Any] | None = None
        self._step_count = 0
        self._ep_return = 0.0

        self._fetch_spec()  # setzt action_space / observation_space

    # ----------------------- HTTP helpers -----------------------
    def _get(self, path: str) -> Dict[str, Any]:
        r = self._session.get(self._base + path, timeout=self.cfg.http_timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._session.post(self._base + path, json=payload, timeout=self.cfg.http_timeout)
        r.raise_for_status()
        return r.json()

    # ----------------------- Spec/Spaces ------------------------
    def _fetch_spec(self) -> None:
        self._spec = self._get("/spec")
        a = self._spec.get("action_space", {})
        o = self._spec.get("observation_space", {})
        n_actions = int(a.get("n", 5))
        size = int(o.get("size", 64))
        dtype = np.float32 if str(o.get("dtype", "float32")) == "float32" else np.float32
        # Range ist serverseitig [0,1] skaliert; belassen wir offen.
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=dtype)

    # ----------------------- Gym API ----------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is None:
            seed = self.cfg.seed
        if options is None:
            options = {}
        payload = {
            "seed": int(seed),
            "render": bool(options.get("render", self.cfg.render)),
            "frame_skip": int(options.get("frame_skip", self.cfg.frame_skip)),
            "render_every": int(options.get("render_every", self.cfg.render_every)),
            "agent": str(options.get("agent", self.cfg.agent)),
            "run_id": str(options.get("run_id", self.cfg.run_id)),
        }
        data = self._post("/reset", payload)
        obs = np.asarray(data.get("obs", []), dtype=self.observation_space.dtype)
        info = dict(data.get("info", {}))
        # Zusatz-Infos für gemeinsame Logger (CSV/TB)
        info.setdefault("mode", "agent")
        info.setdefault("preset", "2d")
        info.setdefault("seed", int(seed))
        info.setdefault("run_id", self.cfg.run_id)
        self._step_count = 0
        self._ep_return = 0.0
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Server erwartet int 0..4 (Up,Right,Down,Left,Stay)
        data = self._post("/step", {"action": int(action)})
        obs = np.asarray(data.get("obs", []), dtype=self.observation_space.dtype)
        rew = float(data.get("reward", 0.0))
        done = bool(data.get("done", False))
        info = dict(data.get("info", {}))

        self._step_count += 1
        self._ep_return += rew

        # terminated/truncated Ableitung: Server liefert bei done die Flags
        terminated = False
        truncated = False
        if done:
            timeout = bool(info.get("timeout", False))
            terminated = not timeout
            truncated = timeout
            # Episode-Zusammenfassung für CSV-Logger (generisch)
            info.setdefault("episode_summary", {
                "ep_return": float(self._ep_return),
                "ep_len": int(self._step_count),
                "seed": int(info.get("seed", self.cfg.seed)),
                "mode": info.get("mode", "agent"),
                "preset": info.get("preset", "2d"),
            })

        return obs, rew, terminated, truncated, info

    # Rendering erfolgt serverseitig; hier nur Convenience-Toggle
    def set_mode(self, *, render: bool | None = None, frame_skip: int | None = None, render_every: int | None = None) -> None:
        payload: Dict[str, Any] = {}
        if render is not None: payload["render"] = bool(render)
        if frame_skip is not None and frame_skip > 0: payload["frame_skip"] = int(frame_skip)
        if render_every is not None and render_every > 0: payload["render_every"] = int(render_every)
        if payload:
            self._post("/set_mode", payload)

    def render(self):
        # Keine lokale Darstellung – der Server zeichnet bei render=True
        return None

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass
