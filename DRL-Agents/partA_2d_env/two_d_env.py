from __future__ import annotations
import time, uuid, json
import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces

class TwoDEnvHTTP(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="127.0.0.1", port=8000,
                 frame_skip=1, render=False, render_every=1,
                 agent_tag="sb3", run_id=None, timeout_s=5.0):
        super().__init__()
        self.base = f"http://{host}:{port}"
        self.frame_skip = int(frame_skip)
        self.render = bool(render)
        self.render_every = int(render_every)
        self.agent_tag = str(agent_tag)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.timeout_s = float(timeout_s)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(64,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self._check_server()

    # ---------- HTTP ----------
    def _get(self, path: str) -> dict:
        r = requests.get(self.base + path, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        r = requests.post(self.base + path, data=json.dumps(payload),
                          headers={"Content-Type": "application/json"},
                          timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def _check_server(self):
        self._get("/health")
        spec = self._get("/spec")
        assert int(spec["action_space"]["n"]) == 5
        assert int(spec["observation_space"]["size"]) == 64

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        body = {
            "seed": int(seed if seed is not None else np.random.randint(0, 2**31 - 1)),
            "render": bool(options.get("render", self.render)),
            "frame_skip": int(options.get("frame_skip", self.frame_skip)),
            "render_every": int(options.get("render_every", self.render_every)),
            "agent": options.get("agent", self.agent_tag),
            "run_id": options.get("run_id", self.run_id),
        }
        out = self._post("/reset", body)
        obs = np.array(out["obs"], dtype=np.float32).reshape(64)
        info = dict(out.get("info", {}))
        return obs, info

    def step(self, action: int):
        a = int(action)
        t0 = time.perf_counter()
        try:
            out = self._post("/step", {"action": a})   # bevorzugt int 0..4
        except requests.HTTPError:
            labels = ["Up","Right","Down","Left","Stay"]   # Fallback falls Server (noch) nur Strings nimmt
            out = self._post("/step", {"action": labels[a]})
        env_step_ms = (time.perf_counter() - t0) * 1000.0

        obs = np.array(out["obs"], dtype=np.float32).reshape(64)
        reward = float(out["reward"])
        done = bool(out["done"])
        info = dict(out.get("info", {}))
        info["env_step_ms"] = env_step_ms

        reason = info.get("terminated_reason", "")
        timeout_flag = bool(info.get("timeout", False))
        terminated = done and reason in ("died", "cleared")
        truncated = done and (timeout_flag or reason == "timeout" or not terminated)

        return obs, reward, terminated, truncated, info