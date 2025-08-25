import time
from typing import Any, Dict, List, Optional, Tuple
import requests

class GameServerClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        for _ in range(20):
            try:
                if self.session.get(f"{self.base}/health", timeout=timeout).ok:
                    break
            except Exception:
                time.sleep(0.2)

    def spec(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base}/spec", timeout=10)
        r.raise_for_status()
        return r.json()

    def reset(self, *, seed: Optional[int], render: bool, frame_skip: int, render_every: int,
              agent: Optional[str], run_id: Optional[int]) -> List[float]:
        payload: Dict[str, Any] = {
            "render": bool(render),
            "frame_skip": int(frame_skip),
            "render_every": int(render_every),
        }
        if seed is not None: payload["seed"] = int(seed)
        if agent is not None: payload["agent"] = str(agent)
        if run_id is not None: payload["run_id"] = int(run_id)
        r = self.session.post(f"{self.base}/reset", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()["obs"]

    def step(self, action_name: str, frame_skip: Optional[int] = None) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        payload: Dict[str, Any] = {"action": action_name}
        if frame_skip is not None:
            payload["frame_skip"] = int(frame_skip)
        r = self.session.post(f"{self.base}/step", json=payload, timeout=10)
        r.raise_for_status()
        d = r.json()
        return d["obs"], float(d["reward"]), bool(d["done"]), dict(d.get("info", {}))

    def set_mode(self, *, render: Optional[bool] = None, frame_skip: Optional[int] = None,
                 render_every: Optional[int] = None) -> None:
        payload: Dict[str, Any] = {}
        if render is not None: payload["render"] = bool(render)
        if frame_skip is not None: payload["frame_skip"] = int(frame_skip)
        if render_every is not None: payload["render_every"] = int(render_every)
        r = self.session.post(f"{self.base}/set_mode", json=payload, timeout=10)
        r.raise_for_status()