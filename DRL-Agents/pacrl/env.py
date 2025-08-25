from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces

from .server_client import GameServerClient

DEFAULT_ACTION_NAMES = ["Up", "Right", "Down", "Left", "Stay"]

class Arcade2DEnv(gym.Env):
    """Gym-kompatibler Wrapper um den C++-HTTP-Server (2D-Environment)."""
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, server_url: str = "http://127.0.0.1:8000", seed: Optional[int] = 123,
                 render: bool = False, frame_skip: int = 1, render_every: int = 1,
                 agent_name: str = "Agent", run_id: int = 1):
        super().__init__()
        self.client = GameServerClient(server_url)
        self.seed_val = seed
        self.render_flag = render
        self.frame_skip = frame_skip
        self.render_every = render_every
        self.agent_name = agent_name
        self.run_id = run_id

        spec: Dict[str, Any] = {}
        try:
            spec = self.client.spec()
        except Exception:
            pass

        # observation aus spec
        obs_size = 64
        obs_desc = spec.get("observation", {}) or spec.get("observation_space", {})
        if isinstance(obs_desc, dict) and obs_desc.get("type") == "vector":
            obs_size = int(obs_desc.get("size", obs_size))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # actions
        self.action_names = DEFAULT_ACTION_NAMES
        act_desc = spec.get("action_space", {})
        if isinstance(act_desc, dict):
            names = act_desc.get("actions") or act_desc.get("names")
            if isinstance(names, list):
                self.action_names = [str(n) for n in names]
        self.action_space = spaces.Discrete(len(self.action_names))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed_val = int(seed)
        obs_list = self.client.reset(
            seed=self.seed_val, render=self.render_flag,
            frame_skip=self.frame_skip, render_every=self.render_every,
            agent=self.agent_name, run_id=self.run_id
        )
        obs = np.asarray(obs_list, dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        idx = int(action)
        name = self.action_names[idx] if 0 <= idx < len(self.action_names) else "Stay"
        obs_list, reward, done, info = self.client.step(name)
        obs = np.asarray(obs_list, dtype=np.float32)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return