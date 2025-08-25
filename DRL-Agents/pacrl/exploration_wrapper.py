# pacrl/exploration_wrapper.py
from typing import Tuple, Set
import gymnasium as gym

class ExplorationRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env,
                 w_visit: float = 0.3,
                 w_frontier: float = 0.4,
                 w_novel: float = 0.2,
                 grid_size: Tuple[int, int] = (28, 31),
                 quadrants: Tuple[int, int] = (4, 4)):
        super().__init__(env)
        self.w_visit, self.w_frontier, self.w_novel = float(w_visit), float(w_frontier), float(w_novel)
        self.W, self.H = int(grid_size[0]), int(grid_size[1])
        self.Qx, self.Qy = int(quadrants[0]), int(quadrants[1])
        self._visited_cells: Set[Tuple[int,int]] = set()
        self._visited_quads: Set[Tuple[int,int]] = set()
        self._novel_hashes: Set[int] = set()
        self._r_int_sum = 0.0

    def _quad(self, x: int, y: int):
        x = max(0, min(self.W-1, int(x))); y = max(0, min(self.H-1, int(y)))
        return (x * self.Qx // self.W, y * self.Qy // self.H)

    def reset(self, **kwargs):
        self._visited_cells.clear(); self._visited_quads.clear(); self._novel_hashes.clear()
        self._r_int_sum = 0.0
        return super().reset(**kwargs)

    def step(self, action: int):
        obs, r_ext, term, trunc, info = self.env.step(action)
        visit = frontier = novel = 0.0

        pc = info.get("player_cell")
        if isinstance(pc, (list, tuple)) and len(pc) == 2:
            x, y = int(pc[0]), int(pc[1])
            if (x, y) not in self._visited_cells:
                visit = 1.0; self._visited_cells.add((x, y))
            q = self._quad(x, y)
            if q not in self._visited_quads:
                frontier = 2.0; self._visited_quads.add(q)

        fp = info.get("state_fingerprint")
        if isinstance(fp, int):
            h = int(fp)
        else:
            if isinstance(pc, (list, tuple)) and len(pc) == 2:
                pl = int(info.get("pellets_left", 0)) & 0xF
                h = (int(pc[0]) & 0xFFFF) | ((int(pc[1]) & 0xFFFF) << 16) | (pl << 32)
            else:
                h = None
        if h is not None and h not in self._novel_hashes:
            novel = 1.0; self._novel_hashes.add(h)

        r_int = self.w_visit*visit + self.w_frontier*frontier + self.w_novel*novel
        self._r_int_sum += r_int

        info = dict(info)
        info["r_extrinsic"] = float(r_ext)
        info["r_intrinsic"] = float(r_int)
        info["r_int_sum_ep"] = float(self._r_int_sum)
        info["coverage_cells_ep"] = len(self._visited_cells)
        info["coverage_quadrants_ep"] = len(self._visited_quads)

        return obs, float(r_ext + r_int), term, trunc, info