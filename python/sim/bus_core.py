from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import math, random

@dataclass
class FlowCfg:
    flow_id: int
    can_id: int
    period: float
    jitter_frac: float
    offset: float
    payload_bytes: int
    deadline_frac: float = 1.0

@dataclass
class Frame:
    ready_t: float
    flow_id: int
    can_id: int
    payload_bytes: int
    header_overhead: int
    size_scale: float
    deadline_t: float
    period: float  # neu: für Miss-Schwere-Normierung

@dataclass
class StepStats:
    util: float = 0.0
    queue_norm: float = 0.0
    collisions: int = 0
    jitter_violations: int = 0
    retransmits: int = 0
    drops: int = 0
    misses: int = 0
    miss_severity_sum: float = 0.0  # neu: Summe der lateness/period pro Step

class BusSim:
    def __init__(
        self,
        flows: List[FlowCfg],
        dt: float = 0.01,
        bandwidth_Bps: float = 10000.0,
        header_overhead: int = 16,
        size_scale: float = 1.0,
        arbitration: bool = True,
        bit_error_rate: float = 0.0,
        queue_limit_bytes: int = 4096,
        event_rate_hz: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        self.dt = dt
        self.bandwidth_Bps = bandwidth_Bps
        self.header_overhead = header_overhead
        self.size_scale = size_scale
        self.arbitration = arbitration
        self.bit_error_rate = bit_error_rate
        self.queue_limit_bytes = queue_limit_bytes
        self.event_rate_hz = event_rate_hz
        self.time = 0.0
        self.flows = list(flows)
        self.rng = random.Random(rng_seed)
        self.queue: List[Frame] = []
        self.bytes_in_queue: int = 0
        self.agg_misses = 0
        self.agg_collisions = 0
        self.agg_jitter_viol = 0
        self.agg_retrans = 0
        self._last_release_times: Dict[int, float] = {f.flow_id: -1e9 for f in flows}

    def reset_time(self, t0: float = 0.0):
        self.time = t0
        self.queue.clear()
        self.bytes_in_queue = 0
        self.agg_misses = self.agg_collisions = self.agg_jitter_viol = self.agg_retrans = 0
        for f in self.flows:
            self._last_release_times[f.flow_id] = -1e9

    def update_globals(self, *, header_overhead: Optional[int] = None, size_scale: Optional[float] = None,
                       bandwidth_Bps: Optional[float] = None, bit_error_rate: Optional[float] = None,
                       queue_limit_bytes: Optional[float] = None, event_rate_hz: Optional[float] = None):
        if header_overhead is not None: self.header_overhead = int(max(0, header_overhead))
        if size_scale is not None: self.size_scale = float(max(0.01, size_scale))
        if bandwidth_Bps is not None: self.bandwidth_Bps = float(max(100.0, bandwidth_Bps))
        if bit_error_rate is not None: self.bit_error_rate = float(min(max(0.0, bit_error_rate), 1.0))
        if queue_limit_bytes is not None: self.queue_limit_bytes = int(max(128, queue_limit_bytes))
        if event_rate_hz is not None: self.event_rate_hz = float(max(0.0, event_rate_hz))

    def update_flow(self, flow_id: int, *, period: Optional[float] = None, jitter_frac: Optional[float] = None,
                    offset: Optional[float] = None, payload_bytes: Optional[int] = None):
        for f in self.flows:
            if f.flow_id == flow_id:
                if period is not None: f.period = max(1e-4, float(period))
                if jitter_frac is not None: f.jitter_frac = min(max(0.0, float(jitter_frac)), 1.0)
                if offset is not None: f.offset = float(max(0.0, min(f.period, offset)))
                if payload_bytes is not None: f.payload_bytes = int(max(0, payload_bytes))
                return

    def _capacity_this_step(self) -> int:
        return int(self.bandwidth_Bps * self.dt)

    def _poisson_events(self) -> int:
        import math
        lam = self.event_rate_hz * self.dt
        L = math.exp(-lam)
        k = 0; p = 1.0
        while p > L:
            k += 1; p *= self.rng.random()
        return max(0, k - 1)

    def step(self) -> StepStats:
        stats = StepStats()
        t = self.time
        ready = []
        for f in self.flows:
            last = self._last_release_times[f.flow_id]
            jitter = f.jitter_frac * f.period
            target = f.period + self.rng.uniform(-jitter, jitter)
            if (t - last) >= max(0.0001, target - 1e-9):
                ready.append(Frame(
                    ready_t=t, flow_id=f.flow_id, can_id=f.can_id, payload_bytes=f.payload_bytes,
                    header_overhead=self.header_overhead, size_scale=self.size_scale,
                    deadline_t=t + f.deadline_frac * f.period, period=f.period
                ))
                self._last_release_times[f.flow_id] = t

        # Event traffic (synthetic, period=1.0 als Normierung)
        for _ in range(self._poisson_events()):
            can_id = 0x700 + int(255 * self.rng.random())
            ready.append(Frame(
                ready_t=t, flow_id=-1, can_id=can_id, payload_bytes=8,
                header_overhead=self.header_overhead, size_scale=self.size_scale,
                deadline_t=t + 1.0, period=1.0
            ))

        # Collisions & Arbitration
        if len(ready) > 1:
            stats.collisions += len(ready) - 1
        if self.arbitration:
            ready.sort(key=lambda fr: (fr.can_id, fr.ready_t))

        # Enqueue or drop
        for fr in ready:
            sz = int((fr.payload_bytes + fr.header_overhead) * max(fr.size_scale, 0.01))
            if self.bytes_in_queue + sz > self.queue_limit_bytes:
                stats.drops += 1
                stats.misses += 1
                self.agg_misses += 1
                # Drops sind keine "verspäteten" Misses → keine Severity
            else:
                self.queue.append(fr)
                self.bytes_in_queue += sz

        # Transmit
        cap = self._capacity_this_step()
        sent = 0
        new_q: List[Frame] = []
        for fr in self.queue:
            sz = int((fr.payload_bytes + fr.header_overhead) * max(fr.size_scale, 0.01))
            if sent + sz <= cap:
                sent += sz
                if self.rng.random() < self.bit_error_rate:
                    stats.retransmits += 1
                    self.agg_retrans += 1
                    new_q.append(fr)
                else:
                    if t > fr.deadline_t:
                        stats.jitter_violations += 1
                        stats.misses += 1
                        self.agg_jitter_viol += 1
                        self.agg_misses += 1
                        # Miss-Schwere: wie spät relativ zur Periodendauer
                        lateness = max(0.0, t - fr.deadline_t) / max(1e-6, fr.period)
                        stats.miss_severity_sum += lateness
            else:
                new_q.append(fr)
        self.queue = new_q
        self.bytes_in_queue = sum(int((fr.payload_bytes + fr.header_overhead) * max(fr.size_scale, 0.01)) for fr in self.queue)

        stats.util = min(1.0, sent / max(1, cap))
        stats.queue_norm = min(1.0, self.bytes_in_queue / max(1, self.queue_limit_bytes))

        self.agg_collisions += stats.collisions
        self.time += self.dt
        return stats
