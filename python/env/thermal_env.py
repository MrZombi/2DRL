
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import gymnasium as gym
from gymnasium import spaces

@dataclass
class ThermalEnvConfig:
    n_cores: int = 4
    dt: float = 0.1
    episode_len: int = 2048
    arrival_rate_hz: float = 80.0
    deadline_s: float = 0.5
    service_coeff_tps_per_ghz: float = 60.0
    queue_limit_jobs: int = 200
    freq_min_ghz: float = 0.5
    freq_max_ghz: float = 4.0
    freq_step_rel_max: float = 0.10
    pwm_step_max: float = 0.15
    cooling_coeff_base: float = 0.06
    cooling_coeff_fan: float = 0.22
    temp_noise_std: float = 0.03
    base_power_w: float = 8.0
    core_dyn_power_coeff: float = 20.0
    dyn_power_exp: float = 1.4
    fan_power_base_w: float = 0.6
    fan_power_coeff_w: float = 5.0
    use_battery: bool = True
    battery_capacity_Wh: float = 40.0
    battery_init_Wh: Optional[float] = None
    power_cap_w: float = 65.0
    T_init_C: float = 45.0
    ambient_C: float = 25.0
    ambient_drift_C_per_s: float = 0.0
    T_safe_C: float = 85.0
    throttle_freq_drop_rel: float = 0.2
    throttle_hysteresis_C: float = 5.0
    util_low_thresh: float = 0.25
    util_high_thresh: float = 0.95
    queue_high_thresh: float = 0.85
    seed: Optional[int] = None
    mode: str = "control"
    preset: str = "baseline"
    obs_lag_steps: int = 0
    early_stop_patience: int = 0
    ablate: List[str] = field(default_factory=list)

class ThermalEnv(gym.Env):
    metadata = {"render_fps": 0}
    def __init__(self, cfg: ThermalEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed if cfg.seed is not None else 1234)
        self._ablate = set(cfg.ablate or [])
        self.n_actions = cfg.n_cores + 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)
        self.obs_dim = 12
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        n = cfg.n_cores
        self.freq = np.full(n, 0.7 * (cfg.freq_max_ghz + cfg.freq_min_ghz) * 0.5, dtype=np.float64)
        self.temp = np.full(n, cfg.T_init_C, dtype=np.float64)
        self.util = np.zeros(n, dtype=np.float64)
        self.queue = np.zeros(n, dtype=np.float64)
        self.battery_Wh = float(cfg.battery_capacity_Wh if cfg.battery_init_Wh is None else cfg.battery_init_Wh)
        self.pwm = 0.3
        self.ambient = float(cfg.ambient_C)
        self._len = 0; self._ret = 0.0; self._nan_fixes = 0
        self._clamp_events = 0; self._throttle_events = 0; self._powercap_events = 0
        self._first_miss_t = -1.0; self._misses_total = 0
        self._misses_per_core: Dict[int, int] = {}; self._uniq_miss_cores: set[int] = set()
        self._util_acc = 0.0; self._queue_acc = 0.0
        self._last_throttle_flag = 0.0; self._steps_since_violation = 0
        if cfg.mode == "explore":
            self.W = dict(miss_base=1.0, novelty_flow=0.0, bigram=0.0, collision=0.0,
                          time_cost=0.001, delta_action=0.01, rarity=0.0,
                          violation_cost=0.0, util_band_cost=0.005, clamp_cost=0.002,
                          miss_severity=0.5)
        else:
            self.W = dict(miss_base=1.0, novelty_flow=0.0, bigram=0.0, collision=1.0,
                          time_cost=0.002, delta_action=0.01, rarity=0.0,
                          violation_cost=0.0, util_band_cost=0.01, clamp_cost=0.002,
                          miss_severity=0.7)
        self._r_terms_acc: Dict[str, float] = {
            "miss_base":0.0,"novelty_flow":0.0,"bigram":0.0,"collision_bonus":0.0,
            "time_cost":0.0,"delta_action_cost":0.0,"rarity_bonus":0.0,"miss_severity":0.0,
            "clamp_cost":0.0,"util_band_cost":0.0
        }
        self._prev_obs: Optional[np.ndarray] = None
    def seed(self, seed: Optional[int]=None): self.rng=np.random.RandomState(seed if seed is not None else 1234)
    def _is_on(self, term:str)->bool: return term not in (self._ablate or set())
    @staticmethod
    def _safe_div(a: np.ndarray,b: np.ndarray,eps: float=1e-8)->np.ndarray:
        return np.divide(a, np.where(np.abs(b) < eps, eps, b))
    def _sanitize_obs(self, x: np.ndarray)->np.ndarray:
        if not np.all(np.isfinite(x)):
            self._nan_fixes += 1
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0); x = np.clip(x, -1e6, 1e6)
        return x.astype(np.float32, copy=False)
    def _apply_action(self, a: np.ndarray)->Dict[str,float]:
        clamp = bool(np.any(np.abs(a) > 0.98))
        if clamp: self._clamp_events += 1
        n=self.cfg.n_cores
        for i in range(n):
            drel=float(np.clip(a[i],-1,1))*self.cfg.freq_step_rel_max
            self.freq[i]=float(np.clip(self.freq[i]*(1.0+drel), self.cfg.freq_min_ghz, self.cfg.freq_max_ghz))
        dpwm=float(np.clip(a[n],-1,1))*self.cfg.pwm_step_max
        self.pwm=float(np.clip(self.pwm+dpwm,0.0,1.0))
        return {"l2": float(np.linalg.norm(a)), "clamped": float(clamp)}
    def _workload_step(self)->tuple[int,float]:
        n=self.cfg.n_cores; dt=self.cfg.dt
        lam=(self.cfg.arrival_rate_hz*dt)/max(1,n)
        arrivals=self.rng.poisson(lam,size=n).astype(np.float64)
        svc_rate=np.clip(self.freq*self.cfg.service_coeff_tps_per_ghz,1e-6,None)
        cap=svc_rate*dt
        self.queue+=arrivals
        processed=np.minimum(self.queue,cap)
        self.queue-=processed
        self.queue=np.clip(self.queue,0.0,float(self.cfg.queue_limit_jobs))
        self.util=np.clip(self._safe_div(processed,np.maximum(cap,1e-9)),0.0,1.0)
        waiting_time=self._safe_div(self.queue,svc_rate)
        excess=np.maximum(0.0,waiting_time-self.cfg.deadline_s)
        misses_step_arr=np.floor(excess*svc_rate*dt).astype(np.int64)
        misses_step=int(np.sum(misses_step_arr))
        if misses_step>0:
            self.queue-=np.minimum(self.queue,misses_step_arr.astype(np.float64))
            self.queue=np.clip(self.queue,0.0,float(self.cfg.queue_limit_jobs))
        miss_severity=float(np.sum(excess))
        if misses_step>0 and self._first_miss_t<0.0: self._first_miss_t=self._len*dt
        if misses_step>0:
            self._misses_total+=misses_step
            for i in range(n):
                if misses_step_arr[i]>0:
                    self._misses_per_core[i]=self._misses_per_core.get(i,0)+int(misses_step_arr[i])
                    self._uniq_miss_cores.add(i)
        return misses_step, miss_severity
    def _thermal_power_step(self)->tuple[float,float,int]:
        n=self.cfg.n_cores; dt=self.cfg.dt
        dyn_scale=np.power(np.clip(self.freq/self.cfg.freq_max_ghz,0.0,2.0),self.cfg.dyn_power_exp)
        core_dyn_power=self.cfg.core_dyn_power_coeff*dyn_scale*self.util
        static_share=(self.cfg.base_power_w/max(1,n))
        core_total_power=core_dyn_power+static_share
        fan_power=self.cfg.fan_power_base_w+self.cfg.fan_power_coeff_w*(self.pwm**3)
        total_power=float(np.sum(core_total_power)+fan_power)
        K=self.cfg.cooling_coeff_base+self.cfg.cooling_coeff_fan*(self.pwm**1.0)
        heat_in=core_total_power
        dT=(0.03*heat_in - K*(self.temp-self.ambient))*dt
        noise=self.rng.normal(0.0,self.cfg.temp_noise_std,size=n)*np.sqrt(dt)
        self.temp=self.temp+dT+noise
        self.ambient+=self.cfg.ambient_drift_C_per_s*dt
        throttle_events=0
        for i in range(n):
            if self.temp[i]>self.cfg.T_safe_C:
                self.freq[i]=max(self.cfg.freq_min_ghz,self.freq[i]*(1.0-self.cfg.throttle_freq_drop_rel))
                throttle_events+=1
        return total_power, fan_power, throttle_events
    def _battery_psu_step(self,total_power_W: float)->tuple[float,float,int]:
        dt=self.cfg.dt
        if total_power_W<=self.cfg.power_cap_w:
            return total_power_W,self.battery_Wh,0
        excess=total_power_W-self.cfg.power_cap_w
        if self.cfg.use_battery and self.battery_Wh>0.0:
            dWh=(excess*dt)/3600.0
            self.battery_Wh=max(0.0,self.battery_Wh-dWh)
            return total_power_W,self.battery_Wh,0
        scale=max(0.1,self.cfg.power_cap_w/max(1e-6,total_power_W))
        self.freq*=float(np.clip(scale,0.1,1.0))
        self._powercap_events+=1
        return self.cfg.power_cap_w,self.battery_Wh,1
    def _obs_now(self,misses_step:int,total_power_W:float,throttle_events:int)->np.ndarray:
        mean_util=float(np.mean(self.util))
        mean_queue_norm=float(np.mean(np.clip(self.queue/self.cfg.queue_limit_jobs,0.0,1.0)))
        mean_temp_norm=float(np.mean(np.clip(self.temp/max(self.cfg.T_safe_C,1.0),0.0,2.0)))
        max_temp_norm=float(np.max(np.clip(self.temp/max(self.cfg.T_safe_C,1.0),0.0,2.0)))
        battery_soc_norm=float(np.clip((self.battery_Wh/max(1e-9,self.cfg.battery_capacity_Wh)) if self.cfg.use_battery else 1.0,0.0,1.0))
        headroom=float(np.clip((self.cfg.power_cap_w-total_power_W)/max(1.0,self.cfg.power_cap_w),-1.0,1.0))
        mean_freq_norm=float(np.mean(np.clip(self.freq/self.cfg.freq_max_ghz,0.0,1.0)))
        throttle_flag=float(1.0 if throttle_events>0 else 0.0)
        misses_step_norm=float(np.clip(misses_step/max(1.0,self.cfg.arrival_rate_hz*self.cfg.dt),0.0,1.0))
        power_norm=float(np.clip(total_power_W/max(1.0,self.cfg.power_cap_w),0.0,2.0))
        ambient_norm=float(np.clip(self.ambient/max(1.0,self.cfg.T_safe_C),0.0,2.0))
        obs64=np.array([mean_util,mean_queue_norm,mean_temp_norm,max_temp_norm,self.pwm,
                        battery_soc_norm,headroom,mean_freq_norm,throttle_flag,
                        misses_step_norm,power_norm,ambient_norm],dtype=np.float64)
        return self._sanitize_obs(obs64)
    def _obs(self,misses_step:int,total_power_W:float,throttle_events:int)->np.ndarray:
        cur=self._obs_now(misses_step,total_power_W,throttle_events)
        if self.cfg.obs_lag_steps>0:
            if self._prev_obs is None: self._prev_obs=cur.copy()
            out=self._prev_obs; self._prev_obs=cur; return out
        return cur
    def reset(self,*,seed: Optional[int]=None, options: Optional[Dict[str,Any]]=None):
        if seed is not None: self.seed(seed)
        n=self.cfg.n_cores
        self.freq[:]=0.7*(self.cfg.freq_max_ghz+self.cfg.freq_min_ghz)*0.5
        self.temp[:]=self.cfg.T_init_C; self.util[:]=0.0; self.queue[:]=0.0
        self.pwm=0.3; self.ambient=float(self.cfg.ambient_C)
        self.battery_Wh=float(self.cfg.battery_capacity_Wh if self.cfg.battery_init_Wh is None else self.cfg.battery_init_Wh)
        self._len=0; self._ret=0.0; self._nan_fixes=0; self._clamp_events=0
        self._throttle_events=0; self._powercap_events=0; self._first_miss_t=-1.0; self._misses_total=0
        self._misses_per_core.clear(); self._uniq_miss_cores.clear()
        self._util_acc=0.0; self._queue_acc=0.0; self._last_throttle_flag=0.0; self._steps_since_violation=0
        for k in self._r_terms_acc: self._r_terms_acc[k]=0.0
        self._prev_obs=None
        obs=self._obs(misses_step=0,total_power_W=0.0,throttle_events=0)
        info=dict(seed=self.cfg.seed,preset=self.cfg.preset,mode=self.cfg.mode)
        return obs, info
    def step(self,action: np.ndarray):
        a=np.asarray(action,dtype=np.float32).reshape(-1)
        deltas=self._apply_action(a)
        misses_step,miss_severity=self._workload_step()
        total_power_W, fan_power_W, th_events=self._thermal_power_step()
        total_power_after,_,cap_events=self._battery_psu_step(total_power_W)
        self._throttle_events+=th_events; self._powercap_events+=cap_events
        self._util_acc+=float(np.mean(self.util))
        self._queue_acc+=float(np.mean(np.clip(self.queue/self.cfg.queue_limit_jobs,0.0,1.0)))
        r_contrib={
            "miss_base":- self.W["miss_base"]*float(misses_step) if self._is_on("miss_base") else 0.0,
            "novelty_flow":0.0,"bigram":0.0,
            "collision_bonus":- self.W["collision"]*float(th_events+cap_events) if self._is_on("collision_bonus") else 0.0,
            "time_cost":- self.W["time_cost"]*float(total_power_after/max(1.0,self.cfg.power_cap_w)) if self._is_on("time_cost") else 0.0,
            "delta_action_cost":- self.W["delta_action"]*float(deltas["l2"]) if self._is_on("delta_action_cost") else 0.0,
            "rarity_bonus":0.0,
            "miss_severity":- self.W["miss_severity"]*float(miss_severity) if self._is_on("miss_severity") else 0.0,
            "clamp_cost":- self.W["clamp_cost"] if (self._is_on("clamp_cost") and deltas.get("clamped",0.0)>=1.0) else 0.0,
            "util_band_cost":- self.W["util_band_cost"]*float(
                (np.mean(self.util)<self.cfg.util_low_thresh) or
                (np.mean(self.util)>self.cfg.util_high_thresh) or
                (np.mean(np.clip(self.queue/self.cfg.queue_limit_jobs,0.0,1.0))>self.cfg.queue_high_thresh)
            ) if self._is_on("util_band_cost") else 0.0,
        }
        r=float(sum(r_contrib.values()))
        for k,v in r_contrib.items(): self._r_terms_acc[k]+=float(v)
        self._ret+=r; self._len+=1
        if (misses_step==0 and th_events==0 and cap_events==0): self._steps_since_violation+=1
        else: self._steps_since_violation=0
        early_stop=(self.cfg.early_stop_patience>0 and self._steps_since_violation>=self.cfg.early_stop_patience)
        terminated=False
        end_battery=(self.cfg.use_battery and self.battery_Wh<=0.0)
        truncated=(self._len>=self.cfg.episode_len) or early_stop or end_battery
        obs=self._obs(misses_step,total_power_after,th_events)
        kpis=dict(
            uniq_flows_missed=len(self._uniq_miss_cores),
            collisions=int(cap_events),
            jitter_violations=int(th_events),
            util=float(np.mean(self.util)),
            queue_norm=float(np.mean(np.clip(self.queue/self.cfg.queue_limit_jobs,0.0,1.0))),
            miss_severity_mean=float(miss_severity),
            rarity_weight_mean=float(1.0 - self.pwm),
            clamp_events=self._clamp_events,
            cooldown_hits=0, bigram_cnt=0, nan_fixes=self._nan_fixes,
        )
        info: Dict[str, Any]=dict(kpis=kpis, r_terms=r_contrib)
        if terminated or truncated:
            steps=max(1,self._len)
            r_terms_mean={f"{k}_mean":(v/steps) for k,v in self._r_terms_acc.items()}
            top=sorted(self._misses_per_core.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_txt=", ".join(f"core{cid}:{cnt}" for cid, cnt in top) if top else "none"
            info["episode_summary"]=dict(
                seed=self.cfg.seed, preset=self.cfg.preset, mode=self.cfg.mode,
                episode=0, ep_len=self._len, ep_return=self._ret,
                uniq_flows_missed=len(self._uniq_miss_cores),
                mean_util=self._util_acc/steps, mean_queue=self._queue_acc/steps,
                first_miss_t=self._first_miss_t if self._first_miss_t>=0.0 else -1.0,
                bigram_cnt=0, collisions=self._powercap_events, jitter_violations=self._throttle_events,
                rarity_weight_mean=float(np.mean([1.0 - self.pwm])),
                clamp_events=self._clamp_events, cooldown_hits=0, nan_fixes=self._nan_fixes,
                top_miss_flows=top_txt, r_terms=r_terms_mean
            )
        return obs, r, False, truncated, info
    def render(self): return None
    def close(self): return None

def make_env(cfg: Optional[ThermalEnvConfig] = None) -> "ThermalEnv":
    return ThermalEnv(cfg or ThermalEnvConfig())
