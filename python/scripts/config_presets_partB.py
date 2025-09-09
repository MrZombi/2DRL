
# Presets & Curriculum for partB_thermal
PRESETS_B = {
    "baseline": dict(n_cores=4, arrival_rate_hz=80.0, power_cap_w=65.0, use_battery=True, battery_capacity_Wh=40.0, ambient_C=25.0, T_safe_C=85.0, obs_lag_steps=0),
    "heavy": dict(n_cores=6, arrival_rate_hz=140.0, power_cap_w=80.0, use_battery=True, battery_capacity_Wh=55.0, ambient_C=27.0, T_safe_C=90.0, obs_lag_steps=0),
    "ambient_hot": dict(n_cores=4, arrival_rate_hz=90.0, power_cap_w=55.0, use_battery=True, battery_capacity_Wh=40.0, ambient_C=35.0, T_safe_C=85.0, obs_lag_steps=0),
    "powercap_tight": dict(n_cores=4, arrival_rate_hz=100.0, power_cap_w=45.0, use_battery=False, battery_capacity_Wh=0.0, ambient_C=25.0, T_safe_C=85.0, obs_lag_steps=0),
    "battery_endurance": dict(n_cores=4, arrival_rate_hz=85.0, power_cap_w=35.0, use_battery=True, battery_capacity_Wh=120.0, ambient_C=25.0, T_safe_C=85.0, obs_lag_steps=0),
}
CURRICULUM_B = {
    "easy": dict(freq_step_rel_max=0.05, pwm_step_max=0.10),
    "mid":  dict(freq_step_rel_max=0.10, pwm_step_max=0.15),
    "hard": dict(freq_step_rel_max=0.15, pwm_step_max=0.20),
}
