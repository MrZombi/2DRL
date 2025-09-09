PRESETS = {
    "baseline":          dict(event_rate_hz=1.0, bandwidth_Bps=10000.0, bit_error_rate=0.01, n_flows=4,  obs_lag_steps=0),
    "stress_low_noise":  dict(event_rate_hz=1.0, bandwidth_Bps=7000.0,  bit_error_rate=0.005, n_flows=5, obs_lag_steps=0),
    "stress_high_bg":    dict(event_rate_hz=1.0, bandwidth_Bps=6000.0,  bit_error_rate=0.02,  n_flows=5, obs_lag_steps=1),
    "no_event":          dict(event_rate_hz=0.0, bandwidth_Bps=9000.0,  bit_error_rate=0.01,  n_flows=3, obs_lag_steps=0),
}

CURRICULUM = {
    "easy": dict(delta_period_rel_max=0.03, delta_jitter_rel_max=0.01, delta_offset_rel_max=0.03, delta_bg_abs_max=0.03, delta_noise_abs_max=0.005, delta_scale_abs_max=0.03),
    "mid":  dict(delta_period_rel_max=0.05, delta_jitter_rel_max=0.02, delta_offset_rel_max=0.05, delta_bg_abs_max=0.05, delta_noise_abs_max=0.01,  delta_scale_abs_max=0.05),
    "hard": dict(delta_period_rel_max=0.10, delta_jitter_rel_max=0.04, delta_offset_rel_max=0.10, delta_bg_abs_max=0.08, delta_noise_abs_max=0.02,  delta_scale_abs_max=0.08),
}
