"""Shared scenario presets for Phase 1/2 experiments."""

from __future__ import annotations

import copy


def build_phase1_scenarios(base_config):
    scenarios = []

    easy = copy.deepcopy(base_config)
    easy.sequence_length = 360
    easy.num_objects_range = (1, 1)
    easy.occlusion_probability = 0.0
    easy.reentry_probability = 0.0
    easy.background_drift.brightness_amplitude = 0.04
    easy.background_drift.texture_noise_std = 0.015
    scenarios.append({"name": "easy_single_object", "config": easy})

    reentry = copy.deepcopy(base_config)
    reentry.sequence_length = 420
    reentry.num_objects_range = (3, 3)
    reentry.occlusion_probability = 0.45
    reentry.reentry_probability = 0.85
    scenarios.append({"name": "multi_object_reentry", "config": reentry})

    hard = build_hard_drift_occlusion_config(base_config)
    scenarios.append({"name": "hard_drift_occlusion", "config": hard})

    return scenarios


def build_hard_drift_occlusion_config(base_config):
    hard = copy.deepcopy(base_config)
    hard.sequence_length = 420
    hard.num_objects_range = (3, 3)
    hard.occlusion_probability = 1.0
    hard.reentry_probability = 1.0
    hard.background_drift.brightness_amplitude = 0.18
    hard.background_drift.texture_noise_std = 0.05
    hard.appearance_perturbation.scale_jitter = 0.16
    hard.appearance_perturbation.intensity_jitter = 0.12
    return hard


def build_phase3_track_scenarios(base_config):
    scenarios = []

    track_a = copy.deepcopy(base_config)
    track_a.sequence_length = max(900, base_config.sequence_length)
    track_a.num_objects_range = (2, 4)
    track_a.occlusion_probability = max(0.85, track_a.occlusion_probability)
    track_a.reentry_probability = max(0.85, track_a.reentry_probability)
    track_a.bridge_synthetic.enabled = True
    track_a.bridge_synthetic.difficulty_preset = "track_a_bridge"
    track_a.bridge_synthetic.occlusion_duration_range = (28, 72)
    track_a.bridge_synthetic.reentry_gap_range = (20, 80)
    track_a.bridge_synthetic.crossing_probability = max(0.65, track_a.bridge_synthetic.crossing_probability)
    scenarios.append({"name": "track_a_bridge", "config": track_a})

    track_c = copy.deepcopy(base_config)
    track_c.sequence_length = max(1800, base_config.sequence_length * 2)
    track_c.num_objects_range = (3, 6)
    track_c.occlusion_probability = max(0.92, track_c.occlusion_probability)
    track_c.reentry_probability = max(0.92, track_c.reentry_probability)
    track_c.bridge_synthetic.enabled = True
    track_c.bridge_synthetic.difficulty_preset = "track_c_long_horizon"
    track_c.bridge_synthetic.occlusion_duration_range = (48, 140)
    track_c.bridge_synthetic.reentry_gap_range = (48, 180)
    track_c.bridge_synthetic.crossing_probability = max(0.75, track_c.bridge_synthetic.crossing_probability)
    track_c.bridge_synthetic.camera_jitter_std = max(1.8, track_c.bridge_synthetic.camera_jitter_std)
    scenarios.append({"name": "track_c_long_horizon", "config": track_c})

    return scenarios
