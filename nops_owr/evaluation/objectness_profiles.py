"""Named objectness profiles for external evaluation.

Profiles are evaluation-time configurations. Selecting a profile here does not
change the default NOPS objectness field or merge a real-video calibration into
the main model.
"""

from __future__ import annotations

from typing import Any

from nops_owr.objectness.field import MinimalObjectnessField


def objectness_profile_specs() -> list[dict[str, Any]]:
    return [
        {
            "profile_name": "A0_current_fixed_tau035_area16_props8",
            "tau_obj": 0.35,
            "threshold_mode": "fixed",
            "min_area": 16,
            "max_proposals": 8,
            "smoothing_kernel_size": 5,
        },
        {
            "profile_name": "A1_fixed_tau025_area12_props12",
            "tau_obj": 0.25,
            "threshold_mode": "fixed",
            "min_area": 12,
            "max_proposals": 12,
            "smoothing_kernel_size": 5,
        },
        {
            "profile_name": "A2_fixed_tau020_area8_props16",
            "tau_obj": 0.20,
            "threshold_mode": "fixed",
            "min_area": 8,
            "max_proposals": 16,
            "smoothing_kernel_size": 5,
        },
        {
            "profile_name": "A3_fixed_tau015_area8_props24",
            "tau_obj": 0.15,
            "threshold_mode": "fixed",
            "min_area": 8,
            "max_proposals": 24,
            "smoothing_kernel_size": 5,
        },
        {
            "profile_name": "A4_quantile_q070_k020_area8_props16",
            "threshold_mode": "quantile_local",
            "q_obj": 0.70,
            "local_k": 0.20,
            "min_area": 8,
            "max_proposals": 16,
            "smoothing_kernel_size": 5,
        },
        {
            "profile_name": "A5_quantile_q060_k000_area8_props24",
            "threshold_mode": "quantile_local",
            "q_obj": 0.60,
            "local_k": 0.00,
            "min_area": 8,
            "max_proposals": 24,
            "smoothing_kernel_size": 3,
        },
    ]


def get_objectness_profile(profile_name: str) -> dict[str, Any]:
    if profile_name in {"", "default", "current"}:
        profile_name = "A0_current_fixed_tau035_area16_props8"
    for profile in objectness_profile_specs():
        if profile["profile_name"] == profile_name:
            return dict(profile)
    known = ", ".join(profile["profile_name"] for profile in objectness_profile_specs())
    raise ValueError(f"Unknown objectness profile {profile_name!r}. Known profiles: {known}")


def build_objectness_from_profile(profile_name: str) -> MinimalObjectnessField:
    kwargs = get_objectness_profile(profile_name)
    kwargs.pop("profile_name", None)
    return MinimalObjectnessField(**kwargs)
