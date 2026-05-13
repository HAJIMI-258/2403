"""Memory-guided proposal profiles for external re-entry diagnostics."""

from __future__ import annotations

from typing import Any

from nops_owr.objectness.memory_guided_proposals import (
    MemoryGuidedProposalAugmenter,
    MemoryGuidedProposalConfig,
)


def memory_guided_profile_specs() -> list[dict[str, Any]]:
    return [
        {
            "profile_name": "M0_disabled",
            "enabled": False,
            "max_memory_episodes": 0,
            "windows_per_episode": 0,
            "max_added_proposals": 0,
            "max_total_proposals": 0,
            "nms_iou": 0.55,
            "min_episode_gap": 8,
        },
        {
            "profile_name": "M1_closed_episode_global_windows_k8",
            "enabled": True,
            "max_memory_episodes": 4,
            "windows_per_episode": 8,
            "max_added_proposals": 8,
            "max_total_proposals": 32,
            "nms_iou": 0.55,
            "min_episode_gap": 8,
        },
        {
            "profile_name": "M2_closed_episode_global_windows_k16",
            "enabled": True,
            "max_memory_episodes": 6,
            "windows_per_episode": 10,
            "max_added_proposals": 16,
            "max_total_proposals": 40,
            "nms_iou": 0.55,
            "min_episode_gap": 8,
        },
        {
            "profile_name": "M3_closed_episode_global_windows_k24",
            "enabled": True,
            "max_memory_episodes": 8,
            "windows_per_episode": 12,
            "max_added_proposals": 24,
            "max_total_proposals": 48,
            "nms_iou": 0.50,
            "min_episode_gap": 8,
        },
        {
            "profile_name": "M4_closed_episode_template_windows_k16",
            "enabled": True,
            "max_memory_episodes": 3,
            "windows_per_episode": 6,
            "max_added_proposals": 8,
            "max_total_proposals": 32,
            "nms_iou": 0.50,
            "min_episode_gap": 8,
            "use_template_matching": True,
            "template_match_count": 4,
            "template_stride_frac": 0.75,
        },
        {
            "profile_name": "M5_closed_episode_template_windows_k24",
            "enabled": True,
            "max_memory_episodes": 4,
            "windows_per_episode": 8,
            "max_added_proposals": 12,
            "max_total_proposals": 36,
            "nms_iou": 0.50,
            "min_episode_gap": 8,
            "use_template_matching": True,
            "template_match_count": 5,
            "template_stride_frac": 0.65,
        },
    ]


def get_memory_guided_profile(profile_name: str) -> dict[str, Any]:
    if profile_name in {"", "default", "disabled"}:
        profile_name = "M0_disabled"
    for profile in memory_guided_profile_specs():
        if profile["profile_name"] == profile_name:
            return dict(profile)
    known = ", ".join(profile["profile_name"] for profile in memory_guided_profile_specs())
    raise ValueError(f"Unknown memory-guided profile {profile_name!r}. Known profiles: {known}")


def build_memory_guided_augmenter(profile_name: str) -> MemoryGuidedProposalAugmenter | None:
    profile = get_memory_guided_profile(profile_name)
    if not bool(profile.pop("enabled", False)):
        return None
    profile.pop("profile_name", None)
    config = MemoryGuidedProposalConfig(**profile)
    return MemoryGuidedProposalAugmenter(config=config)
