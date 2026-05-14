"""GT-free support/refinement box profiles for external diagnostics."""

from __future__ import annotations

from dataclasses import replace

from nops_owr.objectness.field import Proposal

Box = tuple[int, int, int, int]


def support_refinement_profile_specs() -> list[str]:
    return [
        "B0_refined_box_current",
        "B1_raw_box_eval_profile",
        "B2_support_box_eval_profile",
        "B3_blend_raw_refined_50",
        "B4_expand_refined_10",
    ]


def box_for_proposal(proposal: Proposal, profile_name: str, frame_shape: tuple[int, int]) -> Box:
    if profile_name in {"", "default", "B0_refined_box_current"}:
        return proposal.box
    if profile_name == "B1_raw_box_eval_profile":
        return _clip_box(proposal.raw_box, frame_shape)
    if profile_name == "B2_support_box_eval_profile":
        return _clip_box(proposal.support_box, frame_shape)
    if profile_name == "B3_blend_raw_refined_50":
        return _blend_boxes(proposal.box, proposal.raw_box, 0.50, frame_shape)
    if profile_name == "B4_expand_refined_10":
        return _expand_box(proposal.box, 0.10, frame_shape)
    known = ", ".join(support_refinement_profile_specs())
    raise ValueError(f"Unknown support box profile {profile_name!r}. Known profiles: {known}")


def apply_box_profile_to_proposals(
    proposals: list[Proposal],
    profile_name: str = "B0_refined_box_current",
    frame_shape: tuple[int, int] | None = None,
) -> list[Proposal]:
    if profile_name in {"", "default", "B0_refined_box_current"}:
        return list(proposals)
    if frame_shape is None:
        frame_shape = _infer_frame_shape(proposals)
    updated: list[Proposal] = []
    for proposal in proposals:
        box = box_for_proposal(proposal, profile_name, frame_shape)
        updated.append(
            replace(
                proposal,
                box=box,
                centroid=((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5),
                metadata={**dict(proposal.metadata), "support_box_profile": profile_name},
            )
        )
    return updated


def _blend_boxes(refined_box: Box, raw_box: Box, raw_weight: float, frame_shape: tuple[int, int]) -> Box:
    raw_weight = max(0.0, min(float(raw_weight), 1.0))
    refined_weight = 1.0 - raw_weight
    box = (
        int(round(refined_weight * refined_box[0] + raw_weight * raw_box[0])),
        int(round(refined_weight * refined_box[1] + raw_weight * raw_box[1])),
        int(round(refined_weight * refined_box[2] + raw_weight * raw_box[2])),
        int(round(refined_weight * refined_box[3] + raw_weight * raw_box[3])),
    )
    return _clip_box(box, frame_shape)


def _expand_box(box: Box, frac: float, frame_shape: tuple[int, int]) -> Box:
    x1, y1, x2, y2 = box
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    dx = int(round(width * float(frac)))
    dy = int(round(height * float(frac)))
    return _clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), frame_shape)


def _clip_box(box: Box, frame_shape: tuple[int, int]) -> Box:
    height, width = frame_shape
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, max(0, width - 1)))
    y1 = max(0, min(y1, max(0, height - 1)))
    x2 = max(x1 + 1, min(int(x2), int(width)))
    y2 = max(y1 + 1, min(int(y2), int(height)))
    return (x1, y1, x2, y2)


def _infer_frame_shape(proposals: list[Proposal]) -> tuple[int, int]:
    max_x = max((max(proposal.box[2], proposal.raw_box[2], proposal.support_box[2]) for proposal in proposals), default=1)
    max_y = max((max(proposal.box[3], proposal.raw_box[3], proposal.support_box[3]) for proposal in proposals), default=1)
    return (max_y, max_x)
