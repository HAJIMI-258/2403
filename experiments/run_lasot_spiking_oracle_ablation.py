"""Run normal/reappear/history oracle ablation for LaSOT spiking permanence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_lasot_spiking_permanence_eval import run_eval  # noqa: E402


MODES = ("normal", "oracle_reappear_only", "oracle_history_and_reappear")


def run_ablation(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_spiking_oracle_ablation",
    max_events: int = 50,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    category_filter: str = "",
    sequence_filter: str = "",
    max_image_side: int = 160,
    strict_min_iou: float = 0.25,
    frame_stride: int = 1,
    objectness_profile: str = "A8_quantile_q050_component_props48",
    attention_profile: str = "A10_source_spatial_diverse_max16",
    spike_dim: int = 128,
    max_capsules: int = 128,
    match_profile: str = "hash_chroma_deform",
    same_object_threshold: float = 0.90,
    same_object_margin_threshold: float = 0.14,
    false_resurrection_risk_threshold: float = 0.25,
    component_ranking_profile: str = "R0_current_quality",
    support_box_profile: str = "B0_refined_box_current",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mode_dir = out / mode
        summary = run_eval(
            root=root,
            output_dir=mode_dir,
            max_events=max_events,
            min_gap=min_gap,
            pre_context=pre_context,
            post_context=post_context,
            category_filter=category_filter,
            sequence_filter=sequence_filter,
            max_image_side=max_image_side,
            strict_min_iou=strict_min_iou,
            frame_stride=frame_stride,
            objectness_profile=objectness_profile,
            attention_profile=attention_profile,
            spike_dim=spike_dim,
            max_capsules=max_capsules,
            match_profile=match_profile,
            same_object_threshold=same_object_threshold,
            same_object_margin_threshold=same_object_margin_threshold,
            false_resurrection_risk_threshold=false_resurrection_risk_threshold,
            mode=mode,
            component_ranking_profile=component_ranking_profile,
            support_box_profile=support_box_profile,
        )
        summaries[mode] = summary
        (out / f"{mode}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    comparison = _comparison(summaries)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(comparison), encoding="utf-8")
    return {"summaries": summaries, "comparison": comparison}


def _comparison(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in MODES:
        summary = summaries.get(mode, {})
        output[f"{mode}_target_capsule_presence_rate"] = summary.get("target_capsule_presence_rate", 0.0)
        output[f"{mode}_target_capsule_top5_rate"] = summary.get("target_capsule_top5_rate", 0.0)
        output[f"{mode}_same_instance_recall_at_reentry"] = summary.get("same_instance_recall_at_reentry", 0.0)
        output[f"{mode}_false_resurrection_rate"] = summary.get("false_resurrection_rate_at_reentry", 0.0)
        output[f"{mode}_no_object_file_matched"] = summary.get("failure_buckets", {}).get("no_object_file_matched", 0)
    output["normal_vs_oracle_history_top5_delta"] = (
        output.get("oracle_history_and_reappear_target_capsule_top5_rate", 0.0)
        - output.get("normal_target_capsule_top5_rate", 0.0)
    )
    output["normal_vs_oracle_history_recall_delta"] = (
        output.get("oracle_history_and_reappear_same_instance_recall_at_reentry", 0.0)
        - output.get("normal_same_instance_recall_at_reentry", 0.0)
    )
    return output


def _report(comparison: dict[str, Any]) -> str:
    return (
        "# LaSOT Spiking Oracle Ablation\n\n"
        "Normal mode keeps online spiking capsule updates GT-free. Oracle modes "
        "use GT boxes for offline diagnostics only.\n\n"
        "```json\n"
        + json.dumps(comparison, indent=2, ensure_ascii=False)
        + "\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_spiking_oracle_ablation")
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--objectness-profile", default="A8_quantile_q050_component_props48")
    parser.add_argument("--attention-profile", default="A10_source_spatial_diverse_max16")
    parser.add_argument("--spike-dim", type=int, default=128)
    parser.add_argument("--max-capsules", type=int, default=128)
    parser.add_argument("--match-profile", default="hash_chroma_deform")
    parser.add_argument("--same-object-threshold", type=float, default=0.90)
    parser.add_argument("--same-object-margin-threshold", type=float, default=0.14)
    parser.add_argument("--false-resurrection-risk-threshold", type=float, default=0.25)
    parser.add_argument("--component-ranking-profile", default="R0_current_quality")
    parser.add_argument("--support-box-profile", default="B0_refined_box_current")
    args = parser.parse_args()
    result = run_ablation(**vars(args))
    print(json.dumps(result["comparison"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
