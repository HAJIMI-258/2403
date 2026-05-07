"""Freeze the current Phase 2B best config and result snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_FILES = [
    "configs/synth.yaml",
    "results/phase2b_final/scenario_summary_v2.csv",
    "results/phase2b_final/scenario_summary_v2.json",
    "results/phase2b_final/baseline_comparison_v2.csv",
    "results/phase2b_final/baseline_comparison_v2.json",
    "results/phase2b_final/go_no_go_summary_v2.md",
    "results/phase2b_final/tracking_reentry_eval_v1.csv",
    "results/phase2b_final/tracking_reentry_eval_v1.json",
    "results/phase2b_final/tracking_reentry_eval_v1.png",
    "results/phase2b_final/tracking_seq_003.png",
    "results/phase2b_final/tracking_seq_003.json",
    "results/phase2b_final/memory_seq_003.png",
    "results/phase2b_final/memory_seq_003.json",
    "results/phase2b_final_failure_slicing/failure_slices_v2.csv",
    "results/phase2b_final_failure_slicing/failure_slices_v2.json",
    "results/phase2b_final_failure_slicing/failure_slices_v2.png",
    "results/phase2b_adaptive_threshold/adaptive_threshold_compare.csv",
    "results/phase2b_adaptive_threshold/adaptive_threshold_compare.json",
    "results/phase2b_adaptive_threshold/adaptive_threshold_compare.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the current Phase 2B best snapshot.")
    parser.add_argument("--output-dir", default="results/phase2b_best", help="Directory for the frozen snapshot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []

    for rel_path in DEFAULT_FILES:
        source = root / rel_path
        if not source.exists():
            missing.append(rel_path)
            continue
        target = output_dir / rel_path.replace("/", "__")
        shutil.copy2(source, target)
        copied.append(rel_path)

    manifest = {
        "snapshot_name": "phase2b_best",
        "source_root": str(root),
        "copied_files": copied,
        "missing_files": missing,
    }
    (output_dir / "manifest_phase2b_best.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Phase 2B Best Snapshot",
                "",
                "This directory freezes the current Phase 2B best config and the result artifacts used as the base for Phase 3.",
                "",
                "Frozen source files:",
                *[f"- `{path}`" for path in copied],
            ]
        ),
        encoding="utf-8",
    )

    print(f"snapshot_dir={output_dir}")
    print(f"copied_files={len(copied)}")
    if missing:
        print("missing_files=" + ",".join(missing))


if __name__ == "__main__":
    main()
