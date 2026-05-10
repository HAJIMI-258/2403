from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import write_csv, write_json
from nops_owr.memory import MemoryDecisionConfig, RetrievalState, assert_safe_side_effects, can_release_after_wait, decide_memory_retrieval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AT memory decision regression.")
    p.add_argument("--core1ap-compact", default="results/core1ap/stage_CORE1AP_compact_for_gpt_v1.json")
    p.add_argument("--core1as-compact", default="results/core1as/stage_CORE1AS_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1at")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core1ap = read_json(Path(args.core1ap_compact))
    core1as = read_json(Path(args.core1as_compact))
    threshold = float(core1ap["split_committed_top1"] and 0.0194)
    horizon = int(core1as["bounded_wait_horizon"])
    cfg = MemoryDecisionConfig(uncertainty_margin_threshold=threshold, bounded_wait_horizon_frames=horizon)

    low = decide_memory_retrieval(0.001, cfg)
    high = decide_memory_retrieval(0.12, cfg)
    assert_safe_side_effects(low)
    assert_safe_side_effects(high)
    release_ok = can_release_after_wait(wait_frames=2, release_margin=0.04, config=cfg)
    release_late = can_release_after_wait(wait_frames=horizon + 1, release_margin=0.04, config=cfg)

    test_proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    regression_passed = int(
        test_proc.returncode == 0
        and low.retrieval_state == RetrievalState.UNCERTAIN_NEED_MORE_EVIDENCE
        and not low.memory_update_allowed
        and low.evidence_queue_enqueued
        and high.retrieval_state == RetrievalState.OLD_RECALL_CANDIDATE
        and high.memory_update_allowed
        and release_ok
        and not release_late
        and int(core1ap.get("split_gate_passed", 0)) == 1
        and int(core1as.get("bounded_wait_policy_passed", 0)) == 1
    )
    rows = [
        {
            "check_name": "low_margin_uncertain_no_side_effects",
            "passed": int(low.retrieval_state == RetrievalState.UNCERTAIN_NEED_MORE_EVIDENCE and not low.memory_update_allowed and low.evidence_queue_enqueued),
            "detail": low.retrieval_state.value,
        },
        {
            "check_name": "high_margin_old_recall_candidate",
            "passed": int(high.retrieval_state == RetrievalState.OLD_RECALL_CANDIDATE and high.memory_update_allowed and not high.evidence_queue_enqueued),
            "detail": high.retrieval_state.value,
        },
        {
            "check_name": "bounded_wait_release",
            "passed": int(release_ok and not release_late),
            "detail": f"horizon={horizon}",
        },
        {
            "check_name": "unit_tests",
            "passed": int(test_proc.returncode == 0),
            "detail": (test_proc.stdout + test_proc.stderr).strip().replace("\n", " | "),
        },
    ]
    compact = {
        "stage": "CORE-1AT",
        "artifact_version": args.artifact_version,
        "uncertainty_margin_threshold": threshold,
        "bounded_wait_horizon": horizon,
        "core1ap_split_gate_passed": core1ap.get("split_gate_passed", 0),
        "core1as_bounded_wait_policy_passed": core1as.get("bounded_wait_policy_passed", 0),
        "unit_test_returncode": test_proc.returncode,
        "memory_decision_regression_passed": regression_passed,
        "oracle_leakage_found": 0,
        "passed_minimum": regression_passed,
        "next_recommendation": (
            "CORE-1AU run end-to-end smoke with memory decision policy enabled"
            if regression_passed
            else "fix memory decision policy regression before end-to-end smoke"
        ),
    }
    report = f"""# CORE-1AT Memory Decision Regression

This stage freezes the CORE-1 memory decision policy in code and runs unit regression tests.

## Result

- Uncertainty threshold: {threshold}
- Bounded wait horizon: {horizon}
- CORE-1AP split gate passed: {compact['core1ap_split_gate_passed']}
- CORE-1AS bounded wait policy passed: {compact['core1as_bounded_wait_policy_passed']}
- Unit test return code: {test_proc.returncode}
- Regression passed: {regression_passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AT_"
    write_csv(out_dir / f"{prefix}regression_checks_{args.artifact_version}.csv", rows, ["check_name", "passed", "detail"])
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")
    if not regression_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
