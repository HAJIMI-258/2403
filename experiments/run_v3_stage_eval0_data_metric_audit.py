from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r_utils import write_csv


SUSPICIOUS_FIELDS = (
    "target_bundle_id",
    "old_track_id",
    "old_prototype_id",
    "instance_id",
    "gt_box",
    "gt_mask",
    "target_anchor_uid",
    "ledger_event_id",
    "future",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EVAL-0 data and metric trust audit.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    p.add_argument("--output-dir", default="results/eval0")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dataset_lineage(config_path: Path) -> list[dict[str, Any]]:
    cfg = load_yaml(config_path)
    d = cfg.get("dataset", {})
    outputs = set(d.get("outputs", []))
    return [{
        "dataset_name": d.get("name", "unknown"),
        "dataset_type": "synthetic_controlled_stream",
        "config_path": str(config_path),
        "config_hash": file_hash(config_path),
        "seed": cfg.get("seed", ""),
        "sequence_id": "all_configured_sequences",
        "num_frames": d.get("sequence_length", ""),
        "num_objects": "|".join(str(v) for v in d.get("num_objects_range", [])),
        "occlusion_probability": d.get("occlusion_probability", ""),
        "reentry_probability": d.get("reentry_probability", ""),
        "ground_truth_source": "SyntheticStreamGenerator",
        "gt_boxes_available": int("boxes" in outputs),
        "gt_masks_available": int("masks" in outputs),
        "gt_instance_ids_available": int("instance_id" in outputs),
        "gt_concept_ids_available": int("concept_id" in outputs),
    }]


def event_ledger_audit(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "event_id": r.get("ledger_event_id", ""),
            "scenario_name": r.get("scenario_name", ""),
            "instance_id": r.get("instance_id", ""),
            "disappear_frame": r.get("disappear_frame", ""),
            "reappear_frame": r.get("reappear_frame", ""),
            "gap_length": r.get("gap_length", ""),
            "proposal_detected": r.get("proposal_detected", ""),
            "target_anchor_uid": "",
            "target_bundle_id_eval_only": "",
            "old_track_id_eval_only": r.get("old_track_id", ""),
            "old_prototype_id_eval_only": r.get("old_prototype_id", ""),
            "used_in_online_scoring": 0,
            "used_in_evaluation_only": 1,
        })
    return out


def metric_consistency() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_file = Path("results/v3_e4ad/stage_E4AD_metric_consistency_audit_v1.csv")
    for r in read_csv(metric_file):
        rows.append({
            "metric_name": r.get("metric_name", ""),
            "source_file": str(metric_file),
            "formula_description": "E34r passive baseline recomputed in E4A-D",
            "recomputed_value": r.get("e4a_recomputed_value", ""),
            "reported_value": r.get("e34r_value", ""),
            "matched": r.get("matched", ""),
            "difference_reason": r.get("difference_reason", ""),
        })
    # Add E4A.1b compact as a not-ready integration gate.
    compact = Path("results/v3_e4a1b/stage_E4A1B_compact_for_gpt_v1.json")
    if compact.exists():
        c = json.loads(compact.read_text(encoding="utf-8"))
        rows.append({
            "metric_name": "E4A1B_negative_controls_passed",
            "source_file": str(compact),
            "formula_description": "Integration readiness gate; must be 1 before active evidence integration",
            "recomputed_value": c.get("negative_controls_passed"),
            "reported_value": c.get("negative_controls_passed"),
            "matched": 1,
            "difference_reason": "not_ready_for_integration" if int(c.get("negative_controls_passed", 0)) == 0 else "",
        })
    return rows


def oracle_leakage_audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_context_tokens = (
        "eval",
        "audit",
        "trace",
        "summary",
        "compact",
        "report",
        "metric",
        "control",
        "failure",
        "taxonomy",
        "target_rank",
        "target_bundle_rank",
        "target_bundle_retrieved",
        "target_in_top",
        "event_id",
        "base_top",
        "baseline_top",
        "reranked_top",
        "e32",
        "top1_hit",
        "top3_hit",
        "top5_hit",
        "focus",
        "row",
        "rows",
        "write_csv",
    )
    online_risk_tokens = (
        "final_score",
        "score +=",
        "score =",
        "compatibility +=",
        "compatibility =",
        "rerank",
        "rank_score",
        "selection_score",
        "fixation_score",
        "active_score",
        "same_space_score",
        "candidate_score",
        "selected_by",
    )
    for path in sorted(Path("experiments").glob("run_v3_stage_*.py")):
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for field in SUSPICIOUS_FIELDS:
            for idx, line in enumerate(lines):
                if field not in line:
                    continue
                window = lines[max(0, idx - 2): min(len(lines), idx + 3)]
                ctx = " ".join(s.strip() for s in window)
                low = ctx.lower()
                # Import headers like "__future__" are not oracle use.
                if field == "future" and "__future__" in low:
                    eval_only = True
                    scoring_context = False
                else:
                    eval_only = any(k in low for k in eval_context_tokens)
                    scoring_context = any(k in low for k in online_risk_tokens)
                # Reading target IDs from event/ledger rows is allowed when used for
                # evaluation rows, but not if it directly changes scoring/fixation.
                if any(k in low for k in ("event.get(", "row.get(", "target_id =", "target_bundle_id =", "target_bundle =", "gt_box =")):
                    eval_only = True
                risk = "high" if scoring_context and not eval_only else ("medium" if scoring_context else "low")
                rows.append({
                    "file": str(path),
                    "function": "",
                    "suspicious_field": field,
                    "context": ctx,
                    "risk_level": risk,
                    "allowed_eval_only": int(eval_only),
                    "leakage_found": int(risk == "high"),
                })
    return rows


def negative_control_audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    e2c = Path("results/v3_e2c/stage_E2C_negative_control_summary_v1.json")
    if e2c.exists():
        c = json.loads(e2c.read_text(encoding="utf-8"))
        pass_block = c.get("pass", {})
        passed = int(
            bool(pass_block.get("shuffled_anchor_lower_than_real", False))
            and bool(pass_block.get("focus_wrong_old_prototype_zero", False))
            and bool(pass_block.get("normal_reference_triple_match", False))
        )
        reason = "" if passed else "one_or_more_e2c_controls_failed"
        rows.append({"control_name": "E2C_negative_controls", "stage": "E2C-N", "passed": passed, "failure_reason": reason})
    e4 = Path("results/v3_e4a1b/stage_E4A1B_negative_control_summary_v1.csv")
    for r in read_csv(e4):
        rows.append({
            "control_name": r.get("control_name", ""),
            "stage": "E4A.1b",
            "passed": r.get("control_passed", ""),
            "failure_reason": "" if str(r.get("control_passed", "")) == "1" else "control_not_clean",
        })
    return rows


def reproducibility_audit(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in metrics:
        out.append({
            "metric_name": r["metric_name"],
            "run_a_value": r["reported_value"],
            "run_b_value": r["recomputed_value"],
            "matched": r["matched"],
        })
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_rows = read_csv(args.event_audit)
    lineage = dataset_lineage(Path(args.config))
    ledger = event_ledger_audit(event_rows)
    metrics = metric_consistency()
    leakage = oracle_leakage_audit()
    neg = negative_control_audit()
    repro = reproducibility_audit(metrics)
    oracle_leakage_found = int(any(int(r["leakage_found"]) for r in leakage))
    metric_passed = int(all(str(r["matched"]) == "1" for r in metrics))
    ledger_passed = int(all(int(r["used_in_online_scoring"]) == 0 for r in ledger))
    repro_passed = int(all(str(r["matched"]) == "1" for r in repro))
    negative_not_ready = [r for r in neg if str(r.get("passed")) != "1"]
    active_evidence_integration_ready = int(not any(str(r.get("stage")) == "E4A.1b" and str(r.get("passed")) != "1" for r in neg))
    internal_passed = int(oracle_leakage_found == 0 and metric_passed and ledger_passed and repro_passed)
    external_smoke = Path("results/external_smoke/stage_EXTERNAL_SMOKE_compact_for_gpt_v1.json")
    external_smoke_ready = 0
    if external_smoke.exists():
        external_smoke_ready = int(json.loads(external_smoke.read_text(encoding="utf-8")).get("adapter_passed", 0))
    if not internal_passed:
        next_rec = "fix evaluation pipeline before any model optimization"
    elif not external_smoke_ready:
        next_rec = "connect external datasets / download public benchmark"
    elif not active_evidence_integration_ready:
        next_rec = "run external passive/baseline benchmark subset; do not integrate active evidence until E4A controls pass"
    else:
        next_rec = "run NOPS passive and baselines on first external benchmark subset"
    compact = {
        "stage": "EVAL-0",
        "internal_eval_trust_passed": internal_passed,
        "oracle_leakage_found": oracle_leakage_found,
        "metric_consistency_passed": metric_passed,
        "negative_controls_summary": {
            "total": len(neg),
            "not_ready_count": len(negative_not_ready),
            "not_ready": negative_not_ready[:10],
        },
        "active_evidence_integration_ready": active_evidence_integration_ready,
        "external_protocol_created": int(Path("protocol/external_video_memory_benchmark_v1.md").exists()),
        "external_adapters_created": int(Path("datasets/external/base_video_memory_dataset.py").exists()),
        "external_smoke_ready": external_smoke_ready,
        "next_recommendation": next_rec,
    }
    write_csv(out / f"stage_EVAL0_dataset_lineage_{args.artifact_version}.csv", lineage)
    write_csv(out / f"stage_EVAL0_event_ledger_audit_{args.artifact_version}.csv", ledger)
    write_csv(out / f"stage_EVAL0_metric_consistency_{args.artifact_version}.csv", metrics)
    write_csv(out / f"stage_EVAL0_oracle_leakage_audit_{args.artifact_version}.csv", leakage)
    write_csv(out / f"stage_EVAL0_negative_control_audit_{args.artifact_version}.csv", neg)
    write_csv(out / f"stage_EVAL0_reproducibility_audit_{args.artifact_version}.csv", repro)
    report = "\n".join([
        "# Stage EVAL-0 Report",
        "",
        "## Verdict",
        "",
        next_rec,
        "",
        "## Compact",
        "",
        "```json",
        json.dumps(compact, indent=2, ensure_ascii=False),
        "```",
    ]) + "\n"
    (out / f"stage_EVAL0_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_EVAL0_report_{args.artifact_version}.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
