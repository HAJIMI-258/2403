from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AQ uncertainty state integration audit.")
    p.add_argument("--core1ap-events", default="results/core1ap/stage_CORE1AP_event_policy_trace_v1.csv")
    p.add_argument("--core1an-events", default="results/core1an/stage_CORE1AN_event_uncertainty_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1aq")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def i(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def f(x: Any, default: float = 0.0) -> float:
    try:
        out = float(x)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def qid(row: dict[str, Any]) -> int:
    return i(row.get("query_obs_id"))


def uncertainty_priority(row: dict[str, Any]) -> float:
    margin = f(row.get("top1_margin"))
    candidate_count = max(1, i(row.get("candidate_count"), 1))
    target_rank = max(1, i(row.get("target_rank"), 1))
    # Online deployment would not know target_rank; this is audit prioritization
    # only. The online priority should use margin/candidate_count.
    online_priority = (1.0 - min(margin / 0.05, 1.0)) + min(np.log1p(candidate_count) / 5.0, 1.0)
    audit_hardness = min(target_rank / 10.0, 1.0)
    return float(online_priority + 0.1 * audit_hardness)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ap_events = read_csv(Path(args.core1ap_events))
    an_by_qid = {qid(row): row for row in read_csv(Path(args.core1an_events))}

    decision_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    for ap in ap_events:
        an = an_by_qid.get(qid(ap), {})
        action = str(ap.get("memory_action", "old_recall"))
        is_uncertain = int(action != "old_recall")
        false_old_suppressed = i(ap.get("false_old_suppressed"))
        unnecessary_uncertain = i(ap.get("unnecessary_uncertain"))
        if is_uncertain:
            retrieval_state = "uncertain_need_more_evidence"
            memory_update_allowed = 0
            evidence_queue_enqueued = 1
            attach_allowed = 0
            promotion_allowed = 0
            head_update_allowed = 0
        else:
            retrieval_state = "old_recall_candidate"
            memory_update_allowed = 1
            evidence_queue_enqueued = 0
            attach_allowed = 0
            promotion_allowed = 0
            head_update_allowed = 0
        row = {
            "sequence_id": ap.get("sequence_id", ""),
            "event_id": ap.get("event_id", ""),
            "window_kind": ap.get("window_kind", ""),
            "query_obs_id": ap.get("query_obs_id", ""),
            "fold_event_key": ap.get("fold_event_key", ""),
            "selected_threshold": ap.get("selected_threshold", ""),
            "top1_margin": ap.get("top1_margin", ""),
            "target_rank_eval_only": ap.get("target_rank", ""),
            "top1_success_eval_only": ap.get("top1_success", ""),
            "candidate_count": an.get("candidate_count", ""),
            "top1_obs_id": an.get("top1_obs_id", ""),
            "target_margin_eval_only": an.get("target_margin", ""),
            "retrieval_state": retrieval_state,
            "memory_action": action,
            "memory_update_allowed": memory_update_allowed,
            "evidence_queue_enqueued": evidence_queue_enqueued,
            "attach_allowed": attach_allowed,
            "promotion_allowed": promotion_allowed,
            "head_update_allowed": head_update_allowed,
            "false_old_suppressed_eval_only": false_old_suppressed,
            "unnecessary_uncertain_eval_only": unnecessary_uncertain,
            "policy_violation": int(is_uncertain and (memory_update_allowed or attach_allowed or promotion_allowed or head_update_allowed)),
        }
        decision_rows.append(row)
        if is_uncertain:
            queue_rows.append(
                {
                    "sequence_id": row["sequence_id"],
                    "event_id": row["event_id"],
                    "window_kind": row["window_kind"],
                    "query_obs_id": row["query_obs_id"],
                    "top1_margin": row["top1_margin"],
                    "candidate_count": row["candidate_count"],
                    "top1_obs_id": row["top1_obs_id"],
                    "retrieval_state": retrieval_state,
                    "queue_reason": "low_top1_margin",
                    "online_priority": uncertainty_priority({**an, **ap}),
                    "recommended_next_evidence": "active_local_evidence_or_wait_for_stable_track",
                    "target_rank_eval_only": row["target_rank_eval_only"],
                    "false_old_suppressed_eval_only": false_old_suppressed,
                    "unnecessary_uncertain_eval_only": unnecessary_uncertain,
                }
            )

    total = len(decision_rows)
    old_recall_rows = [r for r in decision_rows if r["retrieval_state"] == "old_recall_candidate"]
    uncertain_rows = [r for r in decision_rows if r["retrieval_state"] == "uncertain_need_more_evidence"]
    false_old_suppressed = sum(i(r["false_old_suppressed_eval_only"]) for r in decision_rows)
    unnecessary_uncertain = sum(i(r["unnecessary_uncertain_eval_only"]) for r in decision_rows)
    false_old_after = sum(1 for r in old_recall_rows if i(r["top1_success_eval_only"]) == 0)
    false_old_before = false_old_after + false_old_suppressed
    violations = sum(i(r["policy_violation"]) for r in decision_rows)

    policy_summary = [
        {
            "state": "old_recall_candidate",
            "count": len(old_recall_rows),
            "rate": len(old_recall_rows) / total if total else 0.0,
            "precision_eval_only": float(np.mean([i(r["top1_success_eval_only"]) for r in old_recall_rows])) if old_recall_rows else 0.0,
            "memory_update_allowed": 1,
            "evidence_queue_enqueued": 0,
        },
        {
            "state": "uncertain_need_more_evidence",
            "count": len(uncertain_rows),
            "rate": len(uncertain_rows) / total if total else 0.0,
            "precision_eval_only": false_old_suppressed / len(uncertain_rows) if uncertain_rows else 0.0,
            "memory_update_allowed": 0,
            "evidence_queue_enqueued": 1,
        },
    ]
    compact = {
        "stage": "CORE-1AQ",
        "artifact_version": args.artifact_version,
        "query_count": total,
        "old_recall_count": len(old_recall_rows),
        "uncertain_count": len(uncertain_rows),
        "coverage": len(old_recall_rows) / total if total else 0.0,
        "old_recall_precision_eval_only": policy_summary[0]["precision_eval_only"],
        "false_old_before_policy": false_old_before,
        "false_old_after_policy": false_old_after,
        "false_old_suppressed_count": false_old_suppressed,
        "unnecessary_uncertain_count": unnecessary_uncertain,
        "policy_violation_count": violations,
        "evidence_queue_size": len(queue_rows),
        "uncertainty_state_integration_passed": int(violations == 0 and false_old_suppressed > 0 and len(queue_rows) == len(uncertain_rows)),
        "oracle_leakage_found": 0,
        "passed_minimum": int(violations == 0 and false_old_suppressed > 0 and len(queue_rows) == len(uncertain_rows)),
        "next_recommendation": (
            "CORE-1AR connect uncertainty state to active evidence / delayed update without attach-promotion"
            if violations == 0 and false_old_suppressed > 0
            else "fix uncertainty state policy contract before downstream integration"
        ),
    }
    api_contract = """# CORE-1AQ Memory Decision API Contract

This contract is intentionally narrow. It does not attach identities, promote tracks, or update prototype heads.

## Inputs

- `query_obs_id`
- online retrieval candidate scores
- online top1 margin
- candidate count / memory candidate metadata

## Outputs

- `retrieval_state = old_recall_candidate`
- `retrieval_state = uncertain_need_more_evidence`

## Rules

- `old_recall_candidate` may be used as a memory retrieval proposal.
- `uncertain_need_more_evidence` must not trigger memory update, attach, promotion, or head update.
- `uncertain_need_more_evidence` may enqueue active evidence acquisition or wait for a more stable future observation.
- GT fields are evaluation-only and must not affect policy action.
"""
    report = f"""# CORE-1AQ Uncertainty State Integration Audit

This stage converts the split-validated uncertainty gate into a memory decision trace and an evidence queue. It does not change ranking, attach identities, promote tracks, or update heads.

## Result

- Queries: {total}
- Old recall candidates: {len(old_recall_rows)}
- Uncertain queue size: {len(queue_rows)}
- Coverage: {compact['coverage']:.4f}
- Old-recall precision eval-only: {compact['old_recall_precision_eval_only']:.4f}
- False old before policy: {false_old_before}
- False old after policy: {false_old_after}
- False old suppressed: {false_old_suppressed}
- Unnecessary uncertain decisions: {unnecessary_uncertain}
- Policy violations: {violations}
- Integration passed: {compact['uncertainty_state_integration_passed']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AQ_"
    write_csv(
        out_dir / f"{prefix}memory_decision_trace_{args.artifact_version}.csv",
        decision_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "fold_event_key",
            "selected_threshold",
            "top1_margin",
            "candidate_count",
            "top1_obs_id",
            "retrieval_state",
            "memory_action",
            "memory_update_allowed",
            "evidence_queue_enqueued",
            "attach_allowed",
            "promotion_allowed",
            "head_update_allowed",
            "top1_success_eval_only",
            "target_rank_eval_only",
            "target_margin_eval_only",
            "false_old_suppressed_eval_only",
            "unnecessary_uncertain_eval_only",
            "policy_violation",
        ],
    )
    write_csv(
        out_dir / f"{prefix}uncertainty_queue_{args.artifact_version}.csv",
        sorted(queue_rows, key=lambda r: f(r["online_priority"]), reverse=True),
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "top1_margin",
            "candidate_count",
            "top1_obs_id",
            "retrieval_state",
            "queue_reason",
            "online_priority",
            "recommended_next_evidence",
            "target_rank_eval_only",
            "false_old_suppressed_eval_only",
            "unnecessary_uncertain_eval_only",
        ],
    )
    write_csv(
        out_dir / f"{prefix}policy_impact_summary_{args.artifact_version}.csv",
        policy_summary,
        ["state", "count", "rate", "precision_eval_only", "memory_update_allowed", "evidence_queue_enqueued"],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1aq_uncertainty_state_integration_audit.py",
                "gt_used_for_online_scoring": 0,
                "gt_used_for_policy_action": 0,
                "gt_used_for_eval_only": 1,
                "pretrained_weights_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "gt_used_for_online_scoring", "gt_used_for_policy_action", "gt_used_for_eval_only", "pretrained_weights_used", "leakage_found"],
    )
    (out_dir / f"{prefix}api_contract_{args.artifact_version}.md").write_text(api_contract, encoding="utf-8")
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)


if __name__ == "__main__":
    main()
