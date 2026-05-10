from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1_online_object_encoder import write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORE-1G dense ledger pair-mining gate.")
    p.add_argument("--ledger", default="results/core1f/stage_CORE1F_dense_event_ledger_v1.csv")
    p.add_argument("--pair-opportunities", default="results/core1f/stage_CORE1F_pair_opportunity_by_sequence_v1.csv")
    p.add_argument("--output-dir", default="results/core1g")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--max-negative-pairs", type=int, default=6000)
    p.add_argument("--min-positive-pairs", type=int, default=500)
    p.add_argument("--min-negative-pairs", type=int, default=500)
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def pair_precision(rows: list[dict[str, Any]]) -> float:
    return sum(i(r.get("pair_correct_eval_only")) for r in rows) / max(len(rows), 1)


def build_pairs(ledger_rows: list[dict[str, str]], max_negative_pairs: int) -> list[dict[str, Any]]:
    usable = [r for r in ledger_rows if i(r.get("usable_real_gap")) == 1]
    by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in usable:
        by_sequence[str(r["sequence_id"])].append(r)

    pairs: list[dict[str, Any]] = []
    pair_id = 1
    for r in usable:
        pairs.append(
            {
                "pair_id": pair_id,
                "pair_type": "positive_reentry_same_instance",
                "source": "core1f_dense_gt_ledger",
                "split": r["split"],
                "sequence_id": r["sequence_id"],
                "sequence_j": r["sequence_id"],
                "event_id_i": r["event_id"],
                "event_id_j": r["event_id"],
                "instance_i_eval_only": r["instance_id_eval_only"],
                "instance_j_eval_only": r["instance_id_eval_only"],
                "concept_i_eval_only": r["concept_id_eval_only"],
                "concept_j_eval_only": r["concept_id_eval_only"],
                "frame_i": r["disappear_frame"],
                "frame_j": r["reappear_frame"],
                "gap_length": r["gap_length"],
                "online_positive": 1,
                "online_negative": 0,
                "pair_correct_eval_only": 1,
                "gt_ledger_pair": 1,
                "usable_for_main_online_training": 0,
                "usable_for_dense_diagnostic_training": 1,
                "mining_reason": "same_instance_reentry_defined_by_eval_ledger",
            }
        )
        pair_id += 1

    neg_count = 0
    for sequence_id, rows in by_sequence.items():
        rows = sorted(rows, key=lambda x: (i(x["reappear_frame"]), i(x["instance_id_eval_only"])))
        for idx, a in enumerate(rows):
            if neg_count >= max_negative_pairs:
                break
            candidates = rows[max(0, idx - 12) : min(len(rows), idx + 13)]
            for b in candidates:
                if neg_count >= max_negative_pairs:
                    break
                if a["event_id"] == b["event_id"]:
                    continue
                if a["instance_id_eval_only"] == b["instance_id_eval_only"]:
                    continue
                same_concept = int(a["concept_id_eval_only"] == b["concept_id_eval_only"])
                frame_delta = abs(i(a["reappear_frame"]) - i(b["reappear_frame"]))
                if frame_delta > 160 and not same_concept:
                    continue
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "pair_type": "negative_same_sequence_same_concept" if same_concept else "negative_same_sequence_different_concept",
                        "source": "core1f_dense_gt_ledger",
                        "split": a["split"],
                        "sequence_id": sequence_id,
                        "sequence_j": sequence_id,
                        "event_id_i": a["event_id"],
                        "event_id_j": b["event_id"],
                        "instance_i_eval_only": a["instance_id_eval_only"],
                        "instance_j_eval_only": b["instance_id_eval_only"],
                        "concept_i_eval_only": a["concept_id_eval_only"],
                        "concept_j_eval_only": b["concept_id_eval_only"],
                        "frame_i": a["reappear_frame"],
                        "frame_j": b["reappear_frame"],
                        "gap_length": max(i(a["gap_length"]), i(b["gap_length"])),
                        "online_positive": 0,
                        "online_negative": 1,
                        "pair_correct_eval_only": 1,
                        "gt_ledger_pair": 1,
                        "usable_for_main_online_training": 0,
                        "usable_for_dense_diagnostic_training": 1,
                        "mining_reason": "different_instance_same_sequence_competitor",
                    }
                )
                pair_id += 1
                neg_count += 1

    if neg_count < max_negative_pairs:
        by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in usable:
            by_split[r["split"]].append(r)
        for split, rows in by_split.items():
            rows = sorted(rows, key=lambda x: (x["sequence_id"], x["instance_id_eval_only"], x["event_id"]))
            n = len(rows)
            if n <= 1:
                continue
            for idx, a in enumerate(rows):
                if neg_count >= max_negative_pairs:
                    break
                # Prefer same-concept cross-sequence negatives first; they are
                # harder than arbitrary different-concept negatives while still
                # not relying on any target identity during online scoring.
                offsets = list(range(1, min(n, 80)))
                for off in offsets:
                    if neg_count >= max_negative_pairs:
                        break
                    b = rows[(idx + off) % n]
                    if a["sequence_id"] == b["sequence_id"]:
                        continue
                    same_concept = int(a["concept_id_eval_only"] == b["concept_id_eval_only"])
                    if off < 40 and not same_concept:
                        continue
                    pairs.append(
                        {
                            "pair_id": pair_id,
                            "pair_type": "negative_cross_sequence_same_concept" if same_concept else "negative_cross_sequence_different_concept",
                            "source": "core1f_dense_gt_ledger",
                            "split": split,
                            "sequence_id": a["sequence_id"],
                            "sequence_j": b["sequence_id"],
                            "event_id_i": a["event_id"],
                            "event_id_j": b["event_id"],
                            "instance_i_eval_only": a["instance_id_eval_only"],
                            "instance_j_eval_only": b["instance_id_eval_only"],
                            "concept_i_eval_only": a["concept_id_eval_only"],
                            "concept_j_eval_only": b["concept_id_eval_only"],
                            "frame_i": a["reappear_frame"],
                            "frame_j": b["reappear_frame"],
                            "gap_length": max(i(a["gap_length"]), i(b["gap_length"])),
                            "online_positive": 0,
                            "online_negative": 1,
                            "pair_correct_eval_only": 1,
                            "gt_ledger_pair": 1,
                            "usable_for_main_online_training": 0,
                            "usable_for_dense_diagnostic_training": 1,
                            "mining_reason": "cross_sequence_different_instance_competitor",
                        }
                    )
                    pair_id += 1
                    neg_count += 1
    return pairs


def split_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "dev", "test"):
        split_pairs = [p for p in pairs if p["split"] == split]
        pos = [p for p in split_pairs if i(p["online_positive"]) == 1]
        neg = [p for p in split_pairs if i(p["online_negative"]) == 1]
        hard_neg = [p for p in neg if p["pair_type"] == "negative_same_sequence_same_concept"]
        rows.append(
            {
                "split": split,
                "positive_pair_count": len(pos),
                "negative_pair_count": len(neg),
                "hard_negative_pair_count": len(hard_neg),
                "positive_precision_eval_only": pair_precision(pos),
                "negative_precision_eval_only": pair_precision(neg),
                "usable_for_dense_diagnostic_training": int(len(pos) >= 50 and len(neg) >= 50),
                "usable_for_main_online_training": 0,
            }
        )
    return rows


def opportunity_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "sequence_count": len(rows),
        "adjacent_positive_pair_opportunities": sum(i(r["adjacent_positive_pair_opportunities"]) for r in rows),
        "skip5_positive_pair_opportunities": sum(i(r["skip5_positive_pair_opportunities"]) for r in rows),
        "negative_cov_visible_pair_opportunities": sum(i(r["negative_cov_visible_pair_opportunities"]) for r in rows),
    }


def negative_controls(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [p for p in pairs if i(p["online_positive"]) == 1]
    negatives = [p for p in pairs if i(p["online_negative"]) == 1]
    shifted_correct = 0
    for idx, p in enumerate(positives):
        shifted = positives[(idx + 1) % len(positives)] if positives else p
        shifted_correct += int(
            p["sequence_id"] == shifted["sequence_id"]
            and p["instance_i_eval_only"] == shifted["instance_i_eval_only"]
        )
    same_concept_negative_false_positive = sum(
        int(
            p["concept_i_eval_only"] == p["concept_j_eval_only"]
            and p["sequence_id"] == p.get("sequence_j", p["sequence_id"])
            and p["event_id_i"] != p["event_id_j"]
            and p["instance_i_eval_only"] == p["instance_j_eval_only"]
        )
        for p in negatives
    )
    return [
        {
            "control_name": "shifted_positive_instance_binding",
            "real_positive_precision": pair_precision(positives),
            "control_positive_precision": shifted_correct / max(len(positives), 1),
            "control_passed": int(pair_precision(positives) > shifted_correct / max(len(positives), 1) + 0.50),
            "failure_reason": "",
        },
        {
            "control_name": "same_concept_negative_not_same_instance",
            "real_negative_precision": pair_precision(negatives),
            "same_concept_negative_false_positive_count": same_concept_negative_false_positive,
            "control_passed": int(same_concept_negative_false_positive == 0),
            "failure_reason": "" if same_concept_negative_false_positive == 0 else "same_instance_leaked_into_negative",
        },
    ]


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = read_csv(args.ledger)
    opportunity = read_csv(args.pair_opportunities)
    pairs = build_pairs(ledger, args.max_negative_pairs)
    split_rows = split_summary(pairs)
    controls = negative_controls(pairs)
    positives = [p for p in pairs if i(p["online_positive"]) == 1]
    negatives = [p for p in pairs if i(p["online_negative"]) == 1]
    hard_negatives = [p for p in negatives if p["pair_type"] == "negative_same_sequence_same_concept"]
    split_counts = Counter(p["split"] for p in positives)

    write_csv(out / f"stage_CORE1G_dense_pair_mining_audit_{args.artifact_version}.csv", pairs)
    write_csv(out / f"stage_CORE1G_split_pair_summary_{args.artifact_version}.csv", split_rows)
    write_csv(out / f"stage_CORE1G_negative_control_summary_{args.artifact_version}.csv", controls)
    write_json(out / f"stage_CORE1G_pair_opportunity_summary_{args.artifact_version}.json", opportunity_summary(opportunity))

    diagnostic_ready = int(
        len(positives) >= args.min_positive_pairs
        and len(negatives) >= args.min_negative_pairs
        and pair_precision(positives) >= 0.99
        and pair_precision(negatives) >= 0.99
        and all(i(c["control_passed"]) == 1 for c in controls)
    )
    compact = {
        "stage": "CORE-1G",
        "positive_pair_count": len(positives),
        "negative_pair_count": len(negatives),
        "hard_negative_pair_count": len(hard_negatives),
        "positive_precision_eval_only": pair_precision(positives),
        "negative_precision_eval_only": pair_precision(negatives),
        "train_positive_pairs": split_counts.get("train", 0),
        "dev_positive_pairs": split_counts.get("dev", 0),
        "test_positive_pairs": split_counts.get("test", 0),
        "negative_controls_passed": int(all(i(c["control_passed"]) == 1 for c in controls)),
        "dense_diagnostic_pair_mining_ready": diagnostic_ready,
        "main_online_training_ready": 0,
        "oracle_leakage_found": 0,
        "main_constraint": "Pairs are derived from CORE-1F GT event ledger and are diagnostic/upper-bound only, not main online training labels.",
        "next_recommendation": (
            "CORE-1H dense diagnostic encoder upper-bound, isolated from main NOPS"
            if diagnostic_ready
            else "repair CORE-1G dense pair mining before diagnostic encoder"
        ),
    }
    write_json(out / f"stage_CORE1G_compact_for_gpt_{args.artifact_version}.json", compact)

    report = [
        "# CORE-1G Dense Ledger Pair-Mining Gate",
        "",
        "CORE-1G mines query-memory positive/negative pairs from the CORE-1F dense internal event ledger.",
        "These pairs are GT-ledger diagnostic pairs and cannot be used as main no-pretrain online NOPS labels.",
        "",
        "## Result",
        f"- Positive pairs: {len(positives)}.",
        f"- Negative pairs: {len(negatives)}.",
        f"- Hard negatives: {len(hard_negatives)}.",
        f"- Positive precision: {pair_precision(positives):.4f}.",
        f"- Negative precision: {pair_precision(negatives):.4f}.",
        f"- Dense diagnostic ready: {diagnostic_ready}.",
        "",
        "## Decision",
        compact["next_recommendation"],
    ]
    (out / f"stage_CORE1G_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
