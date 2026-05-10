from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1_online_object_encoder import cosine_np, write_csv, write_json


FRAME_NORM = 960.0
BOX_NORM = 320.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORE-1H dense diagnostic encoder upper-bound.")
    p.add_argument("--ledger", default="results/core1f/stage_CORE1F_dense_event_ledger_v1.csv")
    p.add_argument("--pairs", default="results/core1g/stage_CORE1G_dense_pair_mining_audit_v1.csv")
    p.add_argument("--output-dir", default="results/core1h")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--max-train-pairs", type=int, default=6000)
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


def parse_box(s: str) -> tuple[float, float, float, float]:
    vals = [f(v) for v in str(s).split("|")]
    if len(vals) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    x1, y1, x2, y2 = vals
    return x1, y1, x2, y2


def box_features(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    area = w * h
    aspect = w / max(h, 1.0)
    return np.asarray(
        [
            cx / BOX_NORM,
            cy / BOX_NORM,
            w / BOX_NORM,
            h / BOX_NORM,
            area / (BOX_NORM * BOX_NORM),
            min(aspect / 8.0, 1.0),
        ],
        dtype=np.float32,
    )


def query_features(row: dict[str, str]) -> np.ndarray:
    return np.concatenate(
        [
            box_features(parse_box(row["box_after"])),
            np.asarray(
                [
                    f(row["reappear_frame"]) / FRAME_NORM,
                    f(row["gap_length"]) / 128.0,
                    f(row["same_concept_distractors_at_reappear"]) / 4.0,
                    f(row["num_visible_objects_at_reappear"]) / 8.0,
                ],
                dtype=np.float32,
            ),
        ]
    )


def memory_features(row: dict[str, str]) -> np.ndarray:
    return np.concatenate(
        [
            box_features(parse_box(row["box_before"])),
            np.asarray(
                [
                    f(row["disappear_frame"]) / FRAME_NORM,
                    0.0,
                    f(row["same_concept_distractors_at_reappear"]) / 4.0,
                    f(row["num_visible_objects_at_reappear"]) / 8.0,
                ],
                dtype=np.float32,
            ),
        ]
    )


def try_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except Exception:
        return None, None, None


def make_tower(nn: Any, F: Any, in_dim: int, out_dim: int):
    class Tower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 48),
                nn.ReLU(),
                nn.Linear(48, 32),
                nn.ReLU(),
                nn.Linear(32, out_dim),
            )

        def forward(self, x):
            return F.normalize(self.net(x), dim=1)

    return Tower


def build_arrays(
    pair_rows: list[dict[str, str]],
    ledger_by_event: dict[str, dict[str, str]],
    *,
    split: str,
    max_pairs: int,
    seed: int,
    shuffled_positive: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    rows = [r for r in pair_rows if r["split"] == split]
    pos = [r for r in rows if i(r["online_positive"]) == 1]
    neg = [r for r in rows if i(r["online_negative"]) == 1]
    rng.shuffle(neg)
    neg = neg[: max_pairs - len(pos)] if len(pos) < max_pairs else []
    rows = pos + neg
    rng.shuffle(rows)
    if shuffled_positive:
        positive_memory_ids = [r["event_id_j"] for r in rows if i(r["online_positive"]) == 1]
        rng.shuffle(positive_memory_ids)
        pidx = 0
    q_arr, m_arr, y_arr = [], [], []
    for r in rows[:max_pairs]:
        q_event = ledger_by_event.get(r["event_id_i"])
        m_event_id = r["event_id_j"]
        if shuffled_positive and i(r["online_positive"]) == 1 and positive_memory_ids:
            m_event_id = positive_memory_ids[pidx % len(positive_memory_ids)]
            pidx += 1
        m_event = ledger_by_event.get(m_event_id)
        if q_event is None or m_event is None:
            continue
        q_arr.append(query_features(q_event))
        m_arr.append(memory_features(m_event))
        y_arr.append(float(i(r["online_positive"])))
    return np.asarray(q_arr, dtype=np.float32), np.asarray(m_arr, dtype=np.float32), np.asarray(y_arr, dtype=np.float32)


def train_model(args: argparse.Namespace, pair_rows: list[dict[str, str]], ledger_by_event: dict[str, dict[str, str]], *, shuffled_positive: bool = False):
    torch, nn, F = try_torch()
    if torch is None:
        return None, None, [], {"available": 0}
    torch.manual_seed(args.seed)
    q_np, m_np, y_np = build_arrays(pair_rows, ledger_by_event, split="train", max_pairs=args.max_train_pairs, seed=args.seed, shuffled_positive=shuffled_positive)
    Tower = make_tower(nn, F, q_np.shape[1], args.embedding_dim)
    q_tower, m_tower = Tower(), Tower()
    opt = torch.optim.Adam(list(q_tower.parameters()) + list(m_tower.parameters()), lr=2e-3)
    q = torch.tensor(q_np)
    m = torch.tensor(m_np)
    y = torch.tensor(y_np)
    trace = []
    n = q.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            qe = q_tower(q[idx])
            me = m_tower(m[idx])
            logits = torch.sum(qe * me, dim=1) * 8.0
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            # Mild variance regularization to reduce collapse.
            loss = loss + 0.01 * (1.0 / (torch.var(qe, dim=0).mean() + 1e-4))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch in {0, args.epochs - 1} or epoch % 5 == 0:
            with torch.no_grad():
                qe = q_tower(q)
                collapse = float(torch.var(qe, dim=0).mean().detach().cpu())
            trace.append({"epoch": epoch, "loss": float(np.mean(losses)), "query_embedding_variance": collapse, "shuffled_positive": int(shuffled_positive)})
    return q_tower, m_tower, trace, {"available": 1, "train_pair_count": int(n)}


def embed_np(tower: Any, x: np.ndarray) -> np.ndarray:
    torch, _, _ = try_torch()
    if tower is None or torch is None:
        return x / max(float(np.linalg.norm(x)), 1e-8)
    with torch.no_grad():
        y = tower(torch.tensor(x.reshape(1, -1), dtype=torch.float32)).detach().cpu().numpy()[0]
    return y.astype(np.float32)


def evaluate(
    name: str,
    ledger_rows: list[dict[str, str]],
    q_tower: Any,
    m_tower: Any,
    *,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [r for r in ledger_rows if i(r["usable_real_gap"]) == 1]
    memory_by_split = {
        split: [r for r in rows if r["split"] == split]
        for split in ("train", "dev", "test")
    }
    out = []
    for r in rows:
        split = r["split"]
        qv = query_features(r)
        qe = qv if mode == "raw_geometry" else embed_np(q_tower, qv)
        scored = []
        for cand in memory_by_split[split]:
            mv = memory_features(cand)
            me = mv if mode == "raw_geometry" else embed_np(m_tower, mv)
            scored.append((cand["event_id"], cosine_np(qe, me), cand))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [s[0] for s in scored[:5]]
        top1 = top_ids[0] if top_ids else ""
        hit1 = int(top1 == r["event_id"])
        hit3 = int(r["event_id"] in set(top_ids[:3]))
        hit5 = int(r["event_id"] in set(top_ids[:5]))
        rank = next((idx for idx, eid in enumerate([s[0] for s in scored], 1) if eid == r["event_id"]), "")
        out.append(
            {
                "variant": name,
                "split": split,
                "event_id": r["event_id"],
                "target_event_id_eval_only": r["event_id"],
                "top1_event_id": top1,
                "target_retrieved_top1": hit1,
                "target_retrieved_top3": hit3,
                "target_retrieved_top5": hit5,
                "target_rank": rank,
                "false_retrieval": int(not hit1),
            }
        )
    summary = {"variant": name}
    for split in ("train", "dev", "test", "all"):
        ss = out if split == "all" else [r for r in out if r["split"] == split]
        n = max(len(ss), 1)
        summary[f"{split}_top1"] = sum(i(r["target_retrieved_top1"]) for r in ss) / n
        summary[f"{split}_top3"] = sum(i(r["target_retrieved_top3"]) for r in ss) / n
        summary[f"{split}_top5"] = sum(i(r["target_retrieved_top5"]) for r in ss) / n
        summary[f"{split}_false_retrieval_rate"] = sum(i(r["false_retrieval"]) for r in ss) / n
    return summary, out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_rows = read_csv(args.ledger)
    pair_rows = read_csv(args.pairs)
    ledger_by_event = {r["event_id"]: r for r in ledger_rows}

    q_tower, m_tower, trace, meta = train_model(args, pair_rows, ledger_by_event, shuffled_positive=False)
    sq_tower, sm_tower, shuf_trace, _ = train_model(args, pair_rows, ledger_by_event, shuffled_positive=True)
    for row in shuf_trace:
        row["control"] = "shuffled_positive"
    for row in trace:
        row["control"] = "real_pairs"
    write_csv(out / f"stage_CORE1H_training_trace_{args.artifact_version}.csv", trace + shuf_trace)

    summaries = []
    results = []
    for name, qt, mt, mode in [
        ("A0_raw_geometry_same_space", None, None, "raw_geometry"),
        ("A1_dense_diagnostic_encoder", q_tower, m_tower, "encoder"),
        ("A2_shuffled_positive_encoder_control", sq_tower, sm_tower, "encoder"),
    ]:
        s, r = evaluate(name, ledger_rows, qt, mt, mode=mode)
        summaries.append(s)
        results.extend(r)
    real = next(s for s in summaries if s["variant"] == "A1_dense_diagnostic_encoder")
    raw = next(s for s in summaries if s["variant"] == "A0_raw_geometry_same_space")
    shuf = next(s for s in summaries if s["variant"] == "A2_shuffled_positive_encoder_control")
    for s in summaries:
        s["selected_as_best"] = int(s["variant"] == max(summaries, key=lambda x: x["test_top1"])["variant"])
    write_csv(out / f"stage_CORE1H_ablation_summary_{args.artifact_version}.csv", summaries)
    write_csv(out / f"stage_CORE1H_retrieval_results_{args.artifact_version}.csv", results)

    controls_passed = int(real["test_top1"] > shuf["test_top1"] + 0.05)
    improves_raw = int(real["test_top1"] > raw["test_top1"])
    compact = {
        "stage": "CORE-1H",
        "torch_available": int(meta.get("available", 0)),
        "train_pair_count": int(meta.get("train_pair_count", 0)),
        "raw_geometry_test_top1": raw["test_top1"],
        "diagnostic_encoder_test_top1": real["test_top1"],
        "shuffled_positive_test_top1": shuf["test_top1"],
        "diagnostic_encoder_delta_vs_raw": real["test_top1"] - raw["test_top1"],
        "diagnostic_encoder_delta_vs_shuffled": real["test_top1"] - shuf["test_top1"],
        "controls_passed": controls_passed,
        "improves_raw_geometry": improves_raw,
        "passed_minimum": int(controls_passed and improves_raw),
        "safe_for_main_nops": 0,
        "oracle_leakage_found": 0,
        "main_constraint": "Encoder is trained from GT-ledger diagnostic pairs; this is an upper-bound experiment only.",
        "next_recommendation": (
            "CORE-1I convert diagnostic signal into tracker-derived online pairs"
            if controls_passed and improves_raw
            else "CORE-1I inspect dense diagnostic features; do not train main encoder from GT-ledger pairs"
        ),
    }
    write_json(out / f"stage_CORE1H_compact_for_gpt_{args.artifact_version}.json", compact)
    report = [
        "# CORE-1H Dense Diagnostic Encoder Upper-Bound",
        "",
        "CORE-1H trains a small random-init encoder from CORE-1G GT-ledger diagnostic pairs.",
        "This is an upper-bound experiment and is not safe for main NOPS integration.",
        "",
        "## Result",
        f"- Raw geometry test top1: {raw['test_top1']:.4f}.",
        f"- Diagnostic encoder test top1: {real['test_top1']:.4f}.",
        f"- Shuffled-positive control test top1: {shuf['test_top1']:.4f}.",
        f"- Passed minimum: {compact['passed_minimum']}.",
        "",
        "## Decision",
        compact["next_recommendation"],
    ]
    (out / f"stage_CORE1H_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
