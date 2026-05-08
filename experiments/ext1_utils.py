from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lagot_adapter import LaGOTAdapter
from datasets.external.lasot_adapter import LaSOTAdapter
from datasets.external.lvos_adapter import LVOSAdapter
from datasets.external.tao_adapter import TAOAdapter
from datasets.external.base_video_memory_dataset import FrameSampleExternal


Box = tuple[float, float, float, float]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def configured_adapters() -> list[tuple[str, str, Any]]:
    return [
        ("lvos_hf_sample", "data/external/hf_lvosv1_sample", LVOSAdapter),
        ("lagot_annotations", "data/external/lagot_annotations", LaGOTAdapter),
        ("lvos", "data/external/lvos", LVOSAdapter),
        ("lasot", "data/external/lasot", LaSOTAdapter),
        ("tao", "data/external/tao", TAOAdapter),
    ]


def make_adapter(dataset_name: str, root: str | None = None):
    for name, default_root, cls in configured_adapters():
        if name == dataset_name:
            return cls(root or default_root)
    raise ValueError(dataset_name)


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def box_iou(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = box_area((x1, y1, x2, y2))
    union = box_area(a) + box_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def center_distance(a: Box, b: Box) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.hypot(ax - bx, ay - by)


def size_similarity(a: Box, b: Box) -> float:
    aw, ah = max(1e-6, a[2] - a[0]), max(1e-6, a[3] - a[1])
    bw, bh = max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])
    return math.exp(-abs(math.log(aw / bw)) - abs(math.log(ah / bh)))


def frame_size(frame: FrameSampleExternal | None, boxes: Iterable[Box]) -> tuple[float, float]:
    if frame is not None:
        width = frame.metadata.get("width")
        height = frame.metadata.get("height")
        if width and height:
            return float(width), float(height)
    max_x = max((b[2] for b in boxes), default=1.0)
    max_y = max((b[3] for b in boxes), default=1.0)
    return max(1.0, max_x), max(1.0, max_y)


def normalize_distance(dist: float, width: float, height: float) -> float:
    return dist / max(1.0, math.hypot(width, height))


def box_descriptor(box: Box, width: float, height: float) -> list[float]:
    x1, y1, x2, y2 = box
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx, cy = box_center(box)
    return [
        cx / max(width, 1.0),
        cy / max(height, 1.0),
        w / max(width, 1.0),
        h / max(height, 1.0),
        (w * h) / max(width * height, 1.0),
        w / h,
    ]


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def collect_frames(adapter: Any, sequence_id: str) -> list[FrameSampleExternal]:
    return list(adapter.iter_frames(sequence_id))


def frames_by_index(frames: list[FrameSampleExternal]) -> dict[int, FrameSampleExternal]:
    return {int(fr.frame_idx): fr for fr in frames}


def object_history(frames: list[FrameSampleExternal]) -> dict[str, list[tuple[int, Box, str | int | None]]]:
    hist: dict[str, list[tuple[int, Box, str | int | None]]] = defaultdict(list)
    for fr in frames:
        for idx, iid in enumerate(fr.instance_ids):
            cat = fr.category_ids[idx] if idx < len(fr.category_ids) else None
            hist[str(iid)].append((int(fr.frame_idx), fr.boxes[idx], cat))
    for states in hist.values():
        states.sort(key=lambda x: x[0])
    return hist


def state_at(history: dict[str, list[tuple[int, Box, Any]]], iid: str, frame_idx: int) -> tuple[int, Box, Any] | None:
    for st in history.get(str(iid), []):
        if st[0] == frame_idx:
            return st
    return None


def last_before(history: dict[str, list[tuple[int, Box, Any]]], iid: str, frame_idx: int) -> tuple[int, Box, Any] | None:
    prior = [st for st in history.get(str(iid), []) if st[0] < frame_idx]
    return prior[-1] if prior else None


def last_k_before(history: dict[str, list[tuple[int, Box, Any]]], iid: str, frame_idx: int, k: int = 8) -> list[tuple[int, Box, Any]]:
    prior = [st for st in history.get(str(iid), []) if st[0] < frame_idx]
    return prior[-k:]


def trajectory_prediction(states: list[tuple[int, Box, Any]]) -> Box | None:
    if not states:
        return None
    if len(states) == 1:
        return states[-1][1]
    f1, b1, _ = states[-2]
    f2, b2, _ = states[-1]
    dt = max(1, f2 - f1)
    step = max(1, f2 - f1)
    c1 = box_center(b1)
    c2 = box_center(b2)
    vx = (c2[0] - c1[0]) / dt
    vy = (c2[1] - c1[1]) / dt
    w = b2[2] - b2[0]
    h = b2[3] - b2[1]
    cx = c2[0] + vx * step
    cy = c2[1] + vy * step
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


def dataset_inventory(max_sequences: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    adapters: dict[str, Any] = {}
    for name, root, cls in configured_adapters():
        root_path = Path(root)
        path_exists = int(root_path.exists())
        row: dict[str, Any] = {
            "dataset_name": name,
            "root_path": root,
            "adapter_name": cls.__name__,
            "path_exists": path_exists,
            "num_sequences": 0,
            "num_frames_sampled": 0,
            "num_objects": 0,
            "has_raw_frames": 0,
            "has_boxes": 0,
            "has_masks": 0,
            "has_instance_ids": 0,
            "has_visibility": 0,
            "has_reentry_events": 0,
            "has_occlusion_events": 0,
            "usable_for_memory_eval": 0,
            "not_usable_reason": "path_missing" if not path_exists else "",
        }
        if path_exists:
            try:
                adapter = cls(root_path)
                adapters[name] = adapter
                seqs = list(adapter.iter_sequences())
                sample_seqs = seqs[:max_sequences] if max_sequences else seqs
                events = []
                object_ids: set[str] = set()
                raw_frames = boxes = masks = ids = visibility = 0
                sampled_frames = 0
                for seq in sample_seqs[:10]:
                    frames = list(adapter.iter_frames(seq, limit=20))
                    sampled_frames += len(frames)
                    for fr in frames:
                        raw_frames = raw_frames or int(fr.frame_path is not None)
                        boxes = boxes or int(bool(fr.boxes))
                        masks = masks or int(bool(fr.masks))
                        ids = ids or int(bool(fr.instance_ids))
                        visibility = visibility or int(bool(fr.visibility))
                        object_ids.update(str(i) for i in fr.instance_ids)
                for seq in sample_seqs:
                    events.extend(adapter.derive_events(seq))
                usable_events = [ev for ev in events if ev.gap_length >= 3]
                row.update({
                    "num_sequences": len(seqs),
                    "num_frames_sampled": sampled_frames,
                    "num_objects": len(object_ids),
                    "has_raw_frames": raw_frames,
                    "has_boxes": boxes,
                    "has_masks": masks,
                    "has_instance_ids": ids,
                    "has_visibility": visibility,
                    "has_reentry_events": int(bool(usable_events)),
                    "has_occlusion_events": int(bool(usable_events)),
                    "usable_for_memory_eval": int(bool(usable_events) and boxes and ids),
                    "not_usable_reason": "" if bool(usable_events) and boxes and ids else "no_reentry_or_occlusion_events",
                })
            except Exception as exc:
                row["not_usable_reason"] = f"adapter_error:{exc}"
        rows.append(row)
    return rows, adapters


def build_external_event_ledger(adapters: dict[str, Any], max_sequences: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name, adapter in adapters.items():
        try:
            sequences = list(adapter.iter_sequences())
        except Exception:
            continue
        selected = sequences[:max_sequences] if max_sequences else sequences
        for seq in selected:
            try:
                frames = collect_frames(adapter, seq)
                hist = object_history(frames)
                frame_lookup = frames_by_index(frames)
                events = adapter.derive_events(seq)
            except Exception:
                continue
            for idx, ev in enumerate(events):
                before = state_at(hist, ev.instance_id, ev.disappear_frame)
                after = state_at(hist, ev.instance_id, ev.reappear_frame)
                after_frame = frame_lookup.get(ev.reappear_frame)
                target_cat = after[2] if after else ev.metadata.get("category_id", "")
                same_cat = 0
                total_distractors = 0
                if after_frame is not None:
                    for obj_idx, iid in enumerate(after_frame.instance_ids):
                        if str(iid) == str(ev.instance_id):
                            continue
                        total_distractors += 1
                        cat = after_frame.category_ids[obj_idx] if obj_idx < len(after_frame.category_ids) else target_cat
                        same_cat += int(cat == target_cat)
                usable = int(ev.gap_length >= 3 and before is not None and after is not None)
                reason = "" if usable else ("gap_too_short" if ev.gap_length < 3 else "missing_before_or_after_box")
                rows.append({
                    "dataset_name": dataset_name,
                    "sequence_id": ev.sequence_id,
                    "event_id": f"{dataset_name}:{ev.sequence_id}:{idx}:{ev.instance_id}",
                    "instance_id": ev.instance_id,
                    "category_id": target_cat,
                    "disappear_frame": ev.disappear_frame,
                    "reappear_frame": ev.reappear_frame,
                    "gap_length": ev.gap_length,
                    "absence_type": ev.event_type,
                    "visible_before": int(before is not None),
                    "visible_after": int(after is not None),
                    "num_similar_distractors": same_cat,
                    "num_same_category_objects": same_cat + 1,
                    "has_box_before": int(before is not None),
                    "has_box_after": int(after is not None),
                    "has_mask_before": 0,
                    "has_mask_after": 0,
                    "event_usable": usable,
                    "not_usable_reason": reason,
                })
    return rows


def difficulty_rows(adapters: dict[str, Any], ledger_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    frame_cache: dict[tuple[str, str], tuple[dict[str, list[tuple[int, Box, Any]]], dict[int, FrameSampleExternal]]] = {}
    for row in ledger_rows:
        if str(row.get("event_usable")) != "1":
            continue
        key = (row["dataset_name"], row["sequence_id"])
        if key not in frame_cache:
            frames = collect_frames(adapters[row["dataset_name"]], row["sequence_id"])
            frame_cache[key] = (object_history(frames), frames_by_index(frames))
        hist, _ = frame_cache[key]
        before = state_at(hist, row["instance_id"], int(row["disappear_frame"]))
        after = state_at(hist, row["instance_id"], int(row["reappear_frame"]))
        if before is None or after is None:
            continue
        b0, b1 = before[1], after[1]
        area0, area1 = box_area(b0), box_area(b1)
        center_disp = center_distance(b0, b1)
        scale_change = area1 / max(area0, 1.0)
        gap = int(row["gap_length"])
        distractors = int(row.get("num_similar_distractors", 0))
        hard_score = int(gap >= 30) + int(center_disp > 100) + int(abs(math.log(max(scale_change, 1e-6))) > 0.5) + int(distractors >= 2)
        difficulty = "hard" if hard_score >= 3 else ("medium" if hard_score >= 1 else "easy")
        out.append({
            "event_id": row["event_id"],
            "gap_length": gap,
            "object_area_before": area0,
            "object_area_after": area1,
            "area_change_ratio": scale_change,
            "center_displacement": center_disp,
            "scale_change_ratio": scale_change,
            "visibility_before": 1,
            "visibility_after": 1,
            "num_distractors": int(row.get("num_similar_distractors", 0)),
            "same_category_distractors": distractors,
            "motion_complexity_proxy": center_disp / max(gap, 1),
            "difficulty_level": difficulty,
        })
    return out


def score_candidates(method: str, query_box: Box, candidates: dict[str, list[tuple[int, Box, Any]]], width: float, height: float, reappear_frame: int) -> list[tuple[str, float]]:
    q_desc = box_descriptor(query_box, width, height)
    scored: list[tuple[str, float]] = []
    for iid, states_all in candidates.items():
        states = [st for st in states_all if st[0] < reappear_frame]
        if not states:
            continue
        last_box = states[-1][1]
        dist = normalize_distance(center_distance(query_box, last_box), width, height)
        iou = box_iou(query_box, last_box)
        size = size_similarity(query_box, last_box)
        if method == "B0_tracker_iou_centroid_memory":
            score = 1.2 * iou + 0.5 * size - dist
        elif method == "B1_template_descriptor_nn":
            score = -l2(q_desc, box_descriptor(last_box, width, height))
        elif method == "B2_support_trajectory_memory":
            pred = trajectory_prediction(states[-8:]) or last_box
            pred_dist = normalize_distance(center_distance(query_box, pred), width, height)
            score = 0.8 * box_iou(query_box, pred) + 0.4 * size_similarity(query_box, pred) - pred_dist
        elif method == "B3_nops_anchor_episodic_passive":
            pred = trajectory_prediction(states[-8:]) or last_box
            pred_dist = normalize_distance(center_distance(query_box, pred), width, height)
            shape_score = -l2(q_desc[2:], box_descriptor(last_box, width, height)[2:])
            trajectory_score = 0.8 * box_iou(query_box, pred) - pred_dist
            anchor_recency = -min(1.0, max(0, reappear_frame - states[-1][0]) / 200.0)
            score = 0.45 * trajectory_score + 0.35 * shape_score + 0.20 * anchor_recency
        else:
            raise ValueError(method)
        scored.append((iid, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


METHODS = [
    "B0_tracker_iou_centroid_memory",
    "B1_template_descriptor_nn",
    "B2_support_trajectory_memory",
    "B3_nops_anchor_episodic_passive",
]


def oracle_memory_results(adapters: dict[str, Any], ledger_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    frame_cache: dict[tuple[str, str], tuple[list[FrameSampleExternal], dict[str, list[tuple[int, Box, Any]]], dict[int, FrameSampleExternal]]] = {}
    usable_rows = [r for r in ledger_rows if str(r.get("event_usable")) == "1"]
    for row in usable_rows:
        key = (row["dataset_name"], row["sequence_id"])
        if key not in frame_cache:
            frames = collect_frames(adapters[row["dataset_name"]], row["sequence_id"])
            frame_cache[key] = (frames, object_history(frames), frames_by_index(frames))
        frames, hist, frame_lookup = frame_cache[key]
        target_id = row["instance_id"]
        reappear = int(row["reappear_frame"])
        after = state_at(hist, target_id, reappear)
        frame = frame_lookup.get(reappear)
        width, height = frame_size(frame, [b for states in hist.values() for _, b, _ in states[:1]])
        if after is None:
            continue
        query_box = after[1]
        candidates = {iid: states for iid, states in hist.items() if any(st[0] < reappear for st in states)}
        for method in METHODS:
            ranked = score_candidates(method, query_box, candidates, width, height, reappear)
            ranked_ids = [iid for iid, _ in ranked]
            in_top1 = int(bool(ranked_ids) and ranked_ids[0] == target_id)
            in_top3 = int(target_id in ranked_ids[:3])
            in_top5 = int(target_id in ranked_ids[:5])
            target_in_memory = int(target_id in candidates)
            if not target_in_memory:
                failure = "target_not_in_memory"
            elif not in_top5:
                failure = "target_not_in_top5"
            elif not in_top1 and int(row.get("num_similar_distractors", 0)) > 0:
                failure = "similar_distractor_confusion"
            elif not in_top1:
                failure = "target_in_top5_but_wrong_top1"
            else:
                failure = ""
            out.append({
                "dataset_name": row["dataset_name"],
                "sequence_id": row["sequence_id"],
                "event_id": row["event_id"],
                "method_name": method,
                "proposal_mode": "oracle_gt_box_memory_only",
                "target_instance_id_eval_only": target_id,
                "predicted_memory_id": ranked_ids[0] if ranked_ids else "",
                "target_memory_retrieved_top1": in_top1,
                "target_memory_retrieved_top3": in_top3,
                "target_memory_retrieved_top5": in_top5,
                "false_retrieval": int(target_in_memory and not in_top1),
                "new_object_false_birth": int(not target_in_memory),
                "memory_size": len(candidates),
                "failure_reason": failure,
            })
    return out


def baseline_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_method[(r["dataset_name"], r["method_name"])].append(r)
    rows: list[dict[str, Any]] = []
    for (dataset_name, method), rs in sorted(by_method.items()):
        n = len(rs)
        rows.append({
            "dataset_name": dataset_name,
            "method_name": method,
            "num_events": n,
            "top1": sum(int(r["target_memory_retrieved_top1"]) for r in rs) / max(n, 1),
            "top3": sum(int(r["target_memory_retrieved_top3"]) for r in rs) / max(n, 1),
            "top5": sum(int(r["target_memory_retrieved_top5"]) for r in rs) / max(n, 1),
            "false_retrieval_rate": sum(int(r["false_retrieval"]) for r in rs) / max(n, 1),
            "new_object_false_birth_rate": sum(int(r["new_object_false_birth"]) for r in rs) / max(n, 1),
            "memory_growth": max((int(r["memory_size"]) for r in rs), default=0),
            "mean_memory_size": sum(int(r["memory_size"]) for r in rs) / max(n, 1),
            "valid_event_count": n,
            "invalid_event_count": 0,
            "notes": "oracle proposal memory-only; geometry-only when raw frames unavailable",
        })
    return rows


def failure_taxonomy(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        if r.get("failure_reason"):
            reason = r["failure_reason"]
        else:
            reason = ""
        rows.append({
            "dataset_name": r["dataset_name"],
            "event_id": r["event_id"],
            "method_name": r["method_name"],
            "failure_reason": reason,
        })
    return rows


def metric_consistency(summary: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_key[(r["dataset_name"], r["method_name"])].append(r)
    for s in summary:
        rs = by_key[(s["dataset_name"], s["method_name"])]
        for metric, field in (
            ("top1", "target_memory_retrieved_top1"),
            ("top3", "target_memory_retrieved_top3"),
            ("top5", "target_memory_retrieved_top5"),
            ("false_retrieval_rate", "false_retrieval"),
        ):
            recomputed = sum(int(r[field]) for r in rs) / max(len(rs), 1)
            reported = float(s[metric])
            rows.append({
                "metric_name": metric,
                "method_name": s["method_name"],
                "reported_value": reported,
                "recomputed_value": recomputed,
                "matched": int(abs(reported - recomputed) < 1e-12),
                "difference_reason": "",
            })
    return rows

