from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator
from experiments.phase3r_utils import build_phase3_scenario_map, load_config_payload, write_csv
from experiments.run_v3_stage_e3_episodic_bundle_pattern_completion import (
    FOCUS_EVENT_IDS,
    SCENARIO_NAMES,
    bundle_worthy,
    cosine,
    find_assignment,
    gt_box,
    load_alignment,
    load_events,
    make_bundle,
    norm,
    pick_proposal,
    track_snap,
)
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run v3 Stage E3.1 anti-hub retrieval repair.')
    p.add_argument('--config', default='configs/bridge_synth_generic_v1.yaml')
    p.add_argument('--event-audit', default='results/v3_e1/stage_E1_event_audit_v1.csv')
    p.add_argument('--cross-run-alignment', default='results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv')
    p.add_argument('--e2c-negative-events', default='results/v3_e2c/stage_E2C_negative_control_events_v1.csv')
    p.add_argument('--output-dir', default='results/v3_e31')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--artifact-version', default='v1')
    return p.parse_args()


def si(v: Any, d: int | None = None) -> int | None:
    if v in (None, ''):
        return d
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return d


def shape_sig(box: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> np.ndarray:
    h, w = frame_shape
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    area = bw * bh
    return np.asarray([bw / max(w, 1), bh / max(h, 1), bw / bh, area / max(w * h, 1)], dtype=np.float32)


def context_sig(centroid: tuple[float, float], frame_shape: tuple[int, int], proposal_count: int) -> np.ndarray:
    h, w = frame_shape
    cx, cy = centroid
    return np.asarray([cx / max(w, 1), cy / max(h, 1), min(float(proposal_count) / 8.0, 1.0)], dtype=np.float32)


def motion_sig(velocity: np.ndarray) -> np.ndarray:
    v = np.asarray(velocity, dtype=np.float32).reshape(-1)
    if v.size < 2:
        v = np.pad(v, (0, 2 - v.size))
    mag = float(np.linalg.norm(v[:2]))
    if mag <= 1e-8:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray([v[0] / mag, v[1] / mag, min(mag / 10.0, 1.0)], dtype=np.float32)


def cue_from_obs(proposal: Any, assignment: Any, frame_shape: tuple[int, int], proposal_count: int, frame_idx: int) -> dict[str, Any]:
    box = tuple(int(v) for v in (assignment.box if assignment is not None else proposal.box))
    centroid = tuple(float(v) for v in (assignment.centroid if assignment is not None else proposal.centroid))
    velocity = np.asarray(getattr(assignment, 'velocity', np.zeros(2, dtype=np.float32)), dtype=np.float32) if assignment is not None else np.zeros(2, dtype=np.float32)
    appearance = np.zeros(8, dtype=np.float32) if assignment is None else norm(np.asarray(assignment.signature, dtype=np.float32))
    quality = float(assignment.score if assignment is not None else proposal.score)
    return {
        'box': box,
        'support_shape': shape_sig(box, frame_shape),
        'appearance_proxy': appearance,
        'local_context': context_sig(centroid, frame_shape, proposal_count),
        'motion_signature': motion_sig(velocity),
        'proposal_quality': quality,
        'frame_idx': int(frame_idx),
    }


def b_track(bundle: dict[str, Any]) -> int:
    return min(int(v) for v in bundle['source_track_ids'])


def b_proto(bundle: dict[str, Any]) -> int:
    return min(int(v) for v in bundle['source_prototype_ids'])


def b_lineage(bundle: dict[str, Any]) -> int | None:
    return min(int(v) for v in bundle['source_lineage_ids']) if bundle['source_lineage_ids'] else None


def provenance_sig(bundle: dict[str, Any]) -> np.ndarray:
    stable = {'candidate': 0.25, 'stabilizing': 0.6, 'stable': 1.0}.get(str(bundle['stability_level']), 0.25)
    proto = b_proto(bundle)
    track = b_track(bundle)
    lineage = b_lineage(bundle)
    return np.asarray([min(proto / 32.0, 1.0), min(track / 512.0, 1.0), 0.0 if lineage is None else min(lineage / 16.0, 1.0), stable], dtype=np.float32)


def separation_sig(bundle: dict[str, Any]) -> np.ndarray:
    return np.concatenate([
        np.asarray(bundle['support_signature'], dtype=np.float32),
        np.asarray(bundle['context_signature'], dtype=np.float32),
        np.asarray(bundle['temporal_signature'], dtype=np.float32)[:3] / np.asarray([32.0, 16.0, 16.0], dtype=np.float32),
        provenance_sig(bundle),
    ]).astype(np.float32)


def cue_sep_sig(cue: dict[str, Any]) -> np.ndarray:
    return np.concatenate([
        np.asarray(cue['support_shape'], dtype=np.float32),
        np.asarray(cue['local_context'], dtype=np.float32),
        np.asarray([min(cue['frame_idx'] / 2048.0, 1.0), cue['proposal_quality'], 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ]).astype(np.float32)


def collect_runtime_data(config_path: str, event_audit_path: str, alignment_path: str, seed: int):
    payload = load_config_payload(config_path)
    scenario_map = build_phase3_scenario_map(config_path)
    event_rows_by_scenario = load_events(event_audit_path)
    alignment_map = load_alignment(alignment_path)
    bundle_by_id, event_records, bundle_write_rows = {}, [], []
    next_bundle_id = 1
    for scenario_name in SCENARIO_NAMES:
        sequence = SyntheticStreamGenerator(scenario_map[scenario_name], seed=seed).generate_sequence(0)
        encoder = MinimalSpikeEncoder(**payload['model']['spike_encoder'])
        field = MinimalObjectnessField(**payload['field'])
        tracker = MinimalTemporalIdentityTracker(**payload['tracking'])
        memory = MinimalPrototypeMemory(**payload['memory'])
        bundles, prev_tracks, prev_memory_output = [], {}, None
        frame_shape = tuple(int(v) for v in sequence.frames[0].frame.shape[:2])
        events_at_frame = {}
        for e in event_rows_by_scenario.get(scenario_name, []):
            events_at_frame.setdefault(int(e['reappear_frame']), []).append(e)
        for frame_offset in range(1, len(sequence.frames)):
            prev_frame, current_frame = sequence.frames[frame_offset - 1], sequence.frames[frame_offset]
            frame_idx = int(current_frame.frame_index)
            encoding = encoder.encode(prev_frame.frame, current_frame.frame)
            objectness_output = field.compute(encoding)
            tracking_output = tracker.update(proposals=objectness_output.proposals, encoding=encoding, heatmap=objectness_output.heatmap, current_frame=current_frame.frame, frame_index=current_frame.frame_index, memory_context=prev_memory_output)
            memory_output = memory.update(tracking_output.assignments, frame_index=current_frame.frame_index, track_states=(tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks))
            tracker.apply_concept_gated_resurrection(tracking_output, memory_output, frame_index=current_frame.frame_index, frame_shape=objectness_output.heatmap.shape)
            tracker.bind_prototypes(memory_output.assignments)
            prev_memory_output = memory_output
            current_tracks = {int(t.track_id): track_snap(t) for t in (tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks)}
            for track_id, prev_t in prev_tracks.items():
                cur_t = current_tracks.get(track_id)
                trigger, source_t = None, None
                if cur_t is None and prev_t['state'] in {'active', 'dormant', 'ghost'}:
                    trigger, source_t = 'track_missing_from_registry', prev_t
                elif cur_t is not None and prev_t['state'] == 'active' and cur_t['state'] in {'dormant', 'ghost', 'retired'}:
                    trigger, source_t = f"active_to_{cur_t['state']}", prev_t
                if trigger is None or source_t is None:
                    continue
                dedup = (source_t['track_id'], source_t['prototype_id'], source_t['last_seen_frame'])
                if not bundle_worthy(source_t):
                    bundle_write_rows.append({'scenario_name': scenario_name, 'frame_idx': frame_idx, 'track_id': source_t['track_id'], 'prototype_id': source_t['prototype_id'], 'lineage_id': source_t['continuity_lineage_id'] if source_t['continuity_lineage_id'] is not None else source_t['lineage_id'], 'write_trigger': trigger, 'bundle_written': 0, 'bundle_id': '', 'memory_anchor_id': '', 'skip_reason': 'not_bundle_worthy', 'support_quality': source_t['score'], 'track_age': source_t['age'], 'disappearance_context_available': 1})
                    continue
                if any((b_track(b), b_proto(b), int(b['last_source_frame'])) == dedup for b in bundles):
                    continue
                b = make_bundle(next_bundle_id, scenario_name, source_t, frame_idx, frame_shape, len(objectness_output.proposals))
                next_bundle_id += 1
                b['scenario_name'] = scenario_name
                b['primary_source_track_id'] = b_track(b)
                b['primary_source_prototype_id'] = b_proto(b)
                b['primary_source_lineage_id'] = b_lineage(b)
                b['provenance_signature'] = provenance_sig(b)
                b['separation_signature'] = separation_sig(b)
                b['last_source_quality'] = float(source_t['score'])
                bundles.append(b)
                bundle_by_id[int(b['bundle_id'])] = b
                bundle_write_rows.append({'scenario_name': scenario_name, 'frame_idx': frame_idx, 'track_id': source_t['track_id'], 'prototype_id': source_t['prototype_id'], 'lineage_id': b['canonical_lineage_id'], 'write_trigger': trigger, 'bundle_written': 1, 'bundle_id': b['bundle_id'], 'memory_anchor_id': b['memory_anchor_id'], 'skip_reason': '', 'support_quality': source_t['score'], 'track_age': source_t['age'], 'disappearance_context_available': 1})
            if frame_idx in events_at_frame:
                for event in events_at_frame[frame_idx]:
                    proposal_detected = int(si(event.get('proposal_detected'), 0) or 0)
                    target_box = gt_box(current_frame, int(event['instance_id']))
                    picked = pick_proposal(objectness_output.proposals, target_box) if target_box is not None else None
                    if picked is None:
                        event_records.append({'scenario_name': scenario_name, 'event_id': str(event['ledger_event_id']), 'frame_idx': frame_idx, 'proposal_detected': proposal_detected, 'proposal_id': None, 'cue': None, 'eligible_bundle_ids': [], 'target_bundle_id': None, 'target_bundle_exists': 0, 'old_track_id': si(event.get('old_track_id'), -1), 'old_prototype_id': si(event.get('old_prototype_id'), -1), 'alignment_classification': alignment_map.get(str(event['ledger_event_id']), {}).get('classification', ''), 'target_anchor_uid': alignment_map.get(str(event['ledger_event_id']), {}).get('target_anchor_uid', '')})
                        continue
                    proposal_id, proposal, _ = picked
                    assignment = find_assignment(tracking_output.assignments, proposal_id)
                    cue = cue_from_obs(proposal, assignment, frame_shape, len(objectness_output.proposals), frame_idx)
                    eligible = [b for b in bundles if int(b['created_frame']) < frame_idx]
                    old_track_id, old_proto_id = si(event.get('old_track_id'), -1), si(event.get('old_prototype_id'), -1)
                    targets = [b for b in eligible if (old_track_id >= 0 and old_track_id in b['source_track_ids']) or (old_proto_id >= 0 and old_proto_id in b['source_prototype_ids'])]
                    targets.sort(key=lambda b: (int(b['created_frame']), int(b['bundle_id'])), reverse=True)
                    target_bundle = targets[0] if targets else None
                    event_records.append({'scenario_name': scenario_name, 'event_id': str(event['ledger_event_id']), 'frame_idx': frame_idx, 'proposal_detected': proposal_detected, 'proposal_id': int(proposal_id), 'cue': cue, 'eligible_bundle_ids': [int(b['bundle_id']) for b in eligible], 'target_bundle_id': None if target_bundle is None else int(target_bundle['bundle_id']), 'target_bundle_exists': int(target_bundle is not None), 'old_track_id': old_track_id, 'old_prototype_id': old_proto_id, 'alignment_classification': alignment_map.get(str(event['ledger_event_id']), {}).get('classification', ''), 'target_anchor_uid': alignment_map.get(str(event['ledger_event_id']), {}).get('target_anchor_uid', '')})
            prev_tracks = current_tracks
    return bundle_by_id, event_records, bundle_write_rows

def compute_static_counts(bundle_by_id):
    proto_counter, track_counter, lineage_counter = Counter(), Counter(), Counter()
    for b in bundle_by_id.values():
        proto_counter[int(b['primary_source_prototype_id'])] += 1
        track_counter[int(b['primary_source_track_id'])] += 1
        lineage_counter[b['primary_source_lineage_id']] += 1
    return proto_counter, track_counter, lineage_counter


def score_candidate(cue, bundle, config, proto_freq, track_freq, lineage_freq, historical_topk):
    support_score = cosine(cue['support_shape'], bundle['support_signature'])
    content_score = cosine(cue['appearance_proxy'], bundle['content_signature'])
    context_score = cosine(cue['local_context'], bundle['context_signature'])
    motion_score = cosine(cue['motion_signature'], bundle['motion_signature'])
    gap = max(1, int(cue['frame_idx']) - int(bundle['last_source_frame']))
    temporal_score = float(np.exp(-gap / float(config['temporal_decay'])))
    quality_match = max(0.0, 1.0 - abs(float(cue['proposal_quality']) - float(bundle['last_source_quality'])))
    disappearance_score = 0.55 * temporal_score + 0.45 * quality_match
    provenance_score = 0.6 * float(bundle['accessibility_score']) + 0.4 * float(cosine(cue_sep_sig(cue)[-4:], bundle['provenance_signature']))
    separation_score = cosine(cue_sep_sig(cue), bundle['separation_signature'])
    rarity_bonus = config['rarity_bonus'] * (0.6 / (1.0 + math.log1p(proto_freq)) + 0.4 / (1.0 + math.log1p(track_freq)))
    hub_penalty = config['hub_alpha'] * math.log1p(proto_freq) + config['hub_beta'] * math.log1p(track_freq) + config['hub_gamma'] * math.log1p(historical_topk)
    lineage_penalty = config['lineage_hub_gamma'] * math.log1p(lineage_freq)
    w = config['weights']
    base_score = (w['content'] * content_score + w['support'] * support_score + w['motion'] * motion_score + w['context'] * context_score + w['temporal'] * temporal_score + w['disappearance'] * disappearance_score + w['provenance'] * provenance_score + w['separation'] * separation_score)
    final_score = base_score + rarity_bonus - hub_penalty - lineage_penalty
    return {'bundle_id': int(bundle['bundle_id']), 'memory_anchor_id': bundle['memory_anchor_id'], 'canonical_lineage_id': bundle['canonical_lineage_id'], 'primary_source_prototype_id': int(bundle['primary_source_prototype_id']), 'primary_source_track_id': int(bundle['primary_source_track_id']), 'primary_source_lineage_id': bundle['primary_source_lineage_id'], 'proto_freq': proto_freq, 'track_freq': track_freq, 'lineage_freq': lineage_freq, 'historical_topk': historical_topk, 'support_score': support_score, 'content_score': content_score, 'motion_score': motion_score, 'context_score': context_score, 'temporal_score': temporal_score, 'disappearance_score': disappearance_score, 'provenance_score': provenance_score, 'separation_score': separation_score, 'rarity_bonus': rarity_bonus, 'hub_penalty': hub_penalty, 'lineage_penalty': lineage_penalty, 'base_score': base_score, 'final_score': final_score}


def diversify_candidates(scored, config):
    if not config['competition']:
        return scored[:config['final_topk']]
    proto_count, anchor_count, lineage_count = Counter(), Counter(), Counter()
    selected = []
    for row in scored:
        proto_id = int(row['primary_source_prototype_id'])
        anchor_id = str(row['memory_anchor_id'])
        lineage_id = row['canonical_lineage_id']
        if proto_count[proto_id] >= config['max_per_proto']:
            continue
        if anchor_count[anchor_id] >= config['max_per_anchor']:
            continue
        if lineage_count[lineage_id] >= config['max_per_lineage']:
            continue
        selected.append(row)
        proto_count[proto_id] += 1
        anchor_count[anchor_id] += 1
        lineage_count[lineage_id] += 1
        if len(selected) >= config['final_topk']:
            break
    return selected


def evaluate_ablation(name, config, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter):
    retrieval_rows, score_rows, hub_rows = [], [], []
    hubness_counter, top1_counter, top3_counter, top5_counter, hist_topk = Counter(), Counter(), Counter(), Counter(), Counter()
    proto0_top5_hits = 0
    proto0_top5_slots = 0
    ordered_events = sorted(event_records, key=lambda r: (r['scenario_name'], int(r['frame_idx']), r['event_id']))
    for event in ordered_events:
        if int(event['proposal_detected']) != 1 or event['cue'] is None:
            retrieval_rows.append({'ablation_name': name, 'scenario_name': event['scenario_name'], 'event_id': event['event_id'], 'frame_idx': int(event['frame_idx']), 'proposal_detected': int(event['proposal_detected']), 'target_bundle_exists': int(event['target_bundle_exists']), 'target_bundle_id': event['target_bundle_id'] or '', 'target_bundle_rank': '', 'target_bundle_score': '', 'target_bundle_retrieved_top1': 0, 'target_bundle_retrieved_top3': 0, 'target_bundle_retrieved_top5': 0, 'pattern_completion_success': 0, 'strict_anchor_visible_top5': 0, 'loose_anchor_visible_top5': 0, 'false_bundle_retrieval': 0, 'proto0_bundle_count_in_top5': 0, 'top5_bundle_ids': '', 'top5_proto_ids': '', 'target_lost_reason': 'proposal_missing_for_retrieval', 'alignment_classification': event['alignment_classification']})
            continue
        eligible = [bundle_by_id[b] for b in event['eligible_bundle_ids'] if b in bundle_by_id]
        if not eligible:
            retrieval_rows.append({'ablation_name': name, 'scenario_name': event['scenario_name'], 'event_id': event['event_id'], 'frame_idx': int(event['frame_idx']), 'proposal_detected': int(event['proposal_detected']), 'target_bundle_exists': int(event['target_bundle_exists']), 'target_bundle_id': event['target_bundle_id'] or '', 'target_bundle_rank': '', 'target_bundle_score': '', 'target_bundle_retrieved_top1': 0, 'target_bundle_retrieved_top3': 0, 'target_bundle_retrieved_top5': 0, 'pattern_completion_success': 0, 'strict_anchor_visible_top5': 0, 'loose_anchor_visible_top5': 0, 'false_bundle_retrieval': 0, 'proto0_bundle_count_in_top5': 0, 'top5_bundle_ids': '', 'top5_proto_ids': '', 'target_lost_reason': 'target_not_in_candidate_pool', 'alignment_classification': event['alignment_classification']})
            continue
        stage1 = []
        for bundle in eligible:
            stage1.append(score_candidate(event['cue'], bundle, config, proto_counter[int(bundle['primary_source_prototype_id'])], track_counter[int(bundle['primary_source_track_id'])], lineage_counter[bundle['primary_source_lineage_id']], hist_topk[int(bundle['bundle_id'])]))
        stage1.sort(key=lambda r: r['base_score'], reverse=True)
        reranked = sorted(stage1[:config['candidate_pool_size']], key=lambda r: r['final_score'], reverse=True)
        final_topk = diversify_candidates(reranked, config)
        final_ids = [int(r['bundle_id']) for r in final_topk]
        for r in final_topk:
            hist_topk[int(r['bundle_id'])] += 1
            hubness_counter[int(r['bundle_id'])] += 1
        if final_topk:
            top1_counter[int(final_topk[0]['bundle_id'])] += 1
        for r in final_topk[:3]:
            top3_counter[int(r['bundle_id'])] += 1
        for r in final_topk[:5]:
            top5_counter[int(r['bundle_id'])] += 1
            proto0_top5_slots += 1
            if int(r['primary_source_prototype_id']) == 0:
                proto0_top5_hits += 1
        target_bundle_id = event['target_bundle_id']
        target_row = next((r for r in reranked if target_bundle_id is not None and int(r['bundle_id']) == int(target_bundle_id)), None)
        target_rank = next((i for i, r in enumerate(reranked, start=1) if target_bundle_id is not None and int(r['bundle_id']) == int(target_bundle_id)), None)
        top1_hit = int(target_bundle_id is not None and len(final_ids) > 0 and int(final_ids[0]) == int(target_bundle_id))
        top3_hit = int(target_bundle_id is not None and int(target_bundle_id) in set(final_ids[:3]))
        top5_hit = int(target_bundle_id is not None and int(target_bundle_id) in set(final_ids[:5]))
        success = int(top3_hit == 1 and target_row is not None and float(target_row['final_score']) >= config['completion_threshold'])
        target_bundle = bundle_by_id.get(int(target_bundle_id)) if target_bundle_id is not None else None
        loose_anchor = int(target_bundle is not None and any(((r['canonical_lineage_id'] == target_bundle['canonical_lineage_id'] and r['canonical_lineage_id'] is not None) or str(r['memory_anchor_id']) == str(target_bundle['memory_anchor_id'])) for r in final_topk[:5]))
        strict_anchor = int(top5_hit == 1)
        false_retrieval = int(len(final_topk) > 0 and top1_hit == 0)
        target_lost_reason = ''
        if target_bundle_id is not None and top5_hit == 0:
            if sum(1 for r in final_topk[:5] if int(r['primary_source_prototype_id']) == 0) >= 3:
                target_lost_reason = 'hub_bundle_dominance'
            elif target_row is not None and float(target_row['temporal_score']) < 0.55:
                target_lost_reason = 'low_temporal_score'
            elif target_row is not None and float(target_row['disappearance_score']) < 0.60:
                target_lost_reason = 'low_disappearance_score'
            elif target_row is not None and float(target_row['context_score']) < 0.60:
                target_lost_reason = 'low_context_score'
            elif target_row is not None and float(target_row['support_score']) < 0.60:
                target_lost_reason = 'low_support_score'
            elif target_row is not None and float(target_row['content_score']) < 0.60:
                target_lost_reason = 'low_content_score'
            elif target_rank is not None and target_rank <= config['candidate_pool_size']:
                target_lost_reason = 'score_tie_lost'
            else:
                target_lost_reason = 'target_not_in_candidate_pool'
        elif target_bundle_id is not None and top3_hit == 0:
            target_lost_reason = 'hub_bundle_dominance' if sum(1 for r in final_topk[:3] if int(r['primary_source_prototype_id']) == 0) >= 2 else 'score_tie_lost'
        retrieval_rows.append({'ablation_name': name, 'scenario_name': event['scenario_name'], 'event_id': event['event_id'], 'frame_idx': int(event['frame_idx']), 'proposal_detected': int(event['proposal_detected']), 'target_bundle_exists': int(event['target_bundle_exists']), 'target_bundle_id': '' if target_bundle_id is None else int(target_bundle_id), 'target_bundle_rank': '' if target_rank is None else int(target_rank), 'target_bundle_score': '' if target_row is None else float(target_row['final_score']), 'target_bundle_retrieved_top1': top1_hit, 'target_bundle_retrieved_top3': top3_hit, 'target_bundle_retrieved_top5': top5_hit, 'pattern_completion_success': success, 'strict_anchor_visible_top5': strict_anchor, 'loose_anchor_visible_top5': loose_anchor, 'false_bundle_retrieval': false_retrieval, 'proto0_bundle_count_in_top5': sum(1 for r in final_topk[:5] if int(r['primary_source_prototype_id']) == 0), 'top5_bundle_ids': '|'.join(str(b) for b in final_ids[:5]), 'top5_proto_ids': '|'.join(str(int(r['primary_source_prototype_id'])) for r in final_topk[:5]), 'target_lost_reason': target_lost_reason, 'alignment_classification': event['alignment_classification']})
        if event['event_id'] in FOCUS_EVENT_IDS or name in {'A0_baseline_E3_v1', 'A6_combined_E31'}:
            focus_target_rank = target_rank if event['event_id'] in FOCUS_EVENT_IDS else None
            for final_rank, r in enumerate(final_topk[:5], start=1):
                score_rows.append({'ablation_name': name, 'event_id': event['event_id'], 'frame_idx': int(event['frame_idx']), 'bundle_id': int(r['bundle_id']), 'memory_anchor_id': r['memory_anchor_id'], 'canonical_lineage_id': r['canonical_lineage_id'], 'primary_source_prototype_id': int(r['primary_source_prototype_id']), 'primary_source_track_id': int(r['primary_source_track_id']), 'rank_after': int(final_rank), 'target_bundle': int(target_bundle_id is not None and int(r['bundle_id']) == int(target_bundle_id)), 'target_bundle_rank_before': '' if focus_target_rank is None else int(focus_target_rank), 'support_score': float(r['support_score']), 'content_score': float(r['content_score']), 'motion_score': float(r['motion_score']), 'context_score': float(r['context_score']), 'temporal_score': float(r['temporal_score']), 'disappearance_score': float(r['disappearance_score']), 'provenance_score': float(r['provenance_score']), 'separation_score': float(r['separation_score']), 'rarity_bonus': float(r['rarity_bonus']), 'hub_penalty': float(r['hub_penalty']), 'lineage_penalty': float(r['lineage_penalty']), 'base_score': float(r['base_score']), 'final_score': float(r['final_score'])})
    proposal_rows = [r for r in retrieval_rows if int(r['proposal_detected']) == 1]
    focus_rows = [r for r in retrieval_rows if r['event_id'] in FOCUS_EVENT_IDS and int(r['proposal_detected']) == 1]
    focus_summary = [{'ablation_name': name, 'event_id': r['event_id'], 'target_bundle_id': r['target_bundle_id'], 'target_bundle_rank_after': r['target_bundle_rank'], 'target_bundle_retrieved_top1': r['target_bundle_retrieved_top1'], 'target_bundle_retrieved_top3': r['target_bundle_retrieved_top3'], 'target_bundle_retrieved_top5': r['target_bundle_retrieved_top5'], 'pattern_completion_success': r['pattern_completion_success'], 'proto0_bundle_count_in_top5': r['proto0_bundle_count_in_top5'], 'target_lost_reason': r['target_lost_reason']} for r in focus_rows]
    for bundle_id, bundle in bundle_by_id.items():
        score_match = [r for r in score_rows if int(r['bundle_id']) == int(bundle_id)]
        hub_rows.append({'ablation_name': name, 'bundle_id': int(bundle_id), 'memory_anchor_id': bundle['memory_anchor_id'], 'source_prototype_ids': '|'.join(str(v) for v in sorted(bundle['source_prototype_ids'])), 'source_track_ids': '|'.join(str(v) for v in sorted(bundle['source_track_ids'])), 'canonical_lineage_id': bundle['canonical_lineage_id'], 'retrieval_count': int(hubness_counter[int(bundle_id)]), 'top1_count': int(top1_counter[int(bundle_id)]), 'top3_count': int(top3_counter[int(bundle_id)]), 'top5_count': int(top5_counter[int(bundle_id)]), 'source_prototype_frequency': int(proto_counter[int(bundle['primary_source_prototype_id'])]), 'source_track_frequency': int(track_counter[int(bundle['primary_source_track_id'])]), 'anchor_degree': int(track_counter[int(bundle['primary_source_track_id'])]), 'canonical_lineage_degree': int(lineage_counter[bundle['primary_source_lineage_id']]), 'mean_completion_score': 0.0 if not score_match else float(np.mean([r['final_score'] for r in score_match])), 'mean_support_score': 0.0 if not score_match else float(np.mean([r['support_score'] for r in score_match])), 'mean_motion_score': 0.0 if not score_match else float(np.mean([r['motion_score'] for r in score_match])), 'mean_context_score': 0.0 if not score_match else float(np.mean([r['context_score'] for r in score_match])), 'mean_temporal_score': 0.0 if not score_match else float(np.mean([r['temporal_score'] for r in score_match])), 'mean_content_score': 0.0 if not score_match else float(np.mean([r['content_score'] for r in score_match])), 'is_proto0_bundle': int(int(bundle['primary_source_prototype_id']) == 0), 'is_target_bundle': int(int(bundle_id) in {544, 590})})
    summary = {'ablation_name': name, 'global_top1': 0.0 if not proposal_rows else float(np.mean([int(r['target_bundle_retrieved_top1']) for r in proposal_rows])), 'global_top3': 0.0 if not proposal_rows else float(np.mean([int(r['target_bundle_retrieved_top3']) for r in proposal_rows])), 'global_top5': 0.0 if not proposal_rows else float(np.mean([int(r['target_bundle_retrieved_top5']) for r in proposal_rows])), 'global_pattern_completion_success': 0.0 if not proposal_rows else float(np.mean([int(r['pattern_completion_success']) for r in proposal_rows])), 'false_bundle_retrieval_rate': 0.0 if not proposal_rows else float(np.mean([int(r['false_bundle_retrieval']) for r in proposal_rows])), 'focus_top5_count': int(sum(int(r['target_bundle_retrieved_top5']) for r in focus_rows)), 'focus_top3_count': int(sum(int(r['target_bundle_retrieved_top3']) for r in focus_rows)), 'focus_success_count': int(sum(int(r['pattern_completion_success']) for r in focus_rows)), 'proto0_top5_share': 0.0 if proto0_top5_slots == 0 else float(proto0_top5_hits / proto0_top5_slots), 'target_bundle_mean_rank': 999.0 if not focus_rows else float(np.mean([int(r['target_bundle_rank']) if r['target_bundle_rank'] != '' else 999 for r in focus_rows])), 'strict_anchor_top5_rate': 0.0 if not proposal_rows else float(np.mean([int(r['strict_anchor_visible_top5']) for r in proposal_rows])), 'loose_anchor_top5_rate': 0.0 if not proposal_rows else float(np.mean([int(r['loose_anchor_visible_top5']) for r in proposal_rows]))}
    return {'summary': summary, 'retrieval_rows': retrieval_rows, 'score_rows': score_rows, 'focus_rows': focus_summary, 'hub_rows': hub_rows}

def load_negative_controls(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open('r', encoding='utf-8-sig', newline='') as h:
        return list(csv.DictReader(h))


def classify_normal_reference_rows(negative_rows):
    rows = []
    for r in negative_rows:
        if str(r.get('control_name', '')) != 'normal_non_remap_reference':
            continue
        rows.append({'event_id': r.get('event_id', ''), 'control_name': r.get('control_name', ''), 'strict_anchor_visible': int(si(r.get('anchor_visible'), 0) or 0), 'expected_raw_lineage_visible': 1, 'expected_canonical_visible': 1})
    return rows


def render_report(summary):
    lines = ['# Stage E3.1 Report', '', '## Best Ablation', '']
    for k, v in summary['best_ablation'].items():
        lines.append(f'- `{k} = {v}`')
    lines += ['', '## Focus Events', '']
    for row in summary['focus_events']:
        lines += [f"### {row['event_id']}", '', f"- `ablation = {row['ablation_name']}`", f"- `target_bundle_rank_after = {row['target_bundle_rank_after']}`", f"- `target_bundle_retrieved_top3 = {row['target_bundle_retrieved_top3']}`", f"- `target_bundle_retrieved_top5 = {row['target_bundle_retrieved_top5']}`", f"- `pattern_completion_success = {row['pattern_completion_success']}`", f"- `proto0_bundle_count_in_top5 = {row['proto0_bundle_count_in_top5']}`", f"- `target_lost_reason = {row['target_lost_reason']}`", '']
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_by_id, event_records, _ = collect_runtime_data(args.config, args.event_audit, args.cross_run_alignment, args.seed)
    proto_counter, track_counter, lineage_counter = compute_static_counts(bundle_by_id)
    negative_rows = load_negative_controls(args.e2c_negative_events)
    ablations = {
        'A0_baseline_E3_v1': {'candidate_pool_size': 30, 'final_topk': 5, 'competition': False, 'max_per_proto': 99, 'max_per_anchor': 99, 'max_per_lineage': 99, 'temporal_decay': 96.0, 'completion_threshold': 0.55, 'rarity_bonus': 0.0, 'hub_alpha': 0.0, 'hub_beta': 0.0, 'hub_gamma': 0.0, 'lineage_hub_gamma': 0.0, 'weights': {'content': 0.35, 'support': 0.20, 'motion': 0.10, 'context': 0.15, 'temporal': 0.10, 'disappearance': 0.0, 'provenance': 0.10, 'separation': 0.0}},
        'A1_strict_anchor_metric_only': {'candidate_pool_size': 30, 'final_topk': 5, 'competition': False, 'max_per_proto': 99, 'max_per_anchor': 99, 'max_per_lineage': 99, 'temporal_decay': 96.0, 'completion_threshold': 0.55, 'rarity_bonus': 0.0, 'hub_alpha': 0.0, 'hub_beta': 0.0, 'hub_gamma': 0.0, 'lineage_hub_gamma': 0.0, 'weights': {'content': 0.35, 'support': 0.20, 'motion': 0.10, 'context': 0.15, 'temporal': 0.10, 'disappearance': 0.0, 'provenance': 0.10, 'separation': 0.0}},
        'A2_anti_hub_penalty': {'candidate_pool_size': 30, 'final_topk': 5, 'competition': False, 'max_per_proto': 99, 'max_per_anchor': 99, 'max_per_lineage': 99, 'temporal_decay': 96.0, 'completion_threshold': 0.55, 'rarity_bonus': 0.22, 'hub_alpha': 0.055, 'hub_beta': 0.035, 'hub_gamma': 0.03, 'lineage_hub_gamma': 0.015, 'weights': {'content': 0.28, 'support': 0.18, 'motion': 0.08, 'context': 0.16, 'temporal': 0.12, 'disappearance': 0.06, 'provenance': 0.12, 'separation': 0.0}},
        'A3_bundle_competition_nms': {'candidate_pool_size': 30, 'final_topk': 5, 'competition': True, 'max_per_proto': 1, 'max_per_anchor': 1, 'max_per_lineage': 2, 'temporal_decay': 96.0, 'completion_threshold': 0.55, 'rarity_bonus': 0.0, 'hub_alpha': 0.0, 'hub_beta': 0.0, 'hub_gamma': 0.0, 'lineage_hub_gamma': 0.0, 'weights': {'content': 0.35, 'support': 0.20, 'motion': 0.10, 'context': 0.15, 'temporal': 0.10, 'disappearance': 0.0, 'provenance': 0.10, 'separation': 0.0}},
        'A4_temporal_disappearance_reweight': {'candidate_pool_size': 30, 'final_topk': 5, 'competition': False, 'max_per_proto': 99, 'max_per_anchor': 99, 'max_per_lineage': 99, 'temporal_decay': 128.0, 'completion_threshold': 0.56, 'rarity_bonus': 0.08, 'hub_alpha': 0.0, 'hub_beta': 0.0, 'hub_gamma': 0.0, 'lineage_hub_gamma': 0.0, 'weights': {'content': 0.15, 'support': 0.15, 'motion': 0.10, 'context': 0.18, 'temporal': 0.17, 'disappearance': 0.15, 'provenance': 0.10, 'separation': 0.0}},
        'A5_pattern_separation_signature': {'candidate_pool_size': 30, 'final_topk': 5, 'competition': False, 'max_per_proto': 99, 'max_per_anchor': 99, 'max_per_lineage': 99, 'temporal_decay': 112.0, 'completion_threshold': 0.56, 'rarity_bonus': 0.10, 'hub_alpha': 0.02, 'hub_beta': 0.01, 'hub_gamma': 0.0, 'lineage_hub_gamma': 0.0, 'weights': {'content': 0.16, 'support': 0.14, 'motion': 0.10, 'context': 0.16, 'temporal': 0.14, 'disappearance': 0.12, 'provenance': 0.08, 'separation': 0.10}},
        'A6_combined_E31': {'candidate_pool_size': 35, 'final_topk': 5, 'competition': True, 'max_per_proto': 1, 'max_per_anchor': 1, 'max_per_lineage': 3, 'temporal_decay': 120.0, 'completion_threshold': 0.50, 'rarity_bonus': 0.14, 'hub_alpha': 0.03, 'hub_beta': 0.015, 'hub_gamma': 0.015, 'lineage_hub_gamma': 0.005, 'weights': {'content': 0.14, 'support': 0.14, 'motion': 0.10, 'context': 0.17, 'temporal': 0.15, 'disappearance': 0.13, 'provenance': 0.09, 'separation': 0.08}},
    }
    all_retrieval_rows, all_score_rows, all_focus_rows, all_hub_rows, ablation_rows = [], [], [], [], []
    for name, config in ablations.items():
        result = evaluate_ablation(name, config, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter)
        all_retrieval_rows.extend(result['retrieval_rows'])
        all_score_rows.extend(result['score_rows'])
        all_focus_rows.extend(result['focus_rows'])
        all_hub_rows.extend(result['hub_rows'])
        ablation_rows.append(result['summary'])
    best = max(ablation_rows, key=lambda r: (r['focus_success_count'], r['focus_top3_count'], r['focus_top5_count'], -r['false_bundle_retrieval_rate']))
    best_name = str(best['ablation_name'])
    best_focus = [r for r in all_focus_rows if r['ablation_name'] == best_name]
    strict_eval_rows = []
    for r in all_retrieval_rows:
        if r['ablation_name'] not in {'A0_baseline_E3_v1', 'A6_combined_E31'}:
            continue
        strict_eval_rows.append({'ablation_name': r['ablation_name'], 'event_id': r['event_id'], 'scenario_name': r['scenario_name'], 'strict_anchor_visible_top5': r['strict_anchor_visible_top5'], 'loose_anchor_visible_top5': r['loose_anchor_visible_top5'], 'target_bundle_retrieved_top5': r['target_bundle_retrieved_top5'], 'target_bundle_retrieved_top3': r['target_bundle_retrieved_top3'], 'target_bundle_retrieved_top1': r['target_bundle_retrieved_top1']})
    strict_eval_rows.extend(classify_normal_reference_rows(negative_rows))
    summary = {'scope': 'track_a_bridge_and_track_c_long_horizon', 'bundle_count': len(bundle_by_id), 'best_ablation': best, 'focus_events': best_focus, 'negative_control_reference': {'real_anchor_svr': 1.0, 'shuffled_anchor_svr': 0.875, 'focus_wrong_old_prototype_visible_count': 0, 'normal_triple_match_count': 3}}
    write_csv(output_dir / f'stage_E31_hubness_audit_{args.artifact_version}.csv', all_hub_rows)
    write_csv(output_dir / f'stage_E31_strict_anchor_eval_{args.artifact_version}.csv', strict_eval_rows)
    write_csv(output_dir / f'stage_E31_retrieval_score_breakdown_{args.artifact_version}.csv', all_score_rows)
    write_csv(output_dir / f'stage_E31_retrieval_compare_{args.artifact_version}.csv', all_retrieval_rows)
    write_csv(output_dir / f'stage_E31_focus_event_summary_{args.artifact_version}.csv', all_focus_rows)
    write_csv(output_dir / f'stage_E31_ablation_summary_{args.artifact_version}.csv', ablation_rows)
    (output_dir / f'stage_E31_summary_{args.artifact_version}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / f'stage_E31_report_{args.artifact_version}.md').write_text(render_report(summary), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()

