from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from datasets import SyntheticStreamGenerator
from experiments.phase3r_utils import build_phase3_scenario_map, load_config_payload, write_csv
from experiments.v3_utils import TRACK_A_NAME, TRACK_C_NAME
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker
SCENARIO_NAMES = (TRACK_A_NAME, TRACK_C_NAME)
FOCUS_EVENT_IDS = {'M-RE-TC-012', 'M-RE-TC-013', 'M-RE-TC-014'}

def parse_args():
    p = argparse.ArgumentParser(description='Run v3 Stage E3 episodic bundle and pattern completion.')
    p.add_argument('--config', default='configs/bridge_synth_generic_v1.yaml')
    p.add_argument('--event-audit', default='results/v3_e1/stage_E1_event_audit_v1.csv')
    p.add_argument('--cross-run-alignment', default='results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv')
    p.add_argument('--output-dir', default='results/v3_e3')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--artifact-version', default='v1')
    return p.parse_args()

def si(v, d=None):
    if v in (None, ''): return d
    try: return int(v)
    except Exception:
        try: return int(float(v))
        except Exception: return d

def load_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as h:
        return list(csv.DictReader(h))

def load_events(path):
    by = {TRACK_A_NAME: [], TRACK_C_NAME: []}
    for r in load_csv(path):
        s = str(r.get('scenario_name', ''))
        if s in by: by[s].append(dict(r))
    for s in by: by[s].sort(key=lambda r: (si(r.get('reappear_frame'), -1), si(r.get('instance_id'), -1)))
    return by

def load_alignment(path):
    return {str(r.get('event_id', '')): dict(r) for r in load_csv(path)}

def norm(x):
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return np.zeros_like(a) if n <= 1e-8 else a / n

def cosine(a, b):
    aa, bb = norm(a), norm(b)
    if aa.size == 0 or bb.size == 0: return 0.0
    m = min(aa.size, bb.size)
    aa, bb = aa[:m], bb[:m]
    return float(np.clip(np.dot(aa, bb), -1.0, 1.0) * 0.5 + 0.5)

def shape_sig(box, frame_shape):
    h, w = frame_shape
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    area = bw * bh
    return np.asarray([bw / max(w, 1), bh / max(h, 1), bw / bh, area / max(w * h, 1)], dtype=np.float32)

def context_sig(centroid, frame_shape, proposal_count):
    h, w = frame_shape
    cx, cy = centroid
    return np.asarray([cx / max(w, 1), cy / max(h, 1), min(float(proposal_count) / 8.0, 1.0)], dtype=np.float32)

def motion_sig(velocity):
    v = np.asarray(velocity, dtype=np.float32).reshape(-1)
    if v.size < 2: v = np.pad(v, (0, 2 - v.size))
    mag = float(np.linalg.norm(v[:2]))
    if mag <= 1e-8: return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray([v[0] / mag, v[1] / mag, min(mag / 10.0, 1.0)], dtype=np.float32)

def iou(a, b):
    if a is None or b is None: return 0.0
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union

def gt_box(frame, instance_id):
    try: idx = list(frame.instance_ids).index(int(instance_id))
    except ValueError: return None
    return tuple(int(v) for v in frame.boxes[idx])

def pick_proposal(proposals, target_box):
    best = None; best_iou = -1.0
    for idx, p in enumerate(proposals):
        box = tuple(int(v) for v in p.box); ov = iou(box, target_box)
        if ov > best_iou: best_iou, best = ov, (idx, p, ov)
    return best

def find_assignment(assignments, proposal_id):
    for a in assignments:
        if int(a.proposal_index) == int(proposal_id): return a
    return None

def track_snap(track):
    return {
        'track_id': int(track.track_id), 'state': str(track.state), 'box': tuple(int(v) for v in track.box),
        'centroid': tuple(float(v) for v in track.centroid), 'signature': np.asarray(track.signature, dtype=np.float32),
        'velocity': np.asarray(track.velocity, dtype=np.float32), 'score': float(track.score), 'age': int(track.age),
        'hit_count': int(getattr(track, 'hit_count', getattr(track, 'hits', 0)) or 0), 'gap_length': int(getattr(track, 'gap_length', 0) or 0),
        'last_seen_frame': int(getattr(track, 'last_seen_frame', 0) or 0), 'prototype_id': None if track.prototype_id is None else int(track.prototype_id),
        'lineage_id': None if track.lineage_id is None else int(track.lineage_id),
        'continuity_lineage_id': None if getattr(track, 'continuity_lineage_id', None) is None else int(track.continuity_lineage_id),
    }

def bundle_worthy(t):
    return t['prototype_id'] is not None and t['hit_count'] >= 3 and t['age'] >= 4

def bundle_level(t):
    if t['hit_count'] >= 8: return 'stable'
    if t['hit_count'] >= 5: return 'stabilizing'
    return 'candidate'

def bundle_access(t):
    return float(np.clip(0.25 + 0.06 * t['hit_count'] + 0.01 * t['age'], 0.0, 1.0))

def make_anchor(scenario_name, t, frame_idx):
    return f"{scenario_name}::track_{t['track_id']}::proto_{t['prototype_id']}::frame_{frame_idx}"

def make_bundle(bundle_id, scenario_name, t, frame_idx, frame_shape, proposal_count):
    lineage = t['continuity_lineage_id'] if t['continuity_lineage_id'] is not None else t['lineage_id']
    return {
        'bundle_id': int(bundle_id), 'memory_anchor_id': make_anchor(scenario_name, t, frame_idx), 'canonical_lineage_id': lineage,
        'source_track_ids': {t['track_id']}, 'source_prototype_ids': {t['prototype_id']}, 'source_lineage_ids': set([lineage]) if lineage is not None else set(),
        'runtime_lineage_refs': set([t['lineage_id']]) if t['lineage_id'] is not None else set(), 'runtime_prototype_refs': {t['prototype_id']},
        'content_signature': norm(t['signature']), 'support_signature': shape_sig(t['box'], frame_shape), 'motion_signature': motion_sig(t['velocity']),
        'context_signature': context_sig(t['centroid'], frame_shape, proposal_count),
        'disappearance_signature': np.asarray([float(frame_idx), float(t['score']), float(t['gap_length']), float(t['last_seen_frame'])], dtype=np.float32),
        'temporal_signature': np.asarray([float(t['age']), float(t['hit_count']), float(t['gap_length']), float(t['last_seen_frame'])], dtype=np.float32),
        'accessibility_score': bundle_access(t), 'stability_level': bundle_level(t), 'reactivation_count': 0,
        'replay_priority': 0.0, 'competition_score': 0.0, 'consolidation_state': 'raw', 'reconsolidation_version': 0,
        'created_frame': int(frame_idx), 'last_reactivated_frame': int(frame_idx), 'last_source_frame': int(t['last_seen_frame']),
    }

def cue_from_obs(proposal, assignment, frame_shape, proposal_count):
    box = tuple(int(v) for v in (assignment.box if assignment is not None else proposal.box))
    centroid = tuple(float(v) for v in (assignment.centroid if assignment is not None else proposal.centroid))
    appearance = np.zeros(8, dtype=np.float32) if assignment is None else norm(np.asarray(assignment.signature, dtype=np.float32))
    return {'support_shape': shape_sig(box, frame_shape), 'appearance_proxy': appearance, 'local_context': context_sig(centroid, frame_shape, proposal_count), 'quality': float(assignment.score if assignment is not None else proposal.score)}

def score_bundle(cue, bundle, frame_idx):
    support = cosine(cue['support_shape'], bundle['support_signature'])
    content = cosine(cue['appearance_proxy'], bundle['content_signature'])
    context = cosine(cue['local_context'], bundle['context_signature'])
    motion = 0.5
    temporal = float(np.exp(-max(1, frame_idx - bundle['last_source_frame']) / 96.0))
    stability = 0.1 * float(bundle['accessibility_score'])
    total = float(np.clip(0.35 * content + 0.20 * support + 0.15 * context + 0.10 * motion + 0.10 * temporal + stability, 0.0, 1.0))
    breakdown = {'support_score': support, 'motion_score': motion, 'context_score': context, 'temporal_score': temporal, 'content_score': content}
    missing = [k for k, v in breakdown.items() if v < 0.35]
    return total, breakdown, missing

def summarize(retrieval_rows, focus_rows, bundle_count):
    prop = [r for r in retrieval_rows if int(r.get('proposal_detected', 0) or 0) == 1]
    ns = [r for r in prop if str(r.get('alignment_classification', '')) == 'runtime_namespace_shift']
    def rate(key, rows):
        return 0.0 if not rows else sum(int(r.get(key, 0) or 0) for r in rows) / len(rows)
    return {
        'overall': {
            'bundle_created': int(bundle_count),
            'proposal_detected_events': len(prop),
            'target_bundle_present_rate': rate('target_bundle_present', prop),
            'bundle_retrieval_top1': rate('target_bundle_retrieved_top1', prop),
            'bundle_retrieval_top3': rate('target_bundle_retrieved_top3', prop),
            'bundle_retrieval_top5': rate('target_bundle_retrieved_top5', prop),
            'pattern_completion_success_rate': rate('pattern_completion_success', prop),
            'anchor_reactivation_rate': rate('target_bundle_retrieved_top3', prop),
            'runtime_namespace_shift_recovered_rate': rate('pattern_completion_success', ns),
            'false_bundle_retrieval_rate': rate('false_bundle_retrieval', prop),
        },
        'focus_events': focus_rows,
    }

def render_report(summary):
    lines = ['# Stage E3 Report', '', '## Overall', '']
    for k, v in summary['overall'].items(): lines.append(f'- `{k} = {v}`')
    lines += ['', '## Focus Events', '']
    for row in summary['focus_events']:
        lines += [f"### {row['event_id']}", '',
                  f"- `target_bundle_exists = {row['target_bundle_exists']}`",
                  f"- `target_bundle_retrieved_top1 = {row['target_bundle_retrieved_top1']}`",
                  f"- `target_bundle_retrieved_top3 = {row['target_bundle_retrieved_top3']}`",
                  f"- `target_bundle_retrieved_top5 = {row['target_bundle_retrieved_top5']}`",
                  f"- `pattern_completion_success = {row['pattern_completion_success']}`",
                  f"- `completion_confidence = {row['completion_confidence']}`",
                  f"- `failure_reason = {row['failure_reason']}`", '']
    return '\n'.join(lines) + '\n'

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_config_payload(args.config)
    scenario_map = build_phase3_scenario_map(args.config)
    event_rows_by_scenario = load_events(args.event_audit)
    alignment_map = load_alignment(args.cross_run_alignment)
    bundle_inventory_rows, bundle_write_rows, bundle_index_rows = [], [], []
    retrieval_rows, completion_rows, focus_rows = [], [], []
    next_bundle_id = 1
    for scenario_name in SCENARIO_NAMES:
        sequence = SyntheticStreamGenerator(scenario_map[scenario_name], seed=args.seed).generate_sequence(0)
        encoder = MinimalSpikeEncoder(**payload['model']['spike_encoder'])
        field = MinimalObjectnessField(**payload['field'])
        tracker = MinimalTemporalIdentityTracker(**payload['tracking'])
        memory = MinimalPrototypeMemory(**payload['memory'])
        bundles, prev_tracks, prev_memory_output = [], {}, None
        frame_shape = tuple(int(v) for v in sequence.frames[0].frame.shape[:2])
        events_at_frame = {}
        for e in event_rows_by_scenario.get(scenario_name, []): events_at_frame.setdefault(int(e['reappear_frame']), []).append(e)
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
            created_ids = []
            for track_id, prev_t in prev_tracks.items():
                cur_t = current_tracks.get(track_id)
                trigger, source_t = None, None
                if cur_t is None and prev_t['state'] in {'active', 'dormant', 'ghost'}:
                    trigger, source_t = 'track_missing_from_registry', prev_t
                elif cur_t is not None and prev_t['state'] == 'active' and cur_t['state'] in {'dormant', 'ghost', 'retired'}:
                    trigger, source_t = f"active_to_{cur_t['state']}", prev_t
                if trigger is None or source_t is None: continue
                dedup = (source_t['track_id'], source_t['prototype_id'], source_t['last_seen_frame'])
                if not bundle_worthy(source_t):
                    bundle_write_rows.append({'scenario_name': scenario_name, 'frame_idx': frame_idx, 'track_id': source_t['track_id'], 'prototype_id': source_t['prototype_id'], 'lineage_id': source_t['continuity_lineage_id'] if source_t['continuity_lineage_id'] is not None else source_t['lineage_id'], 'write_trigger': trigger, 'bundle_written': 0, 'bundle_id': '', 'memory_anchor_id': '', 'skip_reason': 'not_bundle_worthy', 'support_quality': source_t['score'], 'track_age': source_t['age'], 'disappearance_context_available': 1})
                    continue
                if any((next(iter(b['source_track_ids'])), next(iter(b['source_prototype_ids'])), b['last_source_frame']) == dedup for b in bundles):
                    continue
                b = make_bundle(next_bundle_id, scenario_name, source_t, frame_idx, frame_shape, len(objectness_output.proposals))
                next_bundle_id += 1; bundles.append(b); created_ids.append(b['bundle_id'])
                bundle_write_rows.append({'scenario_name': scenario_name, 'frame_idx': frame_idx, 'track_id': source_t['track_id'], 'prototype_id': source_t['prototype_id'], 'lineage_id': b['canonical_lineage_id'], 'write_trigger': trigger, 'bundle_written': 1, 'bundle_id': b['bundle_id'], 'memory_anchor_id': b['memory_anchor_id'], 'skip_reason': '', 'support_quality': source_t['score'], 'track_age': source_t['age'], 'disappearance_context_available': 1})
            bundle_index_rows.append({'scenario_name': scenario_name, 'frame_idx': frame_idx, 'bundle_count': len(bundles), 'created_bundle_ids': '|'.join(str(v) for v in created_ids), 'proposal_count': len(objectness_output.proposals), 'active_track_count': len(tracking_output.active_tracks), 'dormant_track_count': len(tracking_output.dormant_tracks), 'ghost_track_count': len(tracking_output.ghost_tracks), 'retired_track_count': len(tracking_output.retired_tracks)})
            if frame_idx in events_at_frame:
                for event in events_at_frame[frame_idx]:
                    instance_id = int(event['instance_id'])
                    target_box = gt_box(current_frame, instance_id)
                    picked = pick_proposal(objectness_output.proposals, target_box) if target_box is not None else None
                    proposal_detected = int(si(event.get('proposal_detected'), 0) or 0)
                    if picked is None:
                        retrieval_rows.append({'scenario_name': scenario_name, 'event_id': event['ledger_event_id'], 'frame_idx': frame_idx, 'proposal_id': '', 'proposal_detected': proposal_detected, 'cue_available': 0, 'candidate_bundle_count': len(bundles), 'top1_bundle_id': '', 'top1_memory_anchor_id': '', 'top1_canonical_lineage_id': '', 'target_bundle_present': 0, 'target_bundle_id': '', 'target_bundle_memory_anchor_id': '', 'target_bundle_retrieved_top1': 0, 'target_bundle_retrieved_top3': 0, 'target_bundle_retrieved_top5': 0, 'pattern_completion_success': 0, 'completion_confidence': 0.0, 'retrieval_failure_reason': 'proposal_missing_for_retrieval', 'alignment_classification': alignment_map.get(str(event['ledger_event_id']), {}).get('classification', ''), 'false_bundle_retrieval': 0})
                        continue
                    proposal_id, proposal, _ = picked
                    assignment = find_assignment(tracking_output.assignments, proposal_id)
                    cue = cue_from_obs(proposal, assignment, frame_shape, len(objectness_output.proposals))
                    eligible = [b for b in bundles if int(b['created_frame']) < frame_idx]
                    old_track_id, old_proto_id = si(event.get('old_track_id'), -1), si(event.get('old_prototype_id'), -1)
                    targets = [b for b in eligible if (old_track_id >= 0 and old_track_id in b['source_track_ids']) or (old_proto_id >= 0 and old_proto_id in b['source_prototype_ids'])]
                    targets.sort(key=lambda b: (int(b['created_frame']), int(b['bundle_id'])), reverse=True)
                    target_bundle = targets[0] if targets else None
                    scored = []
                    for b in eligible:
                        total, breakdown, missing = score_bundle(cue, b, frame_idx)
                        scored.append((total, b, breakdown, missing))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    topk = scored[:5]
                    topk_ids = [int(b['bundle_id']) for _, b, _, _ in topk]
                    top1 = topk[0] if topk else None
                    top1_id = '' if top1 is None else int(top1[1]['bundle_id'])
                    top1_anchor = '' if top1 is None else top1[1]['memory_anchor_id']
                    top1_lineage = '' if top1 is None else top1[1]['canonical_lineage_id']
                    conf = 0.0 if top1 is None else float(top1[0])
                    fail_reason = '' if top1 is not None and conf >= 0.55 else ('low_completion_confidence' if top1 is not None else 'no_candidate_bundle')
                    target_present = int(target_bundle is not None)
                    top1_hit = int(target_bundle is not None and len(topk_ids) > 0 and topk_ids[0] == int(target_bundle['bundle_id']))
                    top3_hit = int(target_bundle is not None and int(target_bundle['bundle_id']) in set(topk_ids[:3]))
                    top5_hit = int(target_bundle is not None and int(target_bundle['bundle_id']) in set(topk_ids))
                    success = int(top3_hit == 1 and conf >= 0.55)
                    if success and target_bundle is not None:
                        target_bundle['reactivation_count'] += 1
                        target_bundle['last_reactivated_frame'] = frame_idx
                        target_bundle['accessibility_score'] = float(np.clip(target_bundle['accessibility_score'] + 0.05, 0.0, 1.0))
                        target_bundle['reconsolidation_version'] += 1
                    false_retrieval = int(top1_id != '' and (target_bundle is None or int(top1_id) != int(target_bundle['bundle_id'])))
                    retrieval_rows.append({'scenario_name': scenario_name, 'event_id': event['ledger_event_id'], 'frame_idx': frame_idx, 'proposal_id': int(proposal_id), 'proposal_detected': proposal_detected, 'cue_available': 1, 'candidate_bundle_count': len(eligible), 'top1_bundle_id': top1_id, 'top1_memory_anchor_id': top1_anchor, 'top1_canonical_lineage_id': top1_lineage, 'target_bundle_present': target_present, 'target_bundle_id': '' if target_bundle is None else int(target_bundle['bundle_id']), 'target_bundle_memory_anchor_id': '' if target_bundle is None else target_bundle['memory_anchor_id'], 'target_bundle_retrieved_top1': top1_hit, 'target_bundle_retrieved_top3': top3_hit, 'target_bundle_retrieved_top5': top5_hit, 'pattern_completion_success': success, 'completion_confidence': conf, 'retrieval_failure_reason': fail_reason, 'alignment_classification': alignment_map.get(str(event['ledger_event_id']), {}).get('classification', ''), 'false_bundle_retrieval': false_retrieval})
                    if str(event['ledger_event_id']) in FOCUS_EVENT_IDS:
                        focus_rows.append({'event_id': str(event['ledger_event_id']), 'target_anchor_uid': alignment_map.get(str(event['ledger_event_id']), {}).get('target_anchor_uid', ''), 'old_prototype_id': old_proto_id, 'old_track_id': old_track_id, 'target_bundle_exists': target_present, 'target_bundle_id': '' if target_bundle is None else int(target_bundle['bundle_id']), 'target_bundle_memory_anchor_id': '' if target_bundle is None else target_bundle['memory_anchor_id'], 'target_bundle_retrieved_top1': top1_hit, 'target_bundle_retrieved_top3': top3_hit, 'target_bundle_retrieved_top5': top5_hit, 'pattern_completion_success': success, 'completion_confidence': conf, 'runtime_namespace_shift_recovered': int(alignment_map.get(str(event['ledger_event_id']), {}).get('classification', '') == 'runtime_namespace_shift' and success == 1), 'failure_reason': fail_reason})
                    for rank, (score, b, breakdown, missing) in enumerate(topk, start=1):
                        completion_rows.append({'scenario_name': scenario_name, 'event_id': event['ledger_event_id'], 'proposal_id': int(proposal_id), 'bundle_id': int(b['bundle_id']), 'memory_anchor_id': b['memory_anchor_id'], 'canonical_lineage_id': b['canonical_lineage_id'], 'completion_confidence': float(score), 'support_score': float(breakdown['support_score']), 'motion_score': float(breakdown['motion_score']), 'context_score': float(breakdown['context_score']), 'temporal_score': float(breakdown['temporal_score']), 'content_score': float(breakdown['content_score']), 'missing_cues': '|'.join(missing), 'rank': int(rank), 'target_bundle': int(target_bundle is not None and int(b['bundle_id']) == int(target_bundle['bundle_id'])), 'pattern_completion_success': int(rank <= 3 and target_bundle is not None and int(b['bundle_id']) == int(target_bundle['bundle_id']) and float(score) >= 0.55)})
            prev_tracks = current_tracks
        for b in bundles:
            bundle_inventory_rows.append({'scenario_name': scenario_name, 'bundle_id': int(b['bundle_id']), 'memory_anchor_id': b['memory_anchor_id'], 'canonical_lineage_id': b['canonical_lineage_id'], 'source_track_ids': '|'.join(str(v) for v in sorted(b['source_track_ids'])), 'source_prototype_ids': '|'.join(str(v) for v in sorted(b['source_prototype_ids'])), 'source_lineage_ids': '|'.join(str(v) for v in sorted(b['source_lineage_ids'])), 'runtime_lineage_refs': '|'.join(str(v) for v in sorted(b['runtime_lineage_refs'])), 'runtime_prototype_refs': '|'.join(str(v) for v in sorted(b['runtime_prototype_refs'])), 'stability_level': b['stability_level'], 'accessibility_score': float(b['accessibility_score']), 'reactivation_count': int(b['reactivation_count']), 'created_frame': int(b['created_frame']), 'last_reactivated_frame': int(b['last_reactivated_frame'])})
    report = summarize(retrieval_rows, focus_rows, len(bundle_inventory_rows))
    write_csv(output_dir / f'stage_E3_bundle_inventory_{args.artifact_version}.csv', bundle_inventory_rows)
    write_csv(output_dir / f'stage_E3_bundle_write_trace_{args.artifact_version}.csv', bundle_write_rows)
    write_csv(output_dir / f'stage_E3_bundle_index_trace_{args.artifact_version}.csv', bundle_index_rows)
    write_csv(output_dir / f'stage_E3_retrieval_trace_{args.artifact_version}.csv', retrieval_rows)
    write_csv(output_dir / f'stage_E3_pattern_completion_events_{args.artifact_version}.csv', completion_rows)
    write_csv(output_dir / f'stage_E3_focus_event_summary_{args.artifact_version}.csv', focus_rows)
    (output_dir / f'stage_E3_summary_{args.artifact_version}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / f'stage_E3_report_{args.artifact_version}.md').write_text(render_report(report), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
if __name__ == '__main__':
    main()
