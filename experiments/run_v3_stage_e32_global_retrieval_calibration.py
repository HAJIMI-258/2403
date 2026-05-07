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

from experiments import run_v3_stage_e31_retrieval_competition_repair as e31


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run v3 Stage E3.2 global episodic retrieval calibration.')
    p.add_argument('--config', default='configs/bridge_synth_generic_v1.yaml')
    p.add_argument('--event-audit', default='results/v3_e1/stage_E1_event_audit_v1.csv')
    p.add_argument('--cross-run-alignment', default='results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv')
    p.add_argument('--e2c-negative-events', default='results/v3_e2c/stage_E2C_negative_control_events_v1.csv')
    p.add_argument('--output-dir', default='results/v3_e32')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--artifact-version', default='v1')
    return p.parse_args()


def geomean(values: list[float], eps: float = 1e-4) -> float:
    vals = np.asarray([max(eps, float(v)) for v in values], dtype=np.float64)
    return float(np.exp(np.mean(np.log(vals))))


def init_bundle_states(bundle_by_id: dict[int, dict[str, Any]], proto_counter: Counter[int], track_counter: Counter[int]) -> dict[int, dict[str, Any]]:
    states = {}
    for bundle_id, bundle in bundle_by_id.items():
        proto_freq = int(proto_counter[int(bundle['primary_source_prototype_id'])])
        track_freq = int(track_counter[int(bundle['primary_source_track_id'])])
        specificity = 0.6 / (1.0 + math.log1p(proto_freq)) + 0.4 / (1.0 + math.log1p(track_freq))
        hubness = 0.6 * math.log1p(proto_freq) + 0.4 * math.log1p(track_freq)
        states[int(bundle_id)] = {
            'accessibility_score': float(bundle['accessibility_score']),
            'specificity_score': float(specificity),
            'hubness_score': float(hubness),
            'cue_consensus_score': 0.0,
            'reactivation_count': 0,
            'false_retrieval_count': 0,
            'suppression_count': 0,
            'reconsolidation_count': 0,
            'replay_priority': 0.0,
            'suppression_score': 0.0,
            'runtime_lineage_refs': set(bundle.get('runtime_lineage_refs', set())),
            'accessibility_state': 'candidate',
        }
    return states


def state_label(state: dict[str, Any]) -> str:
    acc = float(state['accessibility_score'])
    sup = float(state['suppression_score'])
    if sup >= 0.5:
        return 'suppressed'
    if acc >= 0.75:
        return 'stable'
    if acc >= 0.55:
        return 'stabilizing'
    if acc >= 0.35:
        return 'accessible'
    if acc >= 0.18:
        return 'candidate'
    return 'latent'


def cue_disagreement(row: dict[str, Any]) -> float:
    supportive = float(np.mean([row['context_score'], row['temporal_score'], row['disappearance_score'], row['motion_score']]))
    return float(max(0.0, row['content_score'] - supportive))


def score_candidate_e32(base_row: dict[str, Any], state: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    consensus = geomean([base_row['support_score'], base_row['motion_score'], base_row['context_score'], base_row['temporal_score'], base_row['disappearance_score']])
    disagreement = cue_disagreement(base_row)
    accessibility_bonus = cfg['acc_bonus'] * float(state['accessibility_score'])
    replay_bonus = cfg['replay_bonus'] * float(state['replay_priority'])
    recon_bonus = cfg['recon_bonus'] * min(float(state['reconsolidation_count']) / 3.0, 1.0)
    consensus_bonus = cfg['consensus_bonus'] * consensus
    competitor_penalty = cfg['suppression_penalty'] * float(state['suppression_score'])
    ambiguity_penalty = cfg['ambiguity_penalty'] * disagreement
    final_score = float(base_row['final_score'] + accessibility_bonus + replay_bonus + recon_bonus + consensus_bonus - competitor_penalty - ambiguity_penalty)
    out = dict(base_row)
    out.update({
        'cue_consensus_score': consensus,
        'cue_disagreement_score': disagreement,
        'accessibility_bonus': accessibility_bonus,
        'replay_bonus': replay_bonus,
        'reconsolidation_bonus': recon_bonus,
        'cue_consensus_bonus': consensus_bonus,
        'competitor_suppression_penalty': competitor_penalty,
        'ambiguity_penalty': ambiguity_penalty,
        'accessibility_score': float(state['accessibility_score']),
        'specificity_score': float(state['specificity_score']),
        'hubness_score': float(state['hubness_score']),
        'replay_priority': float(state['replay_priority']),
        'suppression_score': float(state['suppression_score']),
        'final_score_e32': final_score,
    })
    return out


def wrong_proto_map_from_negative_rows(negative_rows: list[dict[str, Any]]) -> dict[str, int]:
    mapping = {}
    for row in negative_rows:
        if str(row.get('control_name', '')) != 'wrong_old_prototype':
            continue
        proto = e31.si(row.get('test_old_prototype_id'), None)
        if proto is not None:
            mapping[str(row.get('event_id', ''))] = int(proto)
    return mapping


def classify_false_reason(event: dict[str, Any], target_row: dict[str, Any] | None, top1_row: dict[str, Any] | None, top_ids: list[int], candidate_pool_ids: set[int], cfg: dict[str, Any], top1_margin: float) -> str:
    target_bundle_id = event['target_bundle_id']
    if target_bundle_id is None:
        return 'metric_mismatch'
    if int(target_bundle_id) not in candidate_pool_ids:
        return 'target_not_in_candidate_pool'
    if target_bundle_id in top_ids[:5] and target_bundle_id not in top_ids[:3]:
        return 'target_in_top5_but_lost_top1'
    if target_bundle_id in top_ids[:3] and (len(top_ids) == 0 or int(top_ids[0]) != int(target_bundle_id)):
        return 'target_in_top3_but_lost_top1'
    if target_row is None or top1_row is None:
        return 'metric_mismatch'
    if float(top1_row['hubness_score']) > cfg['hubness_state_thresh'] and float(top1_row['cue_disagreement_score']) > cfg['generic_gap_thresh']:
        return 'hub_bundle_dominance'
    if float(top1_row['cue_disagreement_score']) > cfg['generic_gap_thresh']:
        return 'generic_content_dominance'
    if float(target_row['temporal_score']) < 0.55:
        return 'temporal_gap_mismatch'
    if float(target_row['disappearance_score']) < 0.60:
        return 'disappearance_signature_weak'
    if float(target_row['context_score']) < 0.60:
        return 'context_signature_collision'
    if float(target_row['provenance_score']) < 0.60:
        return 'provenance_too_weak'
    if float(target_row['accessibility_score']) + 0.10 < float(top1_row['accessibility_score']):
        return 'accessibility_too_low'
    if float(top1_row['accessibility_score']) > 0.80 and float(top1_row['hubness_score']) > cfg['hubness_state_thresh']:
        return 'wrong_bundle_accessibility_too_high'
    if cfg['competition'] and target_bundle_id in candidate_pool_ids and target_bundle_id not in top_ids[:5]:
        return 'competition_nms_removed_target'
    if top1_margin < cfg['low_margin_thresh']:
        return 'ambiguous_multi_valid_bundle'
    return 'ambiguous_multi_valid_bundle'

def evaluate_ablation(name: str, base_cfg: dict[str, Any], dyn_cfg: dict[str, Any], bundle_by_id: dict[int, dict[str, Any]], event_records: list[dict[str, Any]], proto_counter: Counter[int], track_counter: Counter[int], lineage_counter: Counter[int | None], wrong_proto_map: dict[str, int]):
    states = init_bundle_states(bundle_by_id, proto_counter, track_counter)
    retrieval_rows, taxonomy_rows, replay_rows, recon_rows, suppression_rows = [], [], [], [], []
    hist_topk = Counter()
    ordered_events = sorted(event_records, key=lambda r: (r['scenario_name'], int(r['frame_idx']), r['event_id']))
    for event in ordered_events:
        if int(event['proposal_detected']) != 1 or event['cue'] is None:
            retrieval_rows.append({'ablation_name': name, 'event_id': event['event_id'], 'scenario_name': event['scenario_name'], 'frame_idx': int(event['frame_idx']), 'proposal_detected': int(event['proposal_detected']), 'target_bundle_id': event['target_bundle_id'] or '', 'target_bundle_rank': '', 'target_bundle_retrieved_top1': 0, 'target_bundle_retrieved_top3': 0, 'target_bundle_retrieved_top5': 0, 'pattern_completion_success': 0, 'false_bundle_retrieval': 0, 'proto0_bundle_count_in_top5': 0, 'hub_bundle_count_in_top5': 0, 'top5_bundle_ids': '', 'top5_proto_ids': '', 'top1_bundle_id': '', 'top1_margin': '', 'strict_anchor_visible_top5': 0, 'target_lost_reason': 'proposal_missing_for_retrieval', 'alignment_classification': event['alignment_classification']})
            continue
        eligible = [bundle_by_id[b] for b in event['eligible_bundle_ids'] if b in bundle_by_id]
        if not eligible:
            retrieval_rows.append({'ablation_name': name, 'event_id': event['event_id'], 'scenario_name': event['scenario_name'], 'frame_idx': int(event['frame_idx']), 'proposal_detected': int(event['proposal_detected']), 'target_bundle_id': event['target_bundle_id'] or '', 'target_bundle_rank': '', 'target_bundle_retrieved_top1': 0, 'target_bundle_retrieved_top3': 0, 'target_bundle_retrieved_top5': 0, 'pattern_completion_success': 0, 'false_bundle_retrieval': 0, 'proto0_bundle_count_in_top5': 0, 'hub_bundle_count_in_top5': 0, 'top5_bundle_ids': '', 'top5_proto_ids': '', 'top1_bundle_id': '', 'top1_margin': '', 'strict_anchor_visible_top5': 0, 'target_lost_reason': 'target_not_in_candidate_pool', 'alignment_classification': event['alignment_classification']})
            continue
        stage1 = []
        for bundle in eligible:
            base = e31.score_candidate(event['cue'], bundle, base_cfg, proto_counter[int(bundle['primary_source_prototype_id'])], track_counter[int(bundle['primary_source_track_id'])], lineage_counter[bundle['primary_source_lineage_id']], hist_topk[int(bundle['bundle_id'])])
            stage1.append(score_candidate_e32(base, states[int(bundle['bundle_id'])], dyn_cfg))
        stage1.sort(key=lambda r: r['base_score'], reverse=True)
        candidate_pool = stage1[:base_cfg['candidate_pool_size']]
        candidate_pool_ids = {int(r['bundle_id']) for r in candidate_pool}
        reranked = sorted(candidate_pool, key=lambda r: r['final_score_e32'], reverse=True)
        final_topk = e31.diversify_candidates(reranked, base_cfg)
        top_ids = [int(r['bundle_id']) for r in final_topk]
        top1 = final_topk[0] if final_topk else None
        top2 = final_topk[1] if len(final_topk) > 1 else None
        top1_margin = 0.0 if top1 is None else (float(top1['final_score_e32']) - float(top2['final_score_e32']) if top2 is not None else float(top1['final_score_e32']))
        target_bundle_id = event['target_bundle_id']
        target_row = next((r for r in reranked if target_bundle_id is not None and int(r['bundle_id']) == int(target_bundle_id)), None)
        target_rank = next((i for i, r in enumerate(reranked, start=1) if target_bundle_id is not None and int(r['bundle_id']) == int(target_bundle_id)), None)
        top1_hit = int(target_bundle_id is not None and len(top_ids) > 0 and int(top_ids[0]) == int(target_bundle_id))
        top3_hit = int(target_bundle_id is not None and int(target_bundle_id) in set(top_ids[:3]))
        top5_hit = int(target_bundle_id is not None and int(target_bundle_id) in set(top_ids[:5]))
        success = int(top1_hit == 1 and target_row is not None and float(target_row['final_score_e32']) >= base_cfg['completion_threshold'])
        false_retrieval = int(len(top_ids) > 0 and top1_hit == 0)
        false_reason = classify_false_reason(event, target_row, top1, top_ids, candidate_pool_ids, dyn_cfg, top1_margin)
        for rank, row in enumerate(final_topk[:3], start=1):
            st = states[int(row['bundle_id'])]
            low_margin_score = max(0.0, 1.0 - min(top1_margin / dyn_cfg['low_margin_scale'], 1.0))
            cue_disagreement_score = float(row['cue_disagreement_score'])
            hub_conflict = min(float(st['hubness_score']) / 2.5, 1.0)
            rarity_score = float(st['specificity_score'])
            replay_priority = 0.25 * cue_disagreement_score + 0.20 * hub_conflict + 0.20 * low_margin_score + 0.15 * (1.0 - float(row['temporal_score'])) + 0.10 * rarity_score + 0.10 * (1.0 - float(row['disappearance_score']))
            old_replay = float(st['replay_priority'])
            st['replay_priority'] = max(old_replay * dyn_cfg['replay_decay'], replay_priority)
            if dyn_cfg['replay_enabled'] and rank <= dyn_cfg['replay_topk']:
                old_acc = float(st['accessibility_score'])
                st['accessibility_score'] = float(np.clip(old_acc + dyn_cfg['replay_accessibility_gain'] * st['replay_priority'] * float(row['cue_consensus_score']), 0.0, 1.0))
                st['accessibility_state'] = state_label(st)
            replay_rows.append({'ablation_name': name, 'event_id': event['event_id'], 'bundle_id': int(row['bundle_id']), 'memory_anchor_id': row['memory_anchor_id'], 'replay_priority': float(st['replay_priority']), 'priority_reason': 'cue_disagreement' if cue_disagreement_score >= hub_conflict else 'hubness_conflict', 'is_focus_event': int(event['event_id'] in e31.FOCUS_EVENT_IDS), 'is_false_retrieval_event': int(false_retrieval), 'is_namespace_shift_event': int(event['alignment_classification'] == 'runtime_namespace_shift'), 'target_rank_before_replay': '' if target_rank is None else int(target_rank), 'top1_margin': float(top1_margin), 'hubness_score': float(st['hubness_score']), 'cue_disagreement_score': float(cue_disagreement_score)})
        if top1 is not None:
            top1_state = states[int(top1['bundle_id'])]
            old_acc = float(top1_state['accessibility_score'])
            old_react = int(top1_state['reactivation_count'])
            if dyn_cfg['suppression_enabled'] and (float(top1['cue_disagreement_score']) > dyn_cfg['generic_gap_thresh'] or (float(top1_state['hubness_score']) > dyn_cfg['hubness_state_thresh'] and float(top1['cue_consensus_score']) < dyn_cfg['good_consensus'])) and top1_margin < dyn_cfg['suppression_margin_max']:
                old_supp = float(top1_state['suppression_score'])
                top1_state['suppression_score'] = float(np.clip(old_supp + dyn_cfg['suppression_up'] * (float(top1['cue_disagreement_score']) + min(float(top1_state['hubness_score']) / 2.5, 1.0)), 0.0, 1.0))
                top1_state['accessibility_score'] = float(np.clip(old_acc - dyn_cfg['accessibility_down'] * (float(top1['cue_disagreement_score']) + 0.2), 0.0, 1.0))
                top1_state['false_retrieval_count'] += 1
                top1_state['suppression_count'] += 1
                top1_state['accessibility_state'] = state_label(top1_state)
                suppression_rows.append({'ablation_name': name, 'event_id': event['event_id'], 'wrong_top1_bundle_id': int(top1['bundle_id']), 'wrong_top1_proto': int(top1['primary_source_prototype_id']), 'wrong_top1_anchor': top1['memory_anchor_id'], 'suppression_applied': 1, 'suppression_reason': false_reason, 'old_accessibility': old_acc, 'new_accessibility': float(top1_state['accessibility_score']), 'old_final_score': float(top1['final_score_e32']), 'new_final_score': float(top1['final_score_e32'] - dyn_cfg['suppression_penalty'] * top1_state['suppression_score']), 'affected_future_events': int(top1_state['suppression_count'])})
            if dyn_cfg['accessibility_enabled'] and float(top1['cue_consensus_score']) >= dyn_cfg['good_consensus']:
                top1_state['accessibility_score'] = float(np.clip(top1_state['accessibility_score'] + dyn_cfg['accessibility_up'] * ((float(top1['cue_consensus_score']) + float(top1_state['specificity_score'])) * 0.5), 0.0, 1.0))
                top1_state['reactivation_count'] += 1
                top1_state['cue_consensus_score'] = float(top1['cue_consensus_score'])
                top1_state['accessibility_state'] = state_label(top1_state)
            if dyn_cfg['reconsolidation_enabled'] and float(top1['cue_consensus_score']) >= dyn_cfg['good_consensus']:
                old_refs = sorted(int(v) for v in top1_state['runtime_lineage_refs']) if top1_state['runtime_lineage_refs'] else []
                if top1['canonical_lineage_id'] is not None:
                    top1_state['runtime_lineage_refs'].add(int(top1['canonical_lineage_id']))
                top1_state['reconsolidation_count'] += 1
                top1_state['accessibility_state'] = state_label(top1_state)
                recon_rows.append({'ablation_name': name, 'frame_idx': int(event['frame_idx']), 'event_id': event['event_id'], 'bundle_id': int(top1['bundle_id']), 'memory_anchor_id': top1['memory_anchor_id'], 'old_runtime_lineage_refs': '|'.join(str(v) for v in old_refs), 'new_runtime_lineage_refs': '|'.join(str(v) for v in sorted(int(v) for v in top1_state['runtime_lineage_refs'])), 'old_accessibility_score': old_acc, 'new_accessibility_score': float(top1_state['accessibility_score']), 'old_reactivation_count': old_react, 'new_reactivation_count': int(top1_state['reactivation_count']), 'reconsolidation_reason': 'high_cue_consensus', 'head_updated': 0, 'attach_triggered': 0, 'promotion_triggered': 0})
        hub_count = sum(1 for r in final_topk[:5] if float(states[int(r['bundle_id'])]['hubness_score']) > dyn_cfg['hubness_state_thresh'])
        proto0_count = sum(1 for r in final_topk[:5] if int(r['primary_source_prototype_id']) == 0)
        retrieval_rows.append({'ablation_name': name, 'event_id': event['event_id'], 'scenario_name': event['scenario_name'], 'frame_idx': int(event['frame_idx']), 'proposal_detected': int(event['proposal_detected']), 'target_bundle_id': '' if target_bundle_id is None else int(target_bundle_id), 'target_bundle_rank': '' if target_rank is None else int(target_rank), 'target_bundle_retrieved_top1': top1_hit, 'target_bundle_retrieved_top3': top3_hit, 'target_bundle_retrieved_top5': top5_hit, 'pattern_completion_success': success, 'false_bundle_retrieval': false_retrieval, 'proto0_bundle_count_in_top5': proto0_count, 'hub_bundle_count_in_top5': hub_count, 'top5_bundle_ids': '|'.join(str(v) for v in top_ids[:5]), 'top5_proto_ids': '|'.join(str(int(r['primary_source_prototype_id'])) for r in final_topk[:5]), 'top1_bundle_id': '' if top1 is None else int(top1['bundle_id']), 'top1_margin': float(top1_margin), 'strict_anchor_visible_top5': int(top5_hit == 1), 'target_lost_reason': false_reason, 'alignment_classification': event['alignment_classification']})
        wrong_top1 = None if top1 is None else top1
        taxonomy_rows.append({'ablation_name': name, 'event_id': event['event_id'], 'target_bundle_id': '' if target_bundle_id is None else int(target_bundle_id), 'target_rank': '' if target_rank is None else int(target_rank), 'target_in_candidate_pool': int(target_bundle_id is not None and int(target_bundle_id) in candidate_pool_ids), 'target_in_top5': top5_hit, 'target_in_top3': top3_hit, 'target_in_top1': top1_hit, 'wrong_top1_bundle_id': '' if wrong_top1 is None else int(wrong_top1['bundle_id']), 'wrong_top1_source_prototype': '' if wrong_top1 is None else int(wrong_top1['primary_source_prototype_id']), 'wrong_top1_memory_anchor': '' if wrong_top1 is None else wrong_top1['memory_anchor_id'], 'wrong_top1_canonical_lineage': '' if wrong_top1 is None else wrong_top1['canonical_lineage_id'], 'target_score': '' if target_row is None else float(target_row['final_score_e32']), 'wrong_top1_score': '' if wrong_top1 is None else float(wrong_top1['final_score_e32']), 'score_margin': '' if wrong_top1 is None or target_row is None else float(wrong_top1['final_score_e32'] - target_row['final_score_e32']), 'target_content_score': '' if target_row is None else float(target_row['content_score']), 'target_support_score': '' if target_row is None else float(target_row['support_score']), 'target_motion_score': '' if target_row is None else float(target_row['motion_score']), 'target_context_score': '' if target_row is None else float(target_row['context_score']), 'target_temporal_score': '' if target_row is None else float(target_row['temporal_score']), 'target_disappearance_score': '' if target_row is None else float(target_row['disappearance_score']), 'target_provenance_score': '' if target_row is None else float(target_row['provenance_score']), 'target_accessibility_score': '' if target_row is None else float(states[int(target_row['bundle_id'])]['accessibility_score']), 'wrong_content_score': '' if wrong_top1 is None else float(wrong_top1['content_score']), 'wrong_support_score': '' if wrong_top1 is None else float(wrong_top1['support_score']), 'wrong_motion_score': '' if wrong_top1 is None else float(wrong_top1['motion_score']), 'wrong_context_score': '' if wrong_top1 is None else float(wrong_top1['context_score']), 'wrong_temporal_score': '' if wrong_top1 is None else float(wrong_top1['temporal_score']), 'wrong_disappearance_score': '' if wrong_top1 is None else float(wrong_top1['disappearance_score']), 'wrong_provenance_score': '' if wrong_top1 is None else float(wrong_top1['provenance_score']), 'wrong_accessibility_score': '' if wrong_top1 is None else float(states[int(wrong_top1['bundle_id'])]['accessibility_score']), 'false_reason': false_reason})
    proposal_rows = [r for r in retrieval_rows if int(r['proposal_detected']) == 1]
    top5_ids_by_event = {r['event_id']: [int(v) for v in str(r['top5_bundle_ids']).split('|') if v not in ('', None)] for r in proposal_rows}
    top5_protos_by_event = {r['event_id']: [int(v) for v in str(r['top5_proto_ids']).split('|') if v not in ('', None)] for r in proposal_rows}
    event_ids = [r['event_id'] for r in proposal_rows if r['target_bundle_id'] != '']
    target_ids = [int(r['target_bundle_id']) for r in proposal_rows if r['target_bundle_id'] != '']
    shuffled_hits = 0
    if target_ids:
        shuffled = target_ids[1:] + target_ids[:1]
        for eid, sid in zip(event_ids, shuffled):
            if sid in top5_ids_by_event.get(eid, []):
                shuffled_hits += 1
    wrong_old_visible = 0
    for eid, wrong_proto in wrong_proto_map.items():
        if wrong_proto in top5_protos_by_event.get(eid, []):
            wrong_old_visible += 1
    hub_share = 0.0 if not proposal_rows else float(np.mean([int(r['hub_bundle_count_in_top5']) / 5.0 for r in proposal_rows]))
    summary = {'ablation_name': name, 'global_top1': 0.0 if not proposal_rows else float(np.mean([int(r['target_bundle_retrieved_top1']) for r in proposal_rows])), 'global_top3': 0.0 if not proposal_rows else float(np.mean([int(r['target_bundle_retrieved_top3']) for r in proposal_rows])), 'global_top5': 0.0 if not proposal_rows else float(np.mean([int(r['target_bundle_retrieved_top5']) for r in proposal_rows])), 'global_pattern_completion_success': 0.0 if not proposal_rows else float(np.mean([int(r['pattern_completion_success']) for r in proposal_rows])), 'false_bundle_retrieval_rate': 0.0 if not proposal_rows else float(np.mean([int(r['false_bundle_retrieval']) for r in proposal_rows])), 'focus_top1_count': int(sum(int(r['target_bundle_retrieved_top1']) for r in proposal_rows if r['event_id'] in e31.FOCUS_EVENT_IDS)), 'focus_top3_count': int(sum(int(r['target_bundle_retrieved_top3']) for r in proposal_rows if r['event_id'] in e31.FOCUS_EVENT_IDS)), 'focus_top5_count': int(sum(int(r['target_bundle_retrieved_top5']) for r in proposal_rows if r['event_id'] in e31.FOCUS_EVENT_IDS)), 'focus_success_count': int(sum(int(r['pattern_completion_success']) for r in proposal_rows if r['event_id'] in e31.FOCUS_EVENT_IDS)), 'proto0_top5_share': 0.0 if not proposal_rows else float(np.mean([int(r['proto0_bundle_count_in_top5']) / 5.0 for r in proposal_rows])), 'hub_bundle_top5_share': hub_share, 'mean_target_rank': 999.0 if not proposal_rows else float(np.mean([int(r['target_bundle_rank']) if r['target_bundle_rank'] != '' else 999 for r in proposal_rows])), 'mean_wrong_top1_margin': 0.0 if not proposal_rows else float(np.mean([float(r['top1_margin']) if r['top1_margin'] != '' else 0.0 for r in proposal_rows])), 'strict_anchor_real_svr': 0.0 if not proposal_rows else float(np.mean([int(r['strict_anchor_visible_top5']) for r in proposal_rows])), 'strict_anchor_shuffled_svr': 0.0 if not event_ids else float(shuffled_hits / len(event_ids)), 'wrong_old_prototype_visible_count': int(wrong_old_visible)}
    return {'summary': summary, 'retrieval_rows': retrieval_rows, 'taxonomy_rows': taxonomy_rows, 'replay_rows': replay_rows, 'recon_rows': recon_rows, 'suppression_rows': suppression_rows}

def render_report(summary: dict[str, Any]) -> str:
    lines = ['# Stage E3.2 Report', '', '## Best Ablation', '']
    for k, v in summary['best_ablation'].items():
        lines.append(f'- `{k} = {v}`')
    lines += ['', '## Focus Events', '']
    for row in summary['focus_events']:
        lines += [f"### {row['event_id']}", '', f"- `target_bundle_retrieved_top1 = {row['target_bundle_retrieved_top1']}`", f"- `target_bundle_retrieved_top3 = {row['target_bundle_retrieved_top3']}`", f"- `target_bundle_retrieved_top5 = {row['target_bundle_retrieved_top5']}`", f"- `pattern_completion_success = {row['pattern_completion_success']}`", f"- `target_lost_reason = {row['target_lost_reason']}`", '']
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_by_id, event_records, _ = e31.collect_runtime_data(args.config, args.event_audit, args.cross_run_alignment, args.seed)
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    negative_rows = e31.load_negative_controls(args.e2c_negative_events)
    wrong_proto_map = wrong_proto_map_from_negative_rows(negative_rows)

    a0_base = {'candidate_pool_size': 35, 'final_topk': 5, 'competition': True, 'max_per_proto': 1, 'max_per_anchor': 1, 'max_per_lineage': 3, 'temporal_decay': 120.0, 'completion_threshold': 0.50, 'rarity_bonus': 0.14, 'hub_alpha': 0.03, 'hub_beta': 0.015, 'hub_gamma': 0.015, 'lineage_hub_gamma': 0.005, 'weights': {'content': 0.14, 'support': 0.14, 'motion': 0.10, 'context': 0.17, 'temporal': 0.15, 'disappearance': 0.13, 'provenance': 0.09, 'separation': 0.08}}
    dyn_baseline = {'acc_bonus': 0.0, 'replay_bonus': 0.0, 'recon_bonus': 0.0, 'consensus_bonus': 0.0, 'suppression_penalty': 0.0, 'ambiguity_penalty': 0.0, 'replay_enabled': False, 'reconsolidation_enabled': False, 'suppression_enabled': False, 'accessibility_enabled': False, 'replay_topk': 3, 'replay_decay': 0.85, 'replay_accessibility_gain': 0.0, 'good_consensus': 0.70, 'good_margin': 0.04, 'suppression_margin_max': 0.12, 'generic_gap_thresh': 0.18, 'hubness_state_thresh': 1.0, 'accessibility_up': 0.0, 'accessibility_down': 0.0, 'suppression_up': 0.0, 'low_margin_scale': 0.15, 'low_margin_thresh': 0.05}
    ablations = {
        'A0_E31_combined_baseline': (a0_base, dyn_baseline),
        'A1_accessibility_only': (a0_base, {**dyn_baseline, 'acc_bonus': 0.10, 'accessibility_enabled': True, 'accessibility_up': 0.06}),
        'A2_replay_priority_only': (a0_base, {**dyn_baseline, 'replay_bonus': 0.10, 'replay_enabled': True, 'replay_accessibility_gain': 0.03}),
        'A3_reconsolidation_only': (a0_base, {**dyn_baseline, 'recon_bonus': 0.08, 'reconsolidation_enabled': True, 'accessibility_enabled': True, 'accessibility_up': 0.03}),
        'A4_competitor_suppression_only': (a0_base, {**dyn_baseline, 'suppression_penalty': 0.12, 'suppression_enabled': True, 'accessibility_down': 0.08, 'suppression_up': 0.18}),
        'A5_cue_consensus_geomean': (a0_base, {**dyn_baseline, 'consensus_bonus': 0.12, 'ambiguity_penalty': 0.10}),
        'A6_accessibility_plus_suppression': (a0_base, {**dyn_baseline, 'acc_bonus': 0.10, 'accessibility_enabled': True, 'accessibility_up': 0.05, 'suppression_penalty': 0.12, 'suppression_enabled': True, 'accessibility_down': 0.08, 'suppression_up': 0.18, 'consensus_bonus': 0.08, 'ambiguity_penalty': 0.08}),
        'A7_full_E32': (a0_base, {**dyn_baseline, 'acc_bonus': 0.10, 'replay_bonus': 0.10, 'recon_bonus': 0.08, 'consensus_bonus': 0.12, 'suppression_penalty': 0.12, 'ambiguity_penalty': 0.10, 'replay_enabled': True, 'reconsolidation_enabled': True, 'suppression_enabled': True, 'accessibility_enabled': True, 'replay_accessibility_gain': 0.03, 'accessibility_up': 0.05, 'accessibility_down': 0.08, 'suppression_up': 0.18})
    }

    results = {}
    ablation_rows, all_replay, all_recon, all_supp, all_tax = [], [], [], [], []
    for name, (base_cfg, dyn_cfg) in ablations.items():
        result = evaluate_ablation(name, base_cfg, dyn_cfg, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter, wrong_proto_map)
        results[name] = result
        ablation_rows.append(result['summary'])
        all_replay.extend(result['replay_rows'])
        all_recon.extend(result['recon_rows'])
        all_supp.extend(result['suppression_rows'])
        all_tax.extend(result['taxonomy_rows'])

    baseline_rows = {r['event_id']: r for r in results['A0_E31_combined_baseline']['retrieval_rows']}
    full_rows = {r['event_id']: r for r in results['A7_full_E32']['retrieval_rows']}
    delta_rows, regression_rows = [], []
    improved_count = unchanged_failure_count = 0
    for event_id, after in full_rows.items():
        before = baseline_rows.get(event_id)
        if before is None:
            continue
        bsucc = int(before['pattern_completion_success'])
        asucc = int(after['pattern_completion_success'])
        brank = 999 if before['target_bundle_rank'] == '' else int(before['target_bundle_rank'])
        arank = 999 if after['target_bundle_rank'] == '' else int(after['target_bundle_rank'])
        if asucc > bsucc or (asucc == bsucc and arank < brank):
            delta = 'improved'
            improved_count += 1
        elif asucc == bsucc == 1:
            delta = 'unchanged_success'
        elif asucc < bsucc or (asucc == bsucc and arank > brank):
            delta = 'regressed'
            regression_rows.append({'event_id': event_id, 'sequence_id': after['scenario_name'], 'baseline_target_rank': brank, 'e32_target_rank': arank, 'baseline_success': bsucc, 'e32_success': asucc, 'baseline_false_reason': before['target_lost_reason'], 'e32_false_reason': after['target_lost_reason']})
        else:
            delta = 'unchanged_failure'
            unchanged_failure_count += 1
        delta_rows.append({'event_id': event_id, 'sequence_id': after['scenario_name'], 'proposal_detected': int(after['proposal_detected']), 'is_focus_event': int(event_id in e31.FOCUS_EVENT_IDS), 'baseline_target_rank': brank if brank != 999 else '', 'e32_target_rank': arank if arank != 999 else '', 'baseline_top1_bundle_id': before['top1_bundle_id'], 'e32_top1_bundle_id': after['top1_bundle_id'], 'baseline_target_in_top3': before['target_bundle_retrieved_top3'], 'e32_target_in_top3': after['target_bundle_retrieved_top3'], 'baseline_target_in_top5': before['target_bundle_retrieved_top5'], 'e32_target_in_top5': after['target_bundle_retrieved_top5'], 'baseline_success': bsucc, 'e32_success': asucc, 'delta_class': delta, 'baseline_false_reason': before['target_lost_reason'], 'e32_false_reason': after['target_lost_reason']})

    best = max(ablation_rows, key=lambda r: (r['focus_success_count'], r['global_top1'], r['global_top3'], -r['false_bundle_retrieval_rate']))
    best_name = str(best['ablation_name'])
    focus_rows = [r for r in results[best_name]['retrieval_rows'] if r['event_id'] in e31.FOCUS_EVENT_IDS]
    summary = {'scope': 'track_a_bridge_and_track_c_long_horizon', 'best_ablation': {**best, 'regression_event_count': len(regression_rows), 'improved_event_count': improved_count, 'unchanged_failure_count': unchanged_failure_count}, 'focus_events': focus_rows, 'negative_control_reference': {'real_anchor_svr': 1.0, 'shuffled_anchor_svr': 0.875, 'wrong_old_prototype_visible_count': 0}}

    e31.write_csv(output_dir / f'stage_E32_event_delta_audit_{args.artifact_version}.csv', delta_rows)
    e31.write_csv(output_dir / f'stage_E32_false_retrieval_taxonomy_{args.artifact_version}.csv', all_tax)
    e31.write_csv(output_dir / f'stage_E32_regression_events_{args.artifact_version}.csv', regression_rows)
    e31.write_csv(output_dir / f'stage_E32_replay_queue_{args.artifact_version}.csv', all_replay)
    e31.write_csv(output_dir / f'stage_E32_reconsolidation_trace_{args.artifact_version}.csv', all_recon)
    e31.write_csv(output_dir / f'stage_E32_competitor_suppression_trace_{args.artifact_version}.csv', all_supp)
    e31.write_csv(output_dir / f'stage_E32_ablation_summary_{args.artifact_version}.csv', ablation_rows)
    (output_dir / f'stage_E32_summary_{args.artifact_version}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / f'stage_E32_report_{args.artifact_version}.md').write_text(render_report(summary), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
