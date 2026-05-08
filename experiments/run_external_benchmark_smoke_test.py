from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lasot_adapter import LaSOTAdapter
from datasets.external.lagot_adapter import LaGOTAdapter
from datasets.external.lvos_adapter import LVOSAdapter
from datasets.external.tao_adapter import TAOAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test external video-memory adapters.")
    p.add_argument("--dataset", default="lvos_hf_sample", choices=["lvos_hf_sample", "lagot_annotations", "lvos", "lasot", "tao"])
    p.add_argument("--root", default="")
    p.add_argument("--output-dir", default="results/external_smoke")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--frame-limit", type=int, default=20)
    return p.parse_args()


def adapter_for(name: str, root: str):
    if name == "lvos_hf_sample":
        return LVOSAdapter(root or "data/external/hf_lvosv1_sample")
    if name == "lagot_annotations":
        return LaGOTAdapter(root or "data/external/lagot_annotations")
    if name == "lvos":
        return LVOSAdapter(root or "data/external/lvos")
    if name == "lasot":
        return LaSOTAdapter(root or "data/external/lasot")
    if name == "tao":
        return TAOAdapter(root or "data/external/tao")
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter = adapter_for(args.dataset, args.root)
    missing_files: list[str] = []
    rows = []
    try:
        sequences = list(adapter.iter_sequences())
        for seq in sequences[:3]:
            frames = list(adapter.iter_frames(seq, limit=args.frame_limit))
            events = adapter.derive_events(seq)
            rows.append({
                "sequence_id": seq,
                "num_frames_loaded": len(frames),
                "num_objects": len({iid for fr in frames for iid in fr.instance_ids}),
                "num_reentry_events": len(events),
                "num_occlusion_events": sum(1 for ev in events if ev.gap_length > 0),
            })
    except Exception as exc:
        sequences = []
        rows = []
        missing_files.append(str(exc))
    compact = {
        "dataset_name": args.dataset,
        "num_sequences_loaded": len(rows),
        "num_frames_loaded": sum(r["num_frames_loaded"] for r in rows),
        "num_objects": sum(r["num_objects"] for r in rows),
        "num_reentry_events": sum(r["num_reentry_events"] for r in rows),
        "num_occlusion_events": sum(r["num_occlusion_events"] for r in rows),
        "adapter_passed": int(len(rows) > 0 and not missing_files),
        "missing_files": missing_files,
    }
    (out / f"stage_EXTERNAL_SMOKE_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
