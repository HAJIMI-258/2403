from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lagot_adapter import LaGOTAdapter
from experiments.ext1_utils import write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-4 LaGOT/LaSOT full-pixel readiness gate.")
    p.add_argument("--lagot-root", default="data/external/lagot_annotations")
    p.add_argument("--lasot-root", default="data/external/lasot")
    p.add_argument("--hf-repo", default="l-lt/LaSOT")
    p.add_argument("--output-dir", default="results/ext4")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def lagot_to_lasot_sequence(sequence_id: str) -> str:
    s = sequence_id
    if s.startswith("lagot_"):
        s = s[len("lagot_"):]
    head, sep, tail = s.rpartition("_")
    if sep and tail.isdigit():
        return head
    return s


def category_from_lasot_sequence(sequence_id: str) -> str:
    return sequence_id.split("-")[0]


def find_lasot_sequence_dir(root: Path, sequence_id: str) -> Path | None:
    direct = root / sequence_id
    if (direct / "img").exists():
        return direct
    category = category_from_lasot_sequence(sequence_id)
    nested = root / category / sequence_id
    if (nested / "img").exists():
        return nested
    if not root.exists():
        return None
    for img_dir in root.rglob("img"):
        if img_dir.is_dir() and img_dir.parent.name == sequence_id:
            return img_dir.parent
    return None


def count_images(seq_dir: Path | None) -> int:
    if seq_dir is None:
        return 0
    return len(list((seq_dir / "img").glob("*.jpg")))


def hf_zip_manifest(repo_id: str) -> dict[str, dict[str, Any]]:
    try:
        from huggingface_hub import HfApi
    except Exception:
        return {}
    try:
        api = HfApi()
        out: dict[str, dict[str, Any]] = {}
        for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=False, expand=True):
            path = getattr(item, "path", "")
            size = getattr(item, "size", None)
            if path.endswith(".zip") and size is not None:
                category = path[:-4]
                out[category] = {
                    "hf_repo": repo_id,
                    "hf_zip_name": path,
                    "hf_zip_size_bytes": int(size),
                    "hf_zip_size_gb": int(size) / (1024 ** 3),
                }
        return out
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lagot = LaGOTAdapter(args.lagot_root)
    lasot_root = Path(args.lasot_root)
    hf_manifest = hf_zip_manifest(args.hf_repo)

    linkage_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    category_counter: Counter[str] = Counter()
    category_events: Counter[str] = Counter()
    ready_events = 0
    total_events = 0
    ready_sequences: set[str] = set()
    linked_sequences: set[str] = set()

    for lagot_seq in lagot.iter_sequences():
        lasot_seq = lagot_to_lasot_sequence(lagot_seq)
        category = category_from_lasot_sequence(lasot_seq)
        linked_sequences.add(lasot_seq)
        category_counter[category] += 1
        events = lagot.derive_events(lagot_seq)
        event_count = len([ev for ev in events if ev.gap_length >= 3])
        total_events += event_count
        category_events[category] += event_count
        seqinfo = lagot._read_seqinfo(lagot_seq)
        seq_dir = find_lasot_sequence_dir(lasot_root, lasot_seq)
        image_count = count_images(seq_dir)
        gt_found = int(seq_dir is not None and (seq_dir / "groundtruth.txt").exists())
        expected_frames = int(seqinfo.get("seq_length", 0) or 0)
        frame_coverage = image_count / max(expected_frames, 1)
        pixel_ready = int(seq_dir is not None and image_count > 0 and frame_coverage >= 0.95)
        if pixel_ready:
            ready_sequences.add(lasot_seq)
            ready_events += event_count
        row = {
            "lagot_sequence_id": lagot_seq,
            "lasot_sequence_id": lasot_seq,
            "category": category,
            "event_count": event_count,
            "expected_frames": expected_frames,
            "lasot_sequence_dir": "" if seq_dir is None else str(seq_dir),
            "found_image_count": image_count,
            "groundtruth_found": gt_found,
            "frame_coverage": frame_coverage,
            "pixel_ready": pixel_ready,
        }
        linkage_rows.append(row)
        if not pixel_ready:
            missing_rows.append(row)

    manifest_rows: list[dict[str, Any]] = []
    for category, sequence_count in sorted(category_counter.items()):
        hf = hf_manifest.get(category, {})
        size_gb = float(hf.get("hf_zip_size_gb", 0.0) or 0.0)
        events = category_events[category]
        manifest_rows.append({
            "category": category,
            "lagot_sequence_count": sequence_count,
            "lagot_event_count": events,
            "local_pixel_ready_sequences": sum(
                1 for r in linkage_rows if r["category"] == category and int(r["pixel_ready"]) == 1
            ),
            "hf_repo": hf.get("hf_repo", args.hf_repo),
            "hf_zip_name": hf.get("hf_zip_name", f"{category}.zip"),
            "hf_zip_size_bytes": hf.get("hf_zip_size_bytes", ""),
            "hf_zip_size_gb": size_gb if size_gb else "",
            "events_per_gb": (events / size_gb) if size_gb > 0 else "",
            "download_command_dry_run": (
                f"python scripts/download_lasot_hf_categories.py --categories {category}"
            ),
            "download_command_execute": (
                f"python scripts/download_lasot_hf_categories.py --categories {category} --execute"
            ),
        })

    manifest_rows.sort(key=lambda r: (float(r["events_per_gb"] or 0), int(r["lagot_event_count"])), reverse=True)
    for idx, row in enumerate(manifest_rows, start=1):
        row["recommended_priority"] = idx

    top_categories = [r["category"] for r in manifest_rows[:5]]
    full_pixel_ready = int(ready_events >= 10)
    hf_available = int(bool(hf_manifest))

    compact = {
        "stage": "EXT-4",
        "purpose": "LaGOT-to-LaSOT full-pixel readiness and download manifest",
        "lagot_sequences": len(linkage_rows),
        "unique_lasot_sequences": len(linked_sequences),
        "total_lagot_reentry_events": total_events,
        "lasot_pixel_ready_sequences": len(ready_sequences),
        "pixel_ready_events": ready_events,
        "external_full_pixel_ready": full_pixel_ready,
        "hf_manifest_available": hf_available,
        "hf_repo": args.hf_repo,
        "download_required": int(not full_pixel_ready),
        "large_download_requires_user_confirmation": 1,
        "recommended_first_categories": top_categories,
        "next_recommendation": (
            "download selected LaSOT category zips with scripts/download_lasot_hf_categories.py "
            "before running full-pixel appearance validation"
            if not full_pixel_ready
            else "run EXT-4 full-pixel appearance validation on pixel-ready sequences"
        ),
    }

    report = f"""# EXT-4 Full-Pixel Readiness

## Decision

Full-pixel validation is not ready unless `external_full_pixel_ready = 1`.

Current result:

- LaGOT sequences linked: `{len(linkage_rows)}`
- Unique LaSOT sequences needed: `{len(linked_sequences)}`
- LaGOT re-entry events: `{total_events}`
- Pixel-ready LaSOT sequences: `{len(ready_sequences)}`
- Pixel-ready events: `{ready_events}`
- HuggingFace LaSOT manifest available: `{hf_available}`

## Download Policy

Do not download full LaSOT automatically. The HuggingFace category zips are multi-GB files.

Use the generated manifest to select categories. Suggested first categories:

`{", ".join(top_categories)}`

Dry-run example:

```powershell
python scripts/download_lasot_hf_categories.py --categories {top_categories[0] if top_categories else "airplane"}
```

Execute example:

```powershell
python scripts/download_lasot_hf_categories.py --categories {top_categories[0] if top_categories else "airplane"} --execute
```

## Next Step

{compact["next_recommendation"]}
"""

    write_csv(out_dir / "stage_EXT4_lagot_lasot_sequence_linkage_v1.csv", linkage_rows)
    write_csv(out_dir / "stage_EXT4_missing_pixel_sequences_v1.csv", missing_rows)
    write_csv(out_dir / "stage_EXT4_lasot_download_manifest_v1.csv", manifest_rows)
    write_json(out_dir / "stage_EXT4_download_manifest_compact_v1.json", {
        "hf_repo": args.hf_repo,
        "recommended_first_categories": top_categories,
        "top_categories": [
            {
                "category": row["category"],
                "lagot_event_count": row["lagot_event_count"],
                "hf_zip_size_gb": row["hf_zip_size_gb"],
                "events_per_gb": row["events_per_gb"],
                "download_command_dry_run": row["download_command_dry_run"],
                "download_command_execute": row["download_command_execute"],
            }
            for row in manifest_rows[:10]
        ],
    })
    write_json(out_dir / "stage_EXT4_compact_for_gpt_v1.json", compact)
    (out_dir / "stage_EXT4_report_v1.md").write_text(report, encoding="utf-8")

    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
