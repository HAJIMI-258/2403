from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download selected LaSOT category zips from HuggingFace.")
    p.add_argument("--repo-id", default="l-lt/LaSOT")
    p.add_argument("--categories", required=True, help="Comma-separated category names, e.g. airplane,basketball")
    p.add_argument("--zip-dir", default="data/external/lasot_zips")
    p.add_argument("--extract-dir", default="data/external/lasot")
    p.add_argument("--execute", action="store_true", help="Actually download and extract. Default is dry-run.")
    p.add_argument("--keep-zip", action="store_true")
    p.add_argument("--manifest", default="data/external/lasot_download_plan.json")
    return p.parse_args()


def repo_zip_sizes(repo_id: str) -> dict[str, int]:
    api = HfApi()
    sizes: dict[str, int] = {}
    for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=False, expand=True):
        path = getattr(item, "path", "")
        size = getattr(item, "size", None)
        if path.endswith(".zip") and size is not None:
            sizes[path] = int(size)
    return sizes


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def main() -> None:
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    zip_dir = Path(args.zip_dir)
    extract_dir = Path(args.extract_dir)
    manifest_path = Path(args.manifest)
    zip_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    sizes = repo_zip_sizes(args.repo_id)
    rows: list[dict[str, Any]] = []
    for cat in categories:
        filename = f"{cat}.zip"
        size = sizes.get(filename)
        row: dict[str, Any] = {
            "repo_id": args.repo_id,
            "category": cat,
            "filename": filename,
            "size_bytes": size,
            "size_gb": None if size is None else size / (1024 ** 3),
            "downloaded": 0,
            "extracted": 0,
            "local_zip": str(zip_dir / filename),
            "extract_dir": str(extract_dir),
            "error": "",
        }
        if size is None:
            row["error"] = "category_zip_not_found"
            rows.append(row)
            continue
        if args.execute:
            try:
                downloaded = hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=zip_dir,
                )
                row["downloaded"] = 1
                row["local_zip"] = downloaded
                extract_zip(Path(downloaded), extract_dir)
                row["extracted"] = 1
                if not args.keep_zip:
                    Path(downloaded).unlink(missing_ok=True)
            except Exception as exc:
                row["error"] = str(exc)
        rows.append(row)

    manifest_path.write_text(json.dumps({
        "execute": bool(args.execute),
        "repo_id": args.repo_id,
        "categories": categories,
        "total_size_gb": sum((r["size_bytes"] or 0) for r in rows) / (1024 ** 3),
        "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
