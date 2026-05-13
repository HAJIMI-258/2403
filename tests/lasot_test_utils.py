from __future__ import annotations

import unittest
from pathlib import Path


def create_pixel_mini_lasot(root: Path) -> Path:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        raise unittest.SkipTest("PIL/Pillow unavailable") from exc

    seq_dir = root / "bicycle" / "bicycle-1"
    img_dir = seq_dir / "img"
    img_dir.mkdir(parents=True)
    frame_count = 25
    for idx in range(frame_count):
        image = Image.new("RGB", (64, 64), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        if idx <= 8 or idx >= 17:
            x = 10 + (idx % 4)
            draw.rectangle((x, 20, x + 15, 35), fill=(255, 255, 255))
        image.save(img_dir / f"{idx + 1:08d}.jpg")
    gt_lines = []
    for idx in range(frame_count):
        x = 10 + (idx % 4)
        gt_lines.append(f"{x},20,16,16")
    (seq_dir / "groundtruth.txt").write_text("\n".join(gt_lines), encoding="utf-8")
    occ = [0] * 9 + [1] * 8 + [0] * 8
    (seq_dir / "full_occlusion.txt").write_text(",".join(str(v) for v in occ), encoding="utf-8")
    (seq_dir / "out_of_view.txt").write_text(",".join("0" for _ in range(frame_count)), encoding="utf-8")
    return root
