from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


FRAME_RE = re.compile(r"frame(?P<frame>\d+)")
MAKESENSE_RE = re.compile(
    r"^(?P<split>\d+)_rec_(?P<rec>\d+)_probe(?P<probe>\d+)_frame(?P<frame>\d+)"
)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, data: dict) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_frame_id(path: str | Path) -> int | None:
    match = FRAME_RE.search(Path(path).stem)
    if not match:
        return None
    return int(match.group("frame"))


def parse_makesense_image_name(image_name: str) -> dict[str, int | str]:
    match = MAKESENSE_RE.match(image_name)
    if not match:
        raise ValueError(f"Unexpected makesense image name: {image_name}")
    split = match.group("split")
    rec_num = int(match.group("rec"))
    probe = int(match.group("probe"))
    frame_id = int(match.group("frame"))
    return {
        "split": split,
        "record_id": f"rec_{rec_num:05d}",
        "record_num": rec_num,
        "probe": probe,
        "frame_id": frame_id,
        "contact_frame_from_name": frame_id + probe,
    }


def bbox_center(row: dict[str, str]) -> tuple[float, float]:
    x = float(row["bbox_x"])
    y = float(row["bbox_y"])
    w = float(row["bbox_width"])
    h = float(row["bbox_height"])
    return x + w / 2.0, y + h / 2.0


def normalized_vector(dx: float, dy: float) -> tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm <= 1e-8:
        return 0.0, 0.0
    return dx / norm, dy / norm


def list_image_files(directory: str | Path, exts: list[str]) -> list[Path]:
    root = Path(directory)
    allowed = {ext.lower() for ext in exts}
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in allowed)


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, radius: int = 5) -> None:
    draw.line((x - radius, y, x + radius, y), fill=color, width=2)
    draw.line((x, y - radius, x, y + radius), fill=color, width=2)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    width: int = 2,
) -> None:
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    head = 9
    for offset in (2.6, -2.6):
        x2 = x1 - head * math.cos(angle + offset)
        y2 = y1 - head * math.sin(angle + offset)
        draw.line((x1, y1, x2, y2), fill=color, width=width)


def save_label_overlay(
    image_path: str | Path,
    output_path: str | Path,
    tip: tuple[float, float],
    base: tuple[float, float],
    target: tuple[float, float] | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw_cross(draw, tip[0], tip[1], "red", 6)
    draw_cross(draw, base[0], base[1], "cyan", 6)
    draw_arrow(draw, base[0], base[1], tip[0], tip[1], "yellow", 3)
    if target is not None:
        draw_cross(draw, target[0], target[1], "lime", 8)
        draw.line((tip[0], tip[1], target[0], target[1]), fill="lime", width=2)
    out = Path(output_path)
    ensure_dir(out.parent)
    image.save(out)
