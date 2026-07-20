from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import load_config, project_path
from .train_contact_region import draw_box
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


FIELDS = [
    "record_id", "probe", "image_name", "a_error_px", "b_error_px", "c_error_px",
    "oracle_corrects", "predicted_ttc", "ttc_error", "ttc_entropy", "ttc_confidence",
    "ttc_probabilities", "a_top5_box48", "b_top5_box48", "c_top5_box48",
    "a_top5_min_error", "b_top5_min_error", "c_top5_min_error", "failure_class",
    "contact_frame", "contact_score", "contact_threshold", "contact_margin",
    "track_points_32", "track_span_32", "speed_5", "speed_10", "speed_20",
    "acceleration", "max_turn_deg", "direction_stability", "mean_tip_confidence",
    "min_tip_confidence", "mean_base_confidence", "min_base_confidence",
    "c_parallel_error", "c_perpendicular_error", "trajectory_warning",
]


def parse_topk(value: str) -> list[tuple[float, float, float]]:
    points = []
    for item in value.split(";"):
        if item:
            x, y, score = item.split(",")
            points.append((float(x), float(y), float(score)))
    return points


def min_topk_error(pred: dict[str, str]) -> float:
    target = np.asarray([float(pred["target_x"]), float(pred["target_y"])])
    return min(float(np.linalg.norm(np.asarray([x, y]) - target)) for x, y, _ in parse_topk(pred["topk_points"]))


def speed_over_window(points: list[dict[str, str]], frame: int, window: int) -> float:
    valid = [row for row in points if frame - window <= int(row["frame_id"]) <= frame]
    if len(valid) < 2:
        return 0.0
    first, last = valid[0], valid[-1]
    dt = max(int(last["frame_id"]) - int(first["frame_id"]), 1)
    return math.hypot(float(last["tip_x"]) - float(first["tip_x"]), float(last["tip_y"]) - float(first["tip_y"])) / dt


def trajectory_diagnostics(points: list[dict[str, str]], frame: int, window: int = 32) -> dict:
    valid = [row for row in points if frame - window <= int(row["frame_id"]) <= frame]
    velocities = []
    for left, right in zip(valid[:-1], valid[1:]):
        dt = max(int(right["frame_id"]) - int(left["frame_id"]), 1)
        velocities.append(np.asarray([
            (float(right["tip_x"]) - float(left["tip_x"])) / dt,
            (float(right["tip_y"]) - float(left["tip_y"])) / dt,
        ]))
    turns = []
    for left, right in zip(velocities[:-1], velocities[1:]):
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom > 1e-6:
            turns.append(math.degrees(math.acos(float(np.clip(np.dot(left, right) / denom, -1.0, 1.0)))))
    acceleration = float(np.mean([np.linalg.norm(b - a) for a, b in zip(velocities[:-1], velocities[1:])])) if len(velocities) > 1 else 0.0
    unit = [velocity / np.linalg.norm(velocity) for velocity in velocities if np.linalg.norm(velocity) > 1e-6]
    stability = float(np.linalg.norm(np.mean(unit, axis=0))) if unit else 0.0
    tip_conf = [float(row.get("tip_confidence", 0.0)) for row in valid]
    base_conf = [float(row.get("base_confidence", 0.0)) for row in valid]
    span = int(valid[-1]["frame_id"]) - int(valid[0]["frame_id"]) if valid else 0
    warnings = []
    if span < 24:
        warnings.append("short_window")
    if turns and max(turns) > 45:
        warnings.append("turn")
    if acceleration > 0.5:
        warnings.append("acceleration")
    if tip_conf and min(tip_conf) < 0.5:
        warnings.append("low_tip_confidence")
    return {
        "track_points_32": len(valid), "track_span_32": span,
        "speed_5": speed_over_window(points, frame, 5),
        "speed_10": speed_over_window(points, frame, 10),
        "speed_20": speed_over_window(points, frame, 20),
        "acceleration": acceleration, "max_turn_deg": max(turns) if turns else 0.0,
        "direction_stability": stability,
        "mean_tip_confidence": float(np.mean(tip_conf)) if tip_conf else 0.0,
        "min_tip_confidence": min(tip_conf) if tip_conf else 0.0,
        "mean_base_confidence": float(np.mean(base_conf)) if base_conf else 0.0,
        "min_base_confidence": min(base_conf) if base_conf else 0.0,
        "trajectory_warning": ";".join(warnings),
    }


def save_comparison(source: dict[str, str], predictions: list[tuple[str, dict[str, str]]], output: Path) -> None:
    base = Image.open(source["vision_path"]).convert("RGB")
    panels = []
    for label, pred in predictions:
        panel = base.copy()
        draw = ImageDraw.Draw(panel)
        draw_box(draw, float(pred["target_x"]), float(pred["target_y"]), 48, "lime", 4)
        for rank, (x, y, _) in enumerate(parse_topk(pred["topk_points"])):
            draw_box(draw, x, y, 48, "magenta" if rank == 0 else "yellow", 4 if rank == 0 else 2)
        draw.rectangle((0, 0, panel.width, 34), fill="black")
        draw.text((8, 10), f"{label} error={float(pred['error_px']):.1f}px", fill="white")
        panels.append(panel)
    canvas = Image.new("RGB", (base.width * len(panels), base.height), "black")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * base.width, 0))
    ensure_dir(output.parent)
    canvas.save(output)


def audit(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    rows = read_csv_rows(project_path(cfg["samples_csv"]))
    rows_by_name = {row["image_name"]: row for row in rows}
    predictions = {
        name: {row["image_name"]: row for row in read_csv_rows(project_path(path))}
        for name, path in cfg["prediction_files"].items()
    }
    ttc_rows = {row["image_name"]: row for row in read_csv_rows(project_path(cfg["ttc_predictions_csv"]))}
    contacts = {row["record_id"]: row for row in read_csv_rows(project_path(cfg["contact_index_csv"]))}
    track_groups: dict[str, list[dict[str, str]]] = {}
    for row in read_csv_rows(project_path(cfg["motion_tracks_csv"])):
        track_groups.setdefault(row["record_id"], []).append(row)
    for group in track_groups.values():
        group.sort(key=lambda row: int(row["frame_id"]))

    target_records = set(cfg.get("records", ["rec_00092", "rec_00096", "rec_00098"]))
    target_probes = {int(value) for value in cfg.get("probes", [75, 100])}
    audit_rows = []
    debug_dir = project_path(cfg["debug_dir"])
    for source in rows:
        if source["record_id"] not in target_records or int(source["probe"]) not in target_probes:
            continue
        a, b, c = (predictions[name][source["image_name"]] for name in ("A", "B", "C"))
        ttc = ttc_rows[source["image_name"]]
        probabilities = np.asarray([float(value) for value in ttc["probabilities"].split(";")])
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))) / np.log(len(probabilities)))
        diag = trajectory_diagnostics(track_groups.get(source["record_id"], []), int(source["frame_id"]))
        contact = contacts.get(source["record_id"], {})
        oracle_corrects = float(b["error_px"]) <= 48 and float(b["error_px"]) < float(a["error_px"])
        c_top5 = c.get("top5_box48_hit") == "1"
        if float(b["error_px"]) > 48:
            failure_class = "oracle_hotspot_or_label_failure"
        elif float(c["error_px"]) > 48 and c_top5 and (float(ttc["absolute_error"]) > 25 or entropy > 0.85):
            failure_class = "ttc_low_confidence_ranking_failure"
        elif float(c["error_px"]) > 48 and c_top5:
            failure_class = "candidate_ranking_failure"
        elif float(c["error_px"]) > 48:
            failure_class = "ttc_or_representation_failure"
        elif oracle_corrects:
            failure_class = "predicted_ttc_resolved"
        else:
            failure_class = "not_oracle_sensitive"
        row = {
            "record_id": source["record_id"], "probe": source["probe"], "image_name": source["image_name"],
            "a_error_px": a["error_px"], "b_error_px": b["error_px"], "c_error_px": c["error_px"],
            "oracle_corrects": "1" if oracle_corrects else "0", "predicted_ttc": ttc["predicted_ttc"],
            "ttc_error": ttc["absolute_error"], "ttc_entropy": f"{entropy:.6f}",
            "ttc_confidence": f"{1.0 - entropy:.6f}", "ttc_probabilities": ttc["probabilities"],
            "a_top5_box48": a["top5_box48_hit"], "b_top5_box48": b["top5_box48_hit"], "c_top5_box48": c["top5_box48_hit"],
            "a_top5_min_error": f"{min_topk_error(a):.3f}", "b_top5_min_error": f"{min_topk_error(b):.3f}",
            "c_top5_min_error": f"{min_topk_error(c):.3f}", "failure_class": failure_class,
            "contact_frame": contact.get("contact_frame", ""), "contact_score": contact.get("contact_score", ""),
            "contact_threshold": contact.get("threshold", ""),
            "contact_margin": f"{float(contact.get('contact_score', 0)) - float(contact.get('threshold', 0)):.6f}",
            "c_parallel_error": c.get("parallel_error", ""), "c_perpendicular_error": c.get("perpendicular_error", ""),
            **{key: f"{value:.6f}" if isinstance(value, float) else value for key, value in diag.items()},
        }
        audit_rows.append(row)
        save_comparison(source, [("A baseline", a), ("B oracle", b), ("C predicted", c)], debug_dir / f"{source['record_id']}_probe{int(source['probe']):03d}.jpg")

    write_csv_rows(project_path(cfg["output_csv"]), audit_rows, FIELDS)
    classes = {name: sum(row["failure_class"] == name for row in audit_rows) for name in sorted({row["failure_class"] for row in audit_rows})}
    summary = {"samples": len(audit_rows), "records": sorted(target_records), "classes": classes, "output_csv": str(project_path(cfg["output_csv"])), "debug_dir": str(debug_dir)}
    write_json(project_path(cfg["metrics_json"]), summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit repeated far-horizon failure records across A/B/C.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="failure_record_audit")
    args = parser.parse_args()
    audit(args.config, args.section)


if __name__ == "__main__":
    main()
