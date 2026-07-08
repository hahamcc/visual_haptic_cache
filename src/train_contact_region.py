from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import load_config, project_path
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


PREDICTION_FIELDS = [
    "dataset_split",
    "split",
    "record_id",
    "image_name",
    "vision_path",
    "touch_path",
    "is_contact_outlier",
    "target_x",
    "target_y",
    "pred_x",
    "pred_y",
    "pred_score",
    "error_px",
    "abs_error_x",
    "abs_error_y",
    "pck_16",
    "pck_32",
    "pck_48",
    "bbox_hit",
    "top5_hit_48",
    "top5_bbox_hit",
    "topk_points",
]

RETRIEVAL_FIELDS = [
    "dataset_split",
    "query_split",
    "query_record_id",
    "query_image_name",
    "query_vision_path",
    "query_pred_x",
    "query_pred_y",
    "query_target_x",
    "query_target_y",
    "retrieved_split",
    "retrieved_record_id",
    "retrieved_image_name",
    "retrieved_vision_path",
    "retrieved_touch_path",
    "retrieved_target_x",
    "retrieved_target_y",
    "distance",
    "same_record",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_heatmap(width: int, height: int, x: float, y: float, sigma: float) -> np.ndarray:
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)[:, None]
    heatmap = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def read_manifest_touch_paths(path: Path) -> dict[tuple[str, str, int], str]:
    if not path.exists():
        return {}
    return {
        (row["split"], row["record_id"], int(row["frame_id"])): row["touch_path"]
        for row in read_csv_rows(path)
    }


def attach_touch_paths(rows: list[dict[str, str]], manifest_csv: Path) -> None:
    touch_by_key = read_manifest_touch_paths(manifest_csv)
    for row in rows:
        split = row["split"]
        record_id = row["record_id"]
        contact_frame = int(row["contact_frame_from_name"])
        current_frame = int(row["frame_id"])
        row["touch_path"] = touch_by_key.get(
            (split, record_id, contact_frame),
            touch_by_key.get((split, record_id, current_frame), ""),
        )


class ContactRegionDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        input_width: int,
        input_height: int,
        geometry_sigma: float,
    ) -> None:
        self.rows = rows
        self.input_width = input_width
        self.input_height = input_height
        self.geometry_sigma = geometry_sigma

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = row["vision_path"] or row["image_path"]
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_arr = np.transpose(image_arr, (2, 0, 1))

        tip_x = float(row["tip_x"]) / orig_w * self.input_width
        tip_y = float(row["tip_y"]) / orig_h * self.input_height
        base_x = float(row["base_x"]) / orig_w * self.input_width
        base_y = float(row["base_y"]) / orig_h * self.input_height
        tip_map = make_heatmap(self.input_width, self.input_height, tip_x, tip_y, self.geometry_sigma)
        base_map = make_heatmap(self.input_width, self.input_height, base_x, base_y, self.geometry_sigma)
        direction_x = np.full((self.input_height, self.input_width), float(row["direction_x"]), dtype=np.float32)
        direction_y = np.full((self.input_height, self.input_width), float(row["direction_y"]), dtype=np.float32)
        features = np.concatenate(
            [image_arr, np.stack([tip_map, base_map, direction_x, direction_y], axis=0)],
            axis=0,
        )

        target = np.load(row["heatmap_path"]).astype(np.float32)[None, :, :]
        coords = np.asarray(
            [
                float(row["target_tip_x"]),
                float(row["target_tip_y"]),
                float(row["image_width"]),
                float(row["image_height"]),
            ],
            dtype=np.float32,
        )
        return {
            "input": torch.from_numpy(features),
            "target": torch.from_numpy(target),
            "coords": torch.from_numpy(coords),
            "row": row,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 1, features: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(features * 2, features * 4)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(features * 4, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(features * 2, features)
        self.head = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.head(d1))


def collate_batch(batch: list[dict]) -> dict:
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "coords": torch.stack([item["coords"] for item in batch]),
        "rows": [item["row"] for item in batch],
    }


def topk_points(
    heatmap: torch.Tensor,
    k: int,
    suppression_radius: int,
) -> list[tuple[float, float, float]]:
    work = heatmap.clone()
    height, width = work.shape
    points = []
    for _ in range(k):
        flat_idx = int(torch.argmax(work).item())
        score = float(work.reshape(-1)[flat_idx].item())
        y = flat_idx // width
        x = flat_idx % width
        points.append((float(x), float(y), score))
        x0 = max(0, x - suppression_radius)
        x1 = min(width, x + suppression_radius + 1)
        y0 = max(0, y - suppression_radius)
        y1 = min(height, y + suppression_radius + 1)
        work[y0:y1, x0:x1] = -1.0
    return points


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_width: int,
    input_height: int,
    split_name: str,
    topk: int,
    suppression_radius: int,
    bbox_half_size: float,
) -> tuple[dict, list[dict]]:
    model.eval()
    criterion = nn.MSELoss()
    losses: list[float] = []
    predictions: list[dict] = []
    errors: list[float] = []
    pck16: list[bool] = []
    pck32: list[bool] = []
    pck48: list[bool] = []
    bbox_hits: list[bool] = []
    top5_hits: list[bool] = []
    top5_bbox_hits: list[bool] = []
    outlier_errors: list[float] = []
    normal_errors: list[float] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            preds = model(inputs)
            losses.append(float(criterion(preds, targets).item()))
            for idx, row in enumerate(batch["rows"]):
                target_x, target_y, orig_w, orig_h = batch["coords"][idx].cpu().numpy()
                points = topk_points(preds[idx, 0].cpu(), topk, suppression_radius)
                scaled_points = [
                    (x / input_width * orig_w, y / input_height * orig_h, score)
                    for x, y, score in points
                ]
                pred_x, pred_y, pred_score = scaled_points[0]
                abs_x = abs(pred_x - float(target_x))
                abs_y = abs(pred_y - float(target_y))
                error = float(np.hypot(abs_x, abs_y))
                is_outlier = row["record_id"] == "rec_00007"
                top5_hit = any(
                    float(np.hypot(x - float(target_x), y - float(target_y))) <= 48.0
                    for x, y, _ in scaled_points
                )
                top5_bbox_hit = any(
                    abs(x - float(target_x)) <= bbox_half_size and abs(y - float(target_y)) <= bbox_half_size
                    for x, y, _ in scaled_points
                )
                errors.append(error)
                pck16.append(error <= 16.0)
                pck32.append(error <= 32.0)
                pck48.append(error <= 48.0)
                bbox_hit = abs_x <= bbox_half_size and abs_y <= bbox_half_size
                bbox_hits.append(bbox_hit)
                top5_hits.append(top5_hit)
                top5_bbox_hits.append(top5_bbox_hit)
                if is_outlier:
                    outlier_errors.append(error)
                else:
                    normal_errors.append(error)
                predictions.append(
                    {
                        "dataset_split": split_name,
                        "split": row["split"],
                        "record_id": row["record_id"],
                        "image_name": row["image_name"],
                        "vision_path": row["vision_path"],
                        "touch_path": row.get("touch_path", ""),
                        "is_contact_outlier": "1" if is_outlier else "0",
                        "target_x": f"{float(target_x):.3f}",
                        "target_y": f"{float(target_y):.3f}",
                        "pred_x": f"{pred_x:.3f}",
                        "pred_y": f"{pred_y:.3f}",
                        "pred_score": f"{pred_score:.6f}",
                        "error_px": f"{error:.3f}",
                        "abs_error_x": f"{abs_x:.3f}",
                        "abs_error_y": f"{abs_y:.3f}",
                        "pck_16": "1" if error <= 16.0 else "0",
                        "pck_32": "1" if error <= 32.0 else "0",
                        "pck_48": "1" if error <= 48.0 else "0",
                        "bbox_hit": "1" if bbox_hit else "0",
                        "top5_hit_48": "1" if top5_hit else "0",
                        "top5_bbox_hit": "1" if top5_bbox_hit else "0",
                        "topk_points": ";".join(f"{x:.3f},{y:.3f},{score:.6f}" for x, y, score in scaled_points),
                    }
                )

    summary = {
        "split": split_name,
        "samples": len(predictions),
        "loss": float(np.mean(losses)) if losses else None,
        "mean_error_px": float(np.mean(errors)) if errors else None,
        "median_error_px": float(np.median(errors)) if errors else None,
        "pck_16": float(np.mean(pck16)) if pck16 else None,
        "pck_32": float(np.mean(pck32)) if pck32 else None,
        "pck_48": float(np.mean(pck48)) if pck48 else None,
        "bbox_hit": float(np.mean(bbox_hits)) if bbox_hits else None,
        "top5_hit_48": float(np.mean(top5_hits)) if top5_hits else None,
        "top5_bbox_hit": float(np.mean(top5_bbox_hits)) if top5_bbox_hits else None,
        "normal_median_error_px": float(np.median(normal_errors)) if normal_errors else None,
        "outlier_median_error_px": float(np.median(outlier_errors)) if outlier_errors else None,
    }
    return summary, predictions


def heatmap_preview(heatmap_path: str | Path, size: tuple[int, int]) -> Image.Image:
    arr = np.load(heatmap_path).astype(np.float32)
    arr = np.clip(arr / max(float(arr.max()), 1e-8) * 255.0, 0, 255).astype(np.uint8)
    return ImageOps.colorize(Image.fromarray(arr, mode="L").resize(size), black="black", white="red")


def draw_prediction_overlay(row: dict, output_path: Path, title: str | None = None) -> None:
    image = Image.open(row["vision_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    target = (float(row["target_x"]), float(row["target_y"]))
    pred = (float(row["pred_x"]), float(row["pred_y"]))
    draw.ellipse((target[0] - 8, target[1] - 8, target[0] + 8, target[1] + 8), outline="lime", width=4)
    draw.ellipse((pred[0] - 8, pred[1] - 8, pred[0] + 8, pred[1] + 8), outline="magenta", width=4)
    draw.line((target[0], target[1], pred[0], pred[1]), fill="white", width=2)
    for point in row["topk_points"].split(";")[1:]:
        x_text, y_text, _ = point.split(",")
        x = float(x_text)
        y = float(y_text)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="yellow", width=2)
    if row["is_contact_outlier"] == "1":
        draw.rectangle((4, 4, 170, 28), fill="black")
        draw.text((10, 9), "contact outlier rec_00007", fill="orange")
    if title:
        draw.rectangle((4, image.height - 28, image.width - 4, image.height - 4), fill="black")
        draw.text((10, image.height - 23), title, fill="white")
    heatmap = heatmap_preview(row["heatmap_path"], image.size) if "heatmap_path" in row else None
    if heatmap is None:
        canvas = image
    else:
        canvas = Image.new("RGB", (image.width * 2, image.height), "black")
        canvas.paste(image, (0, 0))
        canvas.paste(heatmap, (image.width, 0))
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def save_debug_predictions(predictions: list[dict], rows_by_name: dict[str, dict], output_dir: Path, limit: int) -> None:
    ensure_dir(output_dir)
    for idx, pred in enumerate(predictions[:limit]):
        row = {**pred, **rows_by_name[pred["image_name"]]}
        output_path = output_dir / f"{idx:03d}_{Path(pred['image_name']).stem}_proposal.jpg"
        title = f"err={pred['error_px']} pck48={pred['pck_48']} top5={pred['top5_hit_48']}"
        draw_prediction_overlay(row, output_path, title)


def crop_mean_rgb(image_path: str, x: float, y: float, crop_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    half = crop_size // 2
    left = max(0, int(round(x)) - half)
    top = max(0, int(round(y)) - half)
    right = min(image.width, int(round(x)) + half)
    bottom = min(image.height, int(round(y)) + half)
    crop = image.crop((left, top, right, bottom))
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    if arr.size == 0:
        return np.zeros(3, dtype=np.float32)
    return arr.reshape(-1, 3).mean(axis=0).astype(np.float32)


def cache_feature(row: dict, x: float, y: float, crop_size: int) -> np.ndarray:
    width = float(row["image_width"])
    height = float(row["image_height"])
    numeric = np.asarray(
        [
            x / width,
            y / height,
            float(row["tip_x"]) / width,
            float(row["tip_y"]) / height,
            float(row["base_x"]) / width,
            float(row["base_y"]) / height,
            float(row["direction_x"]),
            float(row["direction_y"]),
            float(row["probe"]) / 100.0,
        ],
        dtype=np.float32,
    )
    crop = crop_mean_rgb(row["vision_path"], x, y, crop_size)
    return np.concatenate([numeric, crop], axis=0)


def save_retrieval_debug(row: dict, output_path: Path) -> None:
    query = Image.open(row["query_vision_path"]).convert("RGB")
    retrieved = Image.open(row["retrieved_vision_path"]).convert("RGB")
    touch = Image.open(row["retrieved_touch_path"]).convert("RGB") if row["retrieved_touch_path"] else Image.new("RGB", query.size, "black")
    touch = touch.resize(query.size)
    for image, x_key, y_key, color in (
        (query, "query_pred_x", "query_pred_y", "magenta"),
        (retrieved, "retrieved_target_x", "retrieved_target_y", "lime"),
    ):
        draw = ImageDraw.Draw(image)
        x = float(row[x_key])
        y = float(row[y_key])
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=4)
    canvas = Image.new("RGB", (query.width * 3, query.height), "black")
    canvas.paste(query, (0, 0))
    canvas.paste(retrieved, (query.width, 0))
    canvas.paste(touch, (query.width * 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 28), fill="black")
    draw.text(
        (8, 8),
        f"query | retrieved | touch  dist={float(row['distance']):.4f} same_record={row['same_record']}",
        fill="white",
    )
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def build_retrieval_outputs(
    predictions: list[dict],
    rows_by_name: dict[str, dict],
    crop_size: int,
    output_csv: Path,
    output_json: Path,
    debug_dir: Path,
    debug_samples: int,
) -> dict:
    train_predictions = [pred for pred in predictions if pred["dataset_split"] == "train"]
    query_predictions = [pred for pred in predictions if pred["dataset_split"] in {"val", "test"}]
    cache_vectors = []
    cache_rows = []
    for pred in train_predictions:
        source = rows_by_name[pred["image_name"]]
        x = float(source["target_tip_x"])
        y = float(source["target_tip_y"])
        cache_vectors.append(cache_feature(source, x, y, crop_size))
        cache_rows.append({**source, **pred})
    if not cache_vectors:
        write_csv_rows(output_csv, [], RETRIEVAL_FIELDS)
        summary = {"cache_size": 0, "queries": 0}
        write_json(output_json, summary)
        return summary

    cache_matrix = np.stack(cache_vectors, axis=0)
    retrieval_rows = []
    distances = []
    same_record_hits = []
    for pred in query_predictions:
        source = rows_by_name[pred["image_name"]]
        query_vec = cache_feature(source, float(pred["pred_x"]), float(pred["pred_y"]), crop_size)
        dists = np.linalg.norm(cache_matrix - query_vec[None, :], axis=1)
        best_idx = int(np.argmin(dists))
        best = cache_rows[best_idx]
        distance = float(dists[best_idx])
        same_record = pred["record_id"] == best["record_id"]
        distances.append(distance)
        same_record_hits.append(same_record)
        retrieval_rows.append(
            {
                "dataset_split": pred["dataset_split"],
                "query_split": pred["split"],
                "query_record_id": pred["record_id"],
                "query_image_name": pred["image_name"],
                "query_vision_path": pred["vision_path"],
                "query_pred_x": pred["pred_x"],
                "query_pred_y": pred["pred_y"],
                "query_target_x": pred["target_x"],
                "query_target_y": pred["target_y"],
                "retrieved_split": best["split"],
                "retrieved_record_id": best["record_id"],
                "retrieved_image_name": best["image_name"],
                "retrieved_vision_path": best["vision_path"],
                "retrieved_touch_path": best.get("touch_path", ""),
                "retrieved_target_x": best["target_x"],
                "retrieved_target_y": best["target_y"],
                "distance": f"{distance:.6f}",
                "same_record": "1" if same_record else "0",
            }
        )

    write_csv_rows(output_csv, retrieval_rows, RETRIEVAL_FIELDS)
    for idx, row in enumerate(retrieval_rows[:debug_samples]):
        output_path = debug_dir / f"{idx:03d}_{Path(row['query_image_name']).stem}_retrieval.jpg"
        save_retrieval_debug(row, output_path)
    summary = {
        "cache_size": len(cache_rows),
        "queries": len(retrieval_rows),
        "mean_distance": float(np.mean(distances)) if distances else None,
        "median_distance": float(np.median(distances)) if distances else None,
        "same_record_rate": float(np.mean(same_record_hits)) if same_record_hits else None,
        "output_csv": str(output_csv),
        "debug_dir": str(debug_dir),
    }
    write_json(output_json, summary)
    return summary


def train_contact_region(config_path: str, epochs_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    region_cfg = cfg["contact_region"]
    rows = read_csv_rows(project_path(region_cfg["samples_csv"]))
    attach_touch_paths(rows, project_path(cfg["manifest"]["output_csv"]))
    rows_by_name = {row["image_name"]: row for row in rows}
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_split[row["dataset_split"]].append(row)

    seed = int(region_cfg.get("seed", 42))
    set_seed(seed)
    input_width = int(region_cfg["input_width"])
    input_height = int(region_cfg["input_height"])
    batch_size = int(region_cfg["batch_size"])
    epochs = int(epochs_override or region_cfg["epochs"])
    topk = int(region_cfg["topk"])
    suppression_radius = int(region_cfg["topk_suppression_radius"])
    bbox_half_size = float(region_cfg["bbox_half_size"])

    train_dataset = ContactRegionDataset(rows_by_split["train"], input_width, input_height, float(region_cfg["geometry_sigma"]))
    val_dataset = ContactRegionDataset(rows_by_split["val"], input_width, input_height, float(region_cfg["geometry_sigma"]))
    test_dataset = ContactRegionDataset(rows_by_split["test"], input_width, input_height, float(region_cfg["geometry_sigma"]))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        generator=generator,
    )
    eval_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(region_cfg["learning_rate"]),
        weight_decay=float(region_cfg.get("weight_decay", 0.0)),
    )
    checkpoint_dir = ensure_dir(project_path(region_cfg["checkpoint_dir"]))
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    best_val = float("inf")
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_summary, _ = evaluate(
            model, val_loader, device, input_width, input_height, "val", topk, suppression_radius, bbox_half_size
        )
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(val_summary["loss"]) if val_summary["loss"] is not None else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_median_error_px": val_summary["median_error_px"],
                "val_pck_48": val_summary["pck_48"],
                "val_top5_hit_48": val_summary["top5_hit_48"],
            }
        )
        state = {
            "model": model.state_dict(),
            "config": region_cfg,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_summary": val_summary,
        }
        torch.save(state, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, best_path)
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} val_median_px={val_summary['median_error_px']:.3f} "
                f"val_pck48={val_summary['pck_48']:.3f}"
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_summary, train_predictions = evaluate(
        model, eval_train_loader, device, input_width, input_height, "train", topk, suppression_radius, bbox_half_size
    )
    val_summary, val_predictions = evaluate(
        model, val_loader, device, input_width, input_height, "val", topk, suppression_radius, bbox_half_size
    )
    test_summary, test_predictions = evaluate(
        model, test_loader, device, input_width, input_height, "test", topk, suppression_radius, bbox_half_size
    )
    all_predictions = train_predictions + val_predictions + test_predictions

    predictions_path = project_path(region_cfg["predictions_csv"])
    metrics_path = project_path(region_cfg["metrics_json"])
    retrieval_csv = project_path(region_cfg["retrieval_csv"])
    retrieval_json = project_path(region_cfg["retrieval_json"])
    debug_dir = project_path(region_cfg["debug_dir"])
    retrieval_debug_dir = project_path(region_cfg["retrieval_debug_dir"])
    write_csv_rows(predictions_path, all_predictions, PREDICTION_FIELDS)
    save_debug_predictions(val_predictions + test_predictions, rows_by_name, debug_dir, int(region_cfg["debug_samples"]))
    retrieval_summary = build_retrieval_outputs(
        all_predictions,
        rows_by_name,
        int(region_cfg["cache_crop_size"]),
        retrieval_csv,
        retrieval_json,
        retrieval_debug_dir,
        int(region_cfg["debug_samples"]),
    )

    summary = {
        "device": str(device),
        "seed": seed,
        "input_channels": 7,
        "input_width": input_width,
        "input_height": input_height,
        "epochs": epochs,
        "elapsed_seconds": round(time.time() - start, 2),
        "split_counts": {name: len(items) for name, items in rows_by_split.items()},
        "contact_outlier_record": "rec_00007",
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
        "retrieval": retrieval_summary,
        "history": history,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "predictions_csv": str(predictions_path),
        "debug_dir": str(debug_dir),
    }
    write_json(metrics_path, summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 2 future contact-region baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_contact_region(args.config, args.epochs)


if __name__ == "__main__":
    main()
