from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import load_config, project_path
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


PREDICTION_FIELDS = [
    "dataset_split",
    "split",
    "record_id",
    "image_name",
    "image_path",
    "tip_x",
    "tip_y",
    "base_x",
    "base_y",
    "pred_tip_x",
    "pred_tip_y",
    "pred_base_x",
    "pred_base_y",
    "tip_error_px",
    "base_error_px",
    "mean_error_px",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def record_splits(rows: list[dict[str, str]], train: float = 0.8, val: float = 0.1) -> dict[tuple[str, str], str]:
    records = sorted({(row["split"], row["record_id"]) for row in rows})
    train_end = int(round(len(records) * train))
    val_end = train_end + int(round(len(records) * val))
    split_map = {}
    for idx, key in enumerate(records):
        if idx < train_end:
            split_map[key] = "train"
        elif idx < val_end:
            split_map[key] = "val"
        else:
            split_map[key] = "test"
    return split_map


def make_heatmap(width: int, height: int, x: float, y: float, sigma: float) -> np.ndarray:
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)[:, None]
    heatmap = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


class SensorDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        input_width: int,
        input_height: int,
        sigma: float,
    ) -> None:
        self.rows = rows
        self.input_width = input_width
        self.input_height = input_height
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, str]]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_arr = np.transpose(image_arr, (2, 0, 1))

        tip_x = float(row["tip_x"]) / orig_w * self.input_width
        tip_y = float(row["tip_y"]) / orig_h * self.input_height
        base_x = float(row["base_x"]) / orig_w * self.input_width
        base_y = float(row["base_y"]) / orig_h * self.input_height
        tip_heatmap = make_heatmap(self.input_width, self.input_height, tip_x, tip_y, self.sigma)
        base_heatmap = make_heatmap(self.input_width, self.input_height, base_x, base_y, self.sigma)
        target = np.stack([tip_heatmap, base_heatmap], axis=0)
        coords = np.asarray([tip_x, tip_y, base_x, base_y, orig_w, orig_h], dtype=np.float32)

        return {
            "image": torch.from_numpy(image_arr),
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
    def __init__(self, in_channels: int = 3, out_channels: int = 2, features: int = 16) -> None:
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
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "coords": torch.stack([item["coords"] for item in batch]),
        "rows": [item["row"] for item in batch],
    }


def peak_xy(heatmap: torch.Tensor) -> tuple[float, float]:
    flat_idx = int(torch.argmax(heatmap).item())
    height, width = heatmap.shape
    y = flat_idx // width
    x = flat_idx % width
    return float(x), float(y)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_width: int,
    input_height: int,
    split_name: str,
) -> tuple[dict, list[dict]]:
    model.eval()
    rows_out: list[dict] = []
    tip_errors: list[float] = []
    base_errors: list[float] = []
    losses: list[float] = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            preds = model(images)
            losses.append(float(criterion(preds, targets).item()))
            for idx, row in enumerate(batch["rows"]):
                coords = batch["coords"][idx].cpu().numpy()
                orig_w = float(coords[4])
                orig_h = float(coords[5])
                pred_tip_x, pred_tip_y = peak_xy(preds[idx, 0].cpu())
                pred_base_x, pred_base_y = peak_xy(preds[idx, 1].cpu())
                pred_tip_x = pred_tip_x / input_width * orig_w
                pred_tip_y = pred_tip_y / input_height * orig_h
                pred_base_x = pred_base_x / input_width * orig_w
                pred_base_y = pred_base_y / input_height * orig_h
                tip_x = float(row["tip_x"])
                tip_y = float(row["tip_y"])
                base_x = float(row["base_x"])
                base_y = float(row["base_y"])
                tip_error = float(np.hypot(pred_tip_x - tip_x, pred_tip_y - tip_y))
                base_error = float(np.hypot(pred_base_x - base_x, pred_base_y - base_y))
                tip_errors.append(tip_error)
                base_errors.append(base_error)
                rows_out.append(
                    {
                        "dataset_split": split_name,
                        "split": row["split"],
                        "record_id": row["record_id"],
                        "image_name": row["image_name"],
                        "image_path": row["image_path"],
                        "tip_x": row["tip_x"],
                        "tip_y": row["tip_y"],
                        "base_x": row["base_x"],
                        "base_y": row["base_y"],
                        "pred_tip_x": f"{pred_tip_x:.3f}",
                        "pred_tip_y": f"{pred_tip_y:.3f}",
                        "pred_base_x": f"{pred_base_x:.3f}",
                        "pred_base_y": f"{pred_base_y:.3f}",
                        "tip_error_px": f"{tip_error:.3f}",
                        "base_error_px": f"{base_error:.3f}",
                        "mean_error_px": f"{((tip_error + base_error) / 2.0):.3f}",
                    }
                )

    mean_errors = [(t + b) / 2.0 for t, b in zip(tip_errors, base_errors)]
    summary = {
        "split": split_name,
        "samples": len(rows_out),
        "loss": float(np.mean(losses)) if losses else None,
        "tip_mean_error_px": float(np.mean(tip_errors)) if tip_errors else None,
        "tip_median_error_px": float(np.median(tip_errors)) if tip_errors else None,
        "base_mean_error_px": float(np.mean(base_errors)) if base_errors else None,
        "base_median_error_px": float(np.median(base_errors)) if base_errors else None,
        "mean_error_px": float(np.mean(mean_errors)) if mean_errors else None,
        "median_error_px": float(np.median(mean_errors)) if mean_errors else None,
        "pck_8": float(np.mean([err <= 8.0 for err in mean_errors])) if mean_errors else None,
        "pck_16": float(np.mean([err <= 16.0 for err in mean_errors])) if mean_errors else None,
        "pck_32": float(np.mean([err <= 32.0 for err in mean_errors])) if mean_errors else None,
    }
    return summary, rows_out


def save_prediction_debug(rows: list[dict], output_dir: Path, limit: int) -> None:
    ensure_dir(output_dir)
    for idx, row in enumerate(rows[:limit]):
        image = Image.open(row["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        tip = (float(row["tip_x"]), float(row["tip_y"]))
        base = (float(row["base_x"]), float(row["base_y"]))
        pred_tip = (float(row["pred_tip_x"]), float(row["pred_tip_y"]))
        pred_base = (float(row["pred_base_x"]), float(row["pred_base_y"]))
        draw.line((base[0], base[1], tip[0], tip[1]), fill="yellow", width=3)
        draw.line((pred_base[0], pred_base[1], pred_tip[0], pred_tip[1]), fill="lime", width=3)
        for x, y, color in ((*tip, "red"), (*base, "cyan"), (*pred_tip, "magenta"), (*pred_base, "blue")):
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=3)
        out = output_dir / f"{idx:03d}_{Path(row['image_name']).stem}_pred.jpg"
        image.save(out)


def train_sensor_localizer(config_path: str, epochs_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    localizer_cfg = cfg["sensor_localizer"]
    model_cfg = localizer_cfg["model"]
    labels_csv = project_path(localizer_cfg["labels_output_csv"])
    rows = read_csv_rows(labels_csv)
    split_map = record_splits(rows)
    for row in rows:
        row["dataset_split"] = split_map[(row["split"], row["record_id"])]

    input_width = int(model_cfg["input_width"])
    input_height = int(model_cfg["input_height"])
    sigma = float(model_cfg["gaussian_sigma"])
    batch_size = int(model_cfg["batch_size"])
    epochs = int(epochs_override or model_cfg["epochs"])
    seed = int(model_cfg.get("seed", 42))
    set_seed(seed)

    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_split[row["dataset_split"]].append(row)

    train_dataset = SensorDataset(rows_by_split["train"], input_width, input_height, sigma)
    val_dataset = SensorDataset(rows_by_split["val"], input_width, input_height, sigma)
    test_dataset = SensorDataset(rows_by_split["test"], input_width, input_height, sigma)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg.get("weight_decay", 0.0)),
    )

    checkpoint_dir = ensure_dir(project_path(model_cfg["checkpoint_dir"]))
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    best_val = float("inf")
    history: list[dict] = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_summary, _ = evaluate(model, val_loader, device, input_width, input_height, "val")
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(val_summary["loss"]) if val_summary["loss"] is not None else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_median_error_px": val_summary["median_error_px"],
                "val_pck_16": val_summary["pck_16"],
            }
        )
        state = {
            "model": model.state_dict(),
            "config": model_cfg,
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
                f"val_loss={val_loss:.6f} val_median_px={val_summary['median_error_px']:.3f}"
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_summary, val_predictions = evaluate(model, val_loader, device, input_width, input_height, "val")
    test_summary, test_predictions = evaluate(model, test_loader, device, input_width, input_height, "test")
    train_summary, train_predictions = evaluate(model, train_loader, device, input_width, input_height, "train")

    all_predictions = train_predictions + val_predictions + test_predictions
    metrics_path = project_path(model_cfg["metrics_json"])
    predictions_path = project_path(model_cfg["predictions_csv"])
    debug_dir = project_path(model_cfg["debug_dir"])
    write_csv_rows(predictions_path, all_predictions, PREDICTION_FIELDS)
    save_prediction_debug(val_predictions + test_predictions, debug_dir, int(model_cfg.get("debug_samples", 0)))

    summary = {
        "device": str(device),
        "seed": seed,
        "input_width": input_width,
        "input_height": input_height,
        "epochs": epochs,
        "elapsed_seconds": round(time.time() - start, 2),
        "split_counts": {name: len(items) for name, items in rows_by_split.items()},
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
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
    parser = argparse.ArgumentParser(description="Train a lightweight tip/base sensor localizer.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_sensor_localizer(args.config, args.epochs)


if __name__ == "__main__":
    main()
