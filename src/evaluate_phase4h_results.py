"""Record-bootstrap OOF evaluation for aligned DINO and the Phase4H safety gate."""
from __future__ import annotations

import argparse
import json

from .build_phase4g_dino_v1_fusion import bootstrap_comparison
from .config import load_config, project_path
from .utils import read_csv_rows, write_json


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate(config_path: str, section: str) -> dict:
    cfg = load_config(config_path)[section]
    v1 = read_csv_rows(project_path(cfg["v1_query_csv"]))
    aligned = read_csv_rows(project_path(cfg["aligned_query_csv"]))
    gated = read_csv_rows(project_path(cfg["gated_query_csv"]))
    gate = load_json(project_path(cfg["gate_json"]))
    comparison_cfg = {
        "bootstrap_iterations": int(cfg["bootstrap_iterations"]),
        "bootstrap_seed": int(cfg["bootstrap_seed"]),
    }
    aligned_comparison = bootstrap_comparison(v1, aligned, comparison_cfg)
    gated_comparison = bootstrap_comparison(v1, gated, comparison_cfg)
    selected = gate.get("selection", {}).get("selected", {})
    coverage = float(selected.get("coverage", 0.0))
    precision = float(selected.get("strict_triple_win_precision", 0.0))
    ready = bool(
        gate.get("enabled", False)
        and coverage >= float(cfg["minimum_coverage"])
        and precision >= float(cfg["minimum_precision"])
        and gated_comparison["accepted"]
    )
    report = {
        "mode": "phase4h_strict_oof_alignment_and_gate_evaluation_v1",
        "aligned_dino_minus_v1": aligned_comparison,
        "gated_minus_v1": gated_comparison,
        "gate": {
            "enabled": bool(gate.get("enabled", False)),
            "coverage": coverage,
            "strict_triple_win_precision": precision,
        },
        "ready_for_development_validation": ready,
        "next_action": (
            "build the frozen Phase4H validation features and run validation once"
            if ready
            else "retain V1 on OOF; diagnose the prescribed temporal/LoRA/data branch without touching final holdout"
        ),
        "integrity": {
            "source": "strict development OOF only",
            "sealed_final_holdout_rows_read": 0,
        },
    }
    write_json(project_path(cfg["metrics_json"]), report)
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase4H strict OOF alignment and gate outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--section", default="phase4h_oof_evaluation_v1")
    args = parser.parse_args()
    evaluate(args.config, args.section)


if __name__ == "__main__":
    main()
