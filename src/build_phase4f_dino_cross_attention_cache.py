"""Strict OOF V2 cache ranking with frozen DINOv2 spatial patch-token matching."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .build_cache_retrieval import crop_contact_patch, motion_geometry_feature, standardize, visual_patch_feature_from_patch
from .build_phase4e_oof_multiscale_cache import build_shortlists, normalized_features
from .config import load_config, project_path
from .evaluate_oracle_tactile_retrieval import tactile_difference, tactile_embedding, tactile_metrics
from .phase4f_dino_cross_attention import DinoSpatialCacheRanker, FrozenDinoV2, SCORE_WEIGHT_OPTIONS, composite_score
from .train_phase4b_predicted_box_cache_ranker import is_final_holdout, prediction_map, set_seed
from .train_soft_tactile_cache_ranker import image_tensor, ranks
from .utils import ensure_dir, read_csv_rows, write_csv_rows, write_json


QUERY_FIELDS = ["query_record_id", "query_image_name", "query_probe", "oof_fold", "pred_x", "pred_y", "selected_cache_record_id", "selected_cache_image_name", "ranker_best_score", "ranker_margin", "ranker_oracle_embedding_rank", "tactile_diff_mae", "tactile_ssim", "tactile_mask_iou"]
CANDIDATE_FIELDS = ["query_record_id", "query_image_name", "query_probe", "oof_fold", "candidate_rank", "candidate_score", "embedding_score", "ssim_score", "iou_score", "cross_attention_score", "hard_negative_flag", "candidate_record_id", "candidate_image_name", "candidate_tactile_embedding_distance", "candidate_tactile_ssim", "candidate_tactile_mask_iou", "candidate_oracle_embedding_rank"]


def encode(backbone: FrozenDinoV2, patches: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(patches), batch_size):
        outputs.append(backbone(image_tensor(patches[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def embedding_matrix(rows: list[dict[str, str]], touch, label: str) -> np.ndarray:
    values = []
    for index, row in enumerate(rows, start=1):
        values.append(tactile_embedding(touch(row)))
        if index % 100 == 0 or index == len(rows):
            print(f"phase4f {label}: {index}/{len(rows)} tactile embeddings", flush=True)
    return np.stack(values).astype(np.float32)


def tactile_targets(query_rows: list[dict[str, str]], cache_rows: list[dict[str, str]], candidates: np.ndarray, touch, threshold: float, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Build offline auxiliary labels while reusing the fold's tactile-difference cache."""
    ssim, iou = np.zeros(candidates.shape, np.float32), np.zeros(candidates.shape, np.float32)
    for index, row in enumerate(query_rows):
        source = touch(row)
        for item, cache_index in enumerate(candidates[index]):
            metric = tactile_metrics(source, touch(cache_rows[int(cache_index)]), threshold)
            ssim[index, item], iou[index, item] = metric["tactile_ssim"], metric["tactile_mask_iou"]
        if (index + 1) % 100 == 0 or index + 1 == len(query_rows):
            print(f"phase4f {label}: {index + 1}/{len(query_rows)} query label groups", flush=True)
    return ssim, iou


def hard_flags(hand: np.ndarray, embedding: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """Visually/geometrically close but tactually bad candidates receive extra loss weight."""
    return ((hand <= np.median(hand, axis=1, keepdims=True)) & ((embedding >= np.median(embedding, axis=1, keepdims=True)) | (iou <= np.median(iou, axis=1, keepdims=True)))).astype(np.float32)


def model_scores(model: DinoSpatialCacheRanker, qd: np.ndarray, cd: np.ndarray, qc: np.ndarray, cc: np.ndarray, qg: np.ndarray, cg: np.ndarray, hand: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = ([], [], [])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(qd), batch_size):
            end, indices = start + batch_size, slice(start, start + batch_size)
            out = model(torch.from_numpy(qd[indices]).to(device), torch.from_numpy(cd[indices]).to(device), torch.from_numpy(qc[indices]).to(device), torch.from_numpy(cc[indices]).to(device), torch.from_numpy(qg[indices]).to(device), torch.from_numpy(cg[indices]).to(device), torch.from_numpy(hand[indices]).to(device))
            for bucket, value in zip(result, out, strict=True): bucket.append(value.cpu().numpy())
    return tuple(np.concatenate(value) for value in result)


def fold_run(fold: str, rows: list[dict[str, str]], folds: dict[str, str], predictions: dict[str, dict[str, str]], cfg: dict, device: torch.device) -> tuple[list[dict], list[dict]]:
    fit = [row for row in rows if folds[row["image_name"]] != fold]
    query = [row for row in rows if folds[row["image_name"]] == fold]
    if {r["record_id"] for r in fit} & {r["record_id"] for r in query}: raise RuntimeError("Record leakage in Phase4F fold")
    detail, context = int(cfg["detail_crop_size"]), int(cfg["context_crop_size"])
    train_k, eval_k = min(int(cfg["geometry_filter_k"]), len(fit)), min(int(cfg["geometry_filter_k"]), len(rows))
    gt = [(float(r["target_tip_x"]), float(r["target_tip_y"])) for r in fit]
    all_gt = [(float(r["target_tip_x"]), float(r["target_tip_y"])) for r in rows]
    raw_geo = np.stack([motion_geometry_feature(r, x, y) for r, (x, y) in zip(fit, gt, strict=True)])
    _, gm, gs = standardize(raw_geo, raw_geo)
    hand_detail = np.stack([visual_patch_feature_from_patch(crop_contact_patch(r["vision_path"], x, y, detail)) for r, (x, y) in zip(fit, gt, strict=True)])
    hand_context = np.stack([visual_patch_feature_from_patch(crop_contact_patch(r["vision_path"], x, y, context)) for r, (x, y) in zip(fit, gt, strict=True)])
    _, dm, ds = standardize(hand_detail, hand_detail); _, cm, cs = standardize(hand_context, hand_context)
    # The ranker is fitted only on ``fit`` records, but OOF retrieval must see
    # the complete development cache.  This mirrors Phase4E: same-record cache
    # entries are excluded per query below, not removed wholesale by OOF fold.
    fit_cache_d, fit_cache_c, fit_cache_g, fit_cache_dh, fit_cache_ch = normalized_features(fit, gt, gm, gs, dm, ds, cm, cs, detail, context)
    cache_d, cache_c, cache_g, cache_dh, cache_ch = normalized_features(rows, all_gt, gm, gs, dm, ds, cm, cs, detail, context)
    fit_xy = [(float(predictions[r["image_name"]]["pred_x"]), float(predictions[r["image_name"]]["pred_y"])) for r in fit]
    query_xy = [(float(predictions[r["image_name"]]["pred_x"]), float(predictions[r["image_name"]]["pred_y"])) for r in query]
    fit_d, fit_c, fit_g, fit_dh, fit_ch = normalized_features(fit, fit_xy, gm, gs, dm, ds, cm, cs, detail, context)
    query_d, query_c, query_g, query_dh, query_ch = normalized_features(query, query_xy, gm, gs, dm, ds, cm, cs, detail, context)
    tactile_cache: dict[str, np.ndarray] = {}
    def touch(r: dict[str, str]) -> np.ndarray: return tactile_difference(r["touch_path"], tactile_cache, int(cfg["tactile_size"]))
    print(f"phase4f fold {fold}: caching tactile embeddings for development_cache={len(rows)}", flush=True)
    cache_e = embedding_matrix(rows, touch, f"fold {fold} development cache")
    embedding_by_name = {row["image_name"]: value for row, value in zip(rows, cache_e, strict=True)}
    fit_e = np.stack([embedding_by_name[row["image_name"]] for row in fit]).astype(np.float32)
    query_e = np.stack([embedding_by_name[row["image_name"]] for row in query]).astype(np.float32)
    train_groups = build_shortlists(fit, fit_g, fit_dh, fit_ch, fit_e, fit, fit_cache_g, fit_cache_dh, fit_cache_ch, fit_e, train_k, True)
    eval_groups = build_shortlists(query, query_g, query_dh, query_ch, query_e, rows, cache_g, cache_dh, cache_ch, cache_e, eval_k, True)
    train_idx, train_target, train_hand = train_groups[:3]; indices, targets, hand = eval_groups[:3]
    if any(row["record_id"] == rows[int(cache_index)]["record_id"] for row, group in zip(query, indices, strict=True) for cache_index in group):
        raise RuntimeError("Phase4F OOF evaluation shortlist contains a same-record cache entry.")
    train_ssim, train_iou = tactile_targets(fit, fit, train_idx, touch, float(cfg["tactile_mask_threshold"]), f"fold {fold} train")
    flags = hard_flags(train_hand, train_target, train_iou)
    print(f"phase4f fold {fold}: loading frozen {cfg['dino_model']} and encoding visual patches", flush=True)
    backbone = FrozenDinoV2(str(cfg["dino_model"]), int(cfg["dino_image_size"])).to(device)
    fit_cache_dt, fit_cache_ct = encode(backbone, fit_cache_d, device, int(cfg["batch_size"])), encode(backbone, fit_cache_c, device, int(cfg["batch_size"]))
    cache_dt, cache_ct = encode(backbone, cache_d, device, int(cfg["batch_size"])), encode(backbone, cache_c, device, int(cfg["batch_size"]))
    fit_dt, fit_ct = encode(backbone, fit_d, device, int(cfg["batch_size"])), encode(backbone, fit_c, device, int(cfg["batch_size"]))
    query_dt, query_ct = encode(backbone, query_d, device, int(cfg["batch_size"])), encode(backbone, query_c, device, int(cfg["batch_size"]))
    model = DinoSpatialCacheRanker(backbone.feature_dim, cache_g.shape[1], int(cfg["hidden_dim"]), int(cfg["attention_heads"]), float(cfg["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    target_std = max(float(train_target.std()), 1e-6)
    for epoch in range(int(cfg["epochs"])):
        for start in range(0, len(fit), int(cfg["batch_size"])):
            batch = np.arange(start, min(start + int(cfg["batch_size"]), len(fit))); local = train_idx[batch]
            emb, ssim, iou = model(torch.from_numpy(fit_dt[batch]).to(device), torch.from_numpy(fit_cache_dt[local]).to(device), torch.from_numpy(fit_ct[batch]).to(device), torch.from_numpy(fit_cache_ct[local]).to(device), torch.from_numpy(fit_g[batch]).to(device), torch.from_numpy(fit_cache_g[local]).to(device), torch.from_numpy(train_hand[batch]).to(device))
            target = torch.from_numpy(train_target[batch]).to(device); weight = 1 + (float(cfg["hard_negative_weight"]) - 1) * torch.from_numpy(flags[batch]).to(device)
            regression = (nn.functional.smooth_l1_loss((emb-target)/target_std, torch.zeros_like(emb), reduction="none") * weight).mean()
            dist = torch.softmax(-target / float(cfg["target_temperature"]), 1); listwise = -(dist * torch.log_softmax(-emb / float(cfg["target_temperature"]), 1)).sum(1).mean()
            aux = float(cfg["ssim_loss_weight"]) * nn.functional.mse_loss(ssim, torch.from_numpy(train_ssim[batch]).to(device)) + float(cfg["iou_loss_weight"]) * nn.functional.mse_loss(iou, torch.from_numpy(train_iou[batch]).to(device))
            optimizer.zero_grad(set_to_none=True); (regression + float(cfg["listwise_weight"])*listwise + aux).backward(); optimizer.step()
        print(f"phase4f fold {fold}: finished epoch {epoch + 1}/{int(cfg['epochs'])}", flush=True)
    ed, sd, id_ = model_scores(model, query_dt, cache_dt[indices], query_ct, cache_ct[indices], query_g, cache_g[indices], hand, device, int(cfg["batch_size"]))
    score_weights = str(cfg["score_weights"])
    if score_weights not in SCORE_WEIGHT_OPTIONS:
        raise ValueError(f"Unknown Phase4F score_weights={score_weights!r}; choose one of {sorted(SCORE_WEIGHT_OPTIONS)}.")
    score = composite_score(torch.from_numpy(ed), torch.from_numpy(sd), torch.from_numpy(id_), SCORE_WEIGHT_OPTIONS[score_weights]).numpy(); order = np.argsort(score, 1)
    eval_ssim, eval_iou = tactile_targets(query, rows, indices, touch, float(cfg["tactile_mask_threshold"]), f"fold {fold} OOF"); eval_flags = hard_flags(hand, targets, eval_iou)
    queries, candidates = [], []
    for qi, row in enumerate(query):
        choice = int(order[qi, 0]); selected = rows[int(indices[qi, choice])]; metric = tactile_metrics(touch(row), touch(selected), float(cfg["tactile_mask_threshold"])); oracle = int(np.argmin(targets[qi])); rank = int(ranks(score[qi])[oracle])
        queries.append({"query_record_id":row["record_id"],"query_image_name":row["image_name"],"query_probe":row["probe"],"oof_fold":fold,"pred_x":f"{query_xy[qi][0]:.3f}","pred_y":f"{query_xy[qi][1]:.3f}","selected_cache_record_id":selected["record_id"],"selected_cache_image_name":selected["image_name"],"ranker_best_score":f"{score[qi,choice]:.6f}","ranker_margin":f"{score[qi,order[qi,1]]-score[qi,choice]:.6f}","ranker_oracle_embedding_rank":str(rank),**{key:f"{metric[key]:.6f}" for key in ("tactile_diff_mae", "tactile_ssim", "tactile_mask_iou")}})
        for pos, item in enumerate(order[qi], 1):
            cache_row=rows[int(indices[qi,item])]; candidates.append({"query_record_id":row["record_id"],"query_image_name":row["image_name"],"query_probe":row["probe"],"oof_fold":fold,"candidate_rank":str(pos),"candidate_score":f"{score[qi,item]:.6f}","embedding_score":f"{ed[qi,item]:.6f}","ssim_score":f"{sd[qi,item]:.6f}","iou_score":f"{id_[qi,item]:.6f}","cross_attention_score":f"{score[qi,item]:.6f}","hard_negative_flag":str(int(eval_flags[qi,item])),"candidate_record_id":cache_row["record_id"],"candidate_image_name":cache_row["image_name"],"candidate_tactile_embedding_distance":f"{targets[qi,item]:.6f}","candidate_tactile_ssim":f"{eval_ssim[qi,item]:.6f}","candidate_tactile_mask_iou":f"{eval_iou[qi,item]:.6f}","candidate_oracle_embedding_rank":str(int(ranks(targets[qi])[item]))})
    ensure_dir(project_path(cfg["checkpoint_dir"])); torch.save({"model_state":model.state_dict(),"dino_model":cfg["dino_model"],"fold":fold},project_path(cfg["checkpoint_dir"])/f"fold_{fold}.pt")
    return queries, candidates


def build(config_path: str, section: str) -> dict:
    cfg=load_config(config_path)[section]; set_seed(int(cfg["seed"])); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows=read_csv_rows(project_path(cfg["samples_csv"]));
    if any(is_final_holdout(r) for r in rows): raise RuntimeError("Phase4F refuses sealed final-holdout samples.")
    rows=[r for r in rows if r["dataset_split"]=="train"]; predictions=prediction_map(read_csv_rows(project_path(cfg["oof_predictions_csv"])),rows,"train","Phase4F strict OOF"); folds={name:p["oof_fold"] for name,p in predictions.items()}
    queries: list[dict]=[]; candidates: list[dict]=[]
    for fold in sorted(set(folds.values())):
        q,c=fold_run(fold,rows,folds,predictions,cfg,device); queries+=q; candidates+=c
    if len({r["query_image_name"] for r in queries})!=len(rows): raise RuntimeError("OOF output must cover every development train query once.")
    write_csv_rows(project_path(cfg["query_output_csv"]),queries,QUERY_FIELDS); write_csv_rows(project_path(cfg["candidate_output_csv"]),candidates,CANDIDATE_FIELDS)
    summary={"mode":"phase4f_strict_oof_dinov2_spatial_cross_attention","queries":len(queries),"candidates":len(candidates),"dino_model":cfg["dino_model"],"score_weights":cfg["score_weights"],"integrity":{"evaluation_cache":"complete_development_pool","same_record_cache_excluded":True,"sealed_final_holdout_rows_read":0,"query_tactile_usage":"offline supervision/evaluation only"}}
    write_json(project_path(cfg["metrics_json"]),summary); print(summary); return summary


def main() -> None:
    parser=argparse.ArgumentParser(description="Build strict OOF DINOv2 spatial cache ranker outputs."); parser.add_argument("--config",default="configs/default.yaml"); parser.add_argument("--section",default="phase4f_dino_cross_attention_oof_v1"); args=parser.parse_args(); build(args.config,args.section)
if __name__ == "__main__": main()
