"""Evaluate a trained DAT model on the Unicode character dataset test split.

This script reports top-1/top-5 accuracy and writes a CSV file
containing all misclassified samples together with the predicted
characters and confidences.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml
from yacs.config import CfgNode

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import _C as BASE_CONFIG
from config import get_config
from data.build import build_transform
from data.character_dataset import UnicodeCharacterDataset
from logger import create_logger
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DAT checkpoint on the Unicode dataset test split")
    parser.add_argument("--config", "--cfg", required=True, help="Path to the configuration yaml used for training.")
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        required=True,
        help="Path to the trained checkpoint (.pth) to evaluate.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the dataset root path defined in the config (optional).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (defaults to config value).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader workers (defaults to config value).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to use for evaluation.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to store evaluation artifacts. Defaults to <checkpoint_dir>/evaluation_<split>.",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs="+",
        help="Optional config overrides in KEY VALUE pairs.",
    )
    parser.add_argument(
        "--tag",
        default="eval",
        help="Tag string used when constructing config.OUTPUT (optional).",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=100,
        help="How often to log evaluation progress (in batches).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision evaluation if the config supports it.",
    )

    return parser.parse_args()


def _unwrap_wandb_config(data):
    if isinstance(data, dict):
        if set(data.keys()) == {"value"}:
            return _unwrap_wandb_config(data["value"])
        return {key: _unwrap_wandb_config(val) for key, val in data.items()}
    if isinstance(data, list):
        return [_unwrap_wandb_config(item) for item in data]
    return data


def _align_types(data, reference):
    if isinstance(data, dict):
        aligned = {}
        if isinstance(reference, CfgNode):
            ref_mapping = reference
            allow_new = reference.is_new_allowed()
        elif isinstance(reference, dict):
            ref_mapping = reference
            allow_new = False
        else:
            ref_mapping = None
            allow_new = True

        for key, value in data.items():
            ref_value = None
            if isinstance(ref_mapping, CfgNode):
                if key in ref_mapping:
                    ref_value = ref_mapping[key]
                elif allow_new:
                    ref_value = None
                else:
                    continue
            elif isinstance(ref_mapping, dict):
                if key in ref_mapping:
                    ref_value = ref_mapping[key]
                else:
                    continue
            aligned[key] = _align_types(value, ref_value)
        return aligned
    if isinstance(data, list):
        ref_item = None
        if isinstance(reference, (list, tuple)) and reference:
            ref_item = reference[0]
        elif isinstance(reference, CfgNode):
            ref_item = None
        return [_align_types(item, ref_item) for item in data]
    if isinstance(reference, float) and isinstance(data, int):
        return float(data)
    if isinstance(reference, int) and isinstance(data, float):
        return int(data)
    if isinstance(reference, bool) and isinstance(data, (int, float)):
        return bool(data)
    if isinstance(reference, str) and not isinstance(data, str):
        return str(data)
    return data


def _prepare_config_path(cfg_path: Path) -> tuple[Path, Path | None]:
    cfg_path = cfg_path.expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fp:
        try:
            data = yaml.safe_load(fp)
        except yaml.YAMLError as err:
            raise RuntimeError(f"Failed to parse config file '{cfg_path}': {err}") from err

    if not isinstance(data, dict):
        return cfg_path, None

    if not any(isinstance(val, dict) and "value" in val for val in data.values()):
        return cfg_path, None

    converted = _unwrap_wandb_config(data)
    converted.pop("_wandb", None)
    converted = _align_types(converted, BASE_CONFIG)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(converted, tmp, allow_unicode=True)
        temp_path = Path(tmp.name)

    return temp_path, temp_path


def build_eval_config(args: argparse.Namespace):
    cfg_path = Path(args.config)
    prepared_cfg_path, temp_cfg_path = _prepare_config_path(cfg_path)

    base_args = Namespace(
        cfg=str(prepared_cfg_path),
        opts=args.opts,
        batch_size=args.batch_size,
        data_path=args.data_path,
        resume="",
        amp=args.amp,
        output="eval_output",
        tag=args.tag,
        eval=True,
        throughput=False,
        wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_tags=None,
        wandb_mode=None,
        wandb_resume=None,
        wandb_id=None,
        wandb_notes=None,
        freeze_backbone=False,
        print_freq=args.print_freq,
        save_top_k=None,
    )
    try:
        config = get_config(base_args)
    finally:
        if temp_cfg_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_cfg_path.unlink()

    config.defrost()
    config.EVAL_MODE = True
    config.TRAIN.AUTO_RESUME = False
    if args.batch_size is not None:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        config.DATA.NUM_WORKERS = args.num_workers
    if args.data_path is not None:
        config.DATA.DATA_PATH = args.data_path
    config.AMP = args.amp
    config.freeze()
    return config


def unicode_to_char(label: str) -> str:
    if label.startswith("U+"):
        try:
            return chr(int(label[2:], 16))
        except ValueError:
            return label
    return label


class DatasetWithPaths(Dataset):
    def __init__(self, base_dataset: UnicodeCharacterDataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        path, _ = self.base.samples[index]
        return image, target, path


def ensure_results_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    idx_to_unicode: Sequence[str],
    print_freq: int,
    logger,
    use_amp: bool,
):
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()

    total_samples = 0
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    misclassified: list[dict[str, object]] = []
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch_idx, (images, targets, paths) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                outputs, _, _ = model(images)
                loss = criterion(outputs, targets)

        probs = F.softmax(outputs, dim=1)
        top1_confidence, top1_pred = probs.max(dim=1)
        topk = min(5, probs.size(1))
        topk_confidence, topk_indices = probs.topk(topk, dim=1)

        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

        total_loss += loss.item()
        total_samples += targets.size(0)
        top1_correct += (top1_pred == targets).sum().item()
        top5_correct += (topk_indices == targets.unsqueeze(1)).any(dim=1).sum().item()

        for i in range(targets.size(0)):
            if top1_pred[i].item() == targets[i].item():
                continue
            gt_index = targets[i].item()
            pred_index = top1_pred[i].item()
            gt_unicode = idx_to_unicode[gt_index]
            pred_unicode = idx_to_unicode[pred_index]
            misclassified.append(
                {
                    "image_path": paths[i],
                    "target_index": gt_index,
                    "target_unicode": gt_unicode,
                    "target_char": unicode_to_char(gt_unicode),
                    "pred_index": pred_index,
                    "pred_unicode": pred_unicode,
                    "pred_char": unicode_to_char(pred_unicode),
                    "confidence": float(top1_confidence[i].item()),
                    "top5": [
                        {
                            "index": int(topk_indices[i, j].item()),
                            "unicode": idx_to_unicode[topk_indices[i, j].item()],
                            "char": unicode_to_char(idx_to_unicode[topk_indices[i, j].item()]),
                            "confidence": float(topk_confidence[i, j].item()),
                        }
                        for j in range(topk)
                    ],
                }
            )

        if (batch_idx + 1) % max(1, print_freq) == 0:
            logger.info(
                "Processed %d/%d batches - Acc@1 %.3f%% Acc@5 %.3f%%",
                batch_idx + 1,
                len(dataloader),
                100.0 * top1_correct / max(1, total_samples),
                100.0 * top5_correct / max(1, total_samples),
            )

    metrics = {
        "loss": total_loss / max(1, total_samples),
        "top1": top1_correct / max(1, total_samples),
        "top5": top5_correct / max(1, total_samples),
        "samples": total_samples,
    }
    prob_tensor = torch.cat(all_probs, dim=0) if all_probs else torch.empty((0, len(idx_to_unicode)))
    target_tensor = torch.cat(all_targets, dim=0) if all_targets else torch.empty((0,), dtype=torch.long)
    return metrics, misclassified, prob_tensor, target_tensor


def save_roc_curve(probabilities: torch.Tensor, targets: torch.Tensor, output_path: Path, logger) -> None:
    """Render a micro/macro averaged ROC curve image and save it to disk."""

    if probabilities.numel() == 0 or probabilities.size(1) < 2:
        logger.warning("Skipping ROC curve generation: need at least two classes and predictions.")
        return

    probs_np = probabilities.cpu().numpy()
    targets_np = targets.cpu().numpy().astype(int)
    n_classes = probs_np.shape[1]

    classes = np.arange(n_classes)
    target_bin = label_binarize(targets_np, classes=classes)

    per_class = []
    fpr: dict[int | str, np.ndarray] = {}
    tpr: dict[int | str, np.ndarray] = {}
    roc_auc: dict[int | str, float] = {}

    for class_idx in range(n_classes):
        if target_bin[:, class_idx].sum() == 0:
            continue
        fpr[class_idx], tpr[class_idx], _ = roc_curve(target_bin[:, class_idx], probs_np[:, class_idx])
        roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
        per_class.append(class_idx)

    if not per_class:
        logger.warning("Skipping ROC curve generation: targets contain only a single class.")
        return

    fpr["micro"], tpr["micro"], _ = roc_curve(target_bin.ravel(), probs_np.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[idx] for idx in per_class]))
    mean_tpr = np.zeros_like(all_fpr)
    for idx in per_class:
        mean_tpr += np.interp(all_fpr, fpr[idx], tpr[idx])
    mean_tpr /= len(per_class)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC (AUC = {roc_auc['micro']:.4f})",
        color="deeppink",
        linewidth=2,
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC (AUC = {roc_auc['macro']:.4f})",
        color="navy",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Saved ROC curve to %s", output_path)


def write_misclassified_csv(path: Path, rows: Sequence[dict[str, object]]):
    if not rows:
        path.write_text(
            "image_path,target_index,target_unicode,target_char,pred_index,pred_unicode,pred_char,confidence,top5\n",
            encoding="utf-8",
        )
        return

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "image_path",
                "target_index",
                "target_unicode",
                "target_char",
                "pred_index",
                "pred_unicode",
                "pred_char",
                "confidence",
                "top5",
            ]
        )
        for row in rows:
            top5_repr = " | ".join(
                f"{entry['index']}:{entry['unicode']}({entry['char']}):{entry['confidence']:.4f}" for entry in row["top5"]
            )
            writer.writerow(
                [
                    row["image_path"],
                    row["target_index"],
                    row["target_unicode"],
                    row["target_char"],
                    row["pred_index"],
                    row["pred_unicode"],
                    row["pred_char"],
                    f"{row['confidence']:.6f}",
                    top5_repr,
                ]
            )


def main() -> None:
    args = parse_args()
    config = build_eval_config(args)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.results_dir is not None:
        results_dir = Path(args.results_dir).expanduser().resolve()
    else:
        results_dir = checkpoint_path.parent / f"evaluation_{args.split}"
    ensure_results_dir(results_dir)

    logger = create_logger(str(results_dir), dist_rank=0, name="eval")
    logger.info("Using device: %s", "cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_transform(is_train=False, config=config)

    dataset = UnicodeCharacterDataset(
        split=args.split,
        data_root=config.DATA.DATA_PATH,
        split_file=config.DATA.SPLIT_FILE,
        label_map_file=config.DATA.LABEL_MAP,
        transform=transform,
    )
    dataset_with_paths = DatasetWithPaths(dataset)
    idx_to_unicode = [""] * len(dataset.label_map)
    for unicode_id, label in dataset.label_map.items():
        idx_to_unicode[label] = unicode_id

    dataloader = DataLoader(
        dataset_with_paths,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY and device.type == "cuda",
    )

    model = build_model(config)
    model.to(device)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning("State dict mismatch. Missing keys: %s | Unexpected keys: %s", missing, unexpected)
    logger.info("Checkpoint loaded")

    metrics, misclassified, all_probs, all_targets = evaluate(
        model,
        dataloader,
        device,
        idx_to_unicode,
        print_freq=args.print_freq,
        logger=logger,
        use_amp=args.amp,
    )

    logger.info(
        "Evaluation finished - Samples: %d | Loss: %.6f | Acc@1: %.3f%% | Acc@5: %.3f%%",
        metrics["samples"],
        metrics["loss"],
        metrics["top1"] * 100.0,
        metrics["top5"] * 100.0,
    )

    metrics_path = results_dir / f"metrics_{args.split}.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)
    logger.info("Saved metrics to %s", metrics_path)

    roc_path = results_dir / f"roc_{args.split}.png"
    save_roc_curve(all_probs, all_targets, roc_path, logger)

    miscls_path = results_dir / f"misclassified_{args.split}.csv"
    write_misclassified_csv(miscls_path, misclassified)
    logger.info("Saved misclassification log to %s (%d rows)", miscls_path, len(misclassified))


if __name__ == "__main__":
    main()
