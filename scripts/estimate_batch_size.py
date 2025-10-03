#!/usr/bin/env python3

import argparse
import json
import os
import sys
import warnings

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import get_config
from data.build import build_transform
from data.character_dataset import UnicodeCharacterDataset
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate max power-of-two batch size")
    parser.add_argument("--cfg", required=True, help="Path to config file")
    parser.add_argument("--data-path", required=True, help="Root directory of dataset")
    parser.add_argument("--split-file", required=False, help="Split metadata JSON")
    parser.add_argument("--label-map", required=False, help="Label mapping JSON")
    parser.add_argument("--min-power", type=int, default=3, help="Minimum power (2^p)")
    parser.add_argument("--max-power", type=int, default=10, help="Maximum power (2^p)")
    parser.add_argument("--amp", action="store_true", help="Use AMP during estimation")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--save-path", help="Optional JSON file to save the result")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate info")
    args, unparsed = parser.parse_known_args()
    if unparsed:
        warnings.warn(f"Ignoring unparsed arguments: {unparsed}")
    return args


def build_dataset_from_config(config, split="train", data_path=None, split_file=None, label_map=None):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")

    if data_path:
        config.defrost()
        config.DATA.DATA_PATH = data_path
        config.freeze()

    if split_file:
        config.defrost()
        config.DATA.SPLIT_FILE = split_file
        config.freeze()

    if label_map:
        config.defrost()
        config.DATA.LABEL_MAP = label_map
        config.freeze()

    transform = build_transform(is_train=True, config=config)
    dataset = UnicodeCharacterDataset(
        split=split,
        data_root=config.DATA.DATA_PATH,
        split_file=config.DATA.SPLIT_FILE,
        label_map_file=config.DATA.LABEL_MAP,
        transform=transform,
    )
    return dataset


def attempt_batch_size(model, dataset, bs, device, use_amp, num_workers, verbose):
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    data_iter = iter(loader)
    try:
        samples, targets = next(data_iter)
    except StopIteration:
        raise RuntimeError("Dataset is smaller than the requested batch size") from None

    samples = samples.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    model.zero_grad(set_to_none=True)

    with autocast(enabled=use_amp):
        outputs, _, _ = model(samples)
        loss = outputs.mean()  # dummy scalar

    loss.backward()
    torch.cuda.synchronize(device)

    if verbose:
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        print(f"[INFO] Batch size {bs} OK - allocated {allocated:.1f} MB, reserved {reserved:.1f} MB")

    del loader, data_iter, samples, targets, outputs, loss
    return True


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA GPU is required for batch size estimation", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    class ArgNamespace:
        def __init__(self, cfg, data_path):
            self.cfg = cfg
            self.opts = None
            self.batch_size = None
            self.data_path = data_path
            self.resume = None
            self.amp = False
            self.output = "output"
            self.tag = "estimate"
            self.eval = False
            self.throughput = False
            self.print_freq = 100

    namespace = ArgNamespace(args.cfg, args.data_path)
    config = get_config(namespace)

    if args.split_file:
        config.defrost()
        config.DATA.SPLIT_FILE = args.split_file
        config.freeze()

    if args.label_map:
        config.defrost()
        config.DATA.LABEL_MAP = args.label_map
        config.freeze()

    dataset = build_dataset_from_config(
        config,
        split="train",
        data_path=args.data_path,
        split_file=args.split_file or config.DATA.SPLIT_FILE,
        label_map=args.label_map or config.DATA.LABEL_MAP,
    )

    model = build_model(config)
    model.to(args.device)
    model.train()

    use_amp = args.amp or config.AMP

    best_bs = None
    tried = []

    for power in range(args.min_power, args.max_power + 1):
        bs = 2**power
        if bs > len(dataset):
            break
        try:
            attempt_batch_size(model, dataset, bs, args.device, use_amp, args.num_workers, args.verbose)
            best_bs = bs
            tried.append({"batch_size": bs, "status": "ok"})
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            message = str(exc).lower()
            tried.append({"batch_size": bs, "status": "fail", "reason": str(exc)})
            torch.cuda.empty_cache()
            if "out of memory" in message:
                if args.verbose:
                    print(f"[WARN] Batch size {bs} failed due to OOM.")
                break
            raise

    if best_bs is None:
        print("No valid batch size found in tested range", file=sys.stderr)
        sys.exit(2)

    result = {
        "best_batch_size": best_bs,
        "best_power": best_bs.bit_length() - 1,
        "tried": tried,
        "dataset_size": len(dataset),
        "amp": use_amp,
    }

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print(best_bs)


if __name__ == "__main__":
    main()
