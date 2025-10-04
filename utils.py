# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# --------------------------------------------------------

import json
import os
import pickle
import shutil
import subprocess

import torch
import torch.distributed as dist


_cfgnode_allowlisted = False


def _allow_cfg_node_for_torch_load(logger=None):
    global _cfgnode_allowlisted
    if _cfgnode_allowlisted:
        return

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    try:
        from yacs.config import CfgNode
    except ImportError as err:
        if logger is not None:
            logger.warning(f"yacs.config.CfgNode を allowlist できませんでした: {err}")
        return

    try:
        add_safe_globals([CfgNode])
    except (RuntimeError, TypeError) as err:
        if logger is not None:
            logger.warning(f"torch.load のための allowlist 登録に失敗しました: {err}")
        return

    _cfgnode_allowlisted = True


def _safe_torch_load(path, logger=None, map_location="cpu"):
    _allow_cfg_node_for_torch_load(logger)
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as err:
        if logger is not None:
            logger.warning(
                "weights_only=True でのチェックポイント読み込みに失敗したため、weights_only=False で再試行します: %s",
                err,
            )
        return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    _allow_cfg_node_for_torch_load(logger)
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(config.MODEL.RESUME, map_location="cpu", check_hash=True)
    else:
        checkpoint = _safe_torch_load(config.MODEL.RESUME, logger)
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"]
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(ckpt_path, model, logger):
    logger.info(f"==============> Loading pretrained form {ckpt_path}....................")
    checkpoint = _safe_torch_load(ckpt_path, logger)
    msg = model.load_pretrained(checkpoint["model"])
    logger.info(msg)
    logger.info(f"=> Loaded successfully {ckpt_path} ")
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, accuracy=None):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "epoch": epoch,
        "config": config,
    }

    if accuracy is not None:
        save_state["accuracy"] = float(accuracy)

    filename = f"ckpt_epoch_{epoch}.pth"
    save_path = os.path.join(config.OUTPUT, filename)
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

    latest_filename = "ckpt_latest.pth"
    latest_path = os.path.join(config.OUTPUT, latest_filename)
    try:
        shutil.copyfile(save_path, latest_path)
        logger.info(f"{latest_path} updated with latest checkpoint")
    except OSError as err:
        logger.warning(f"最新チェックポイントの更新に失敗しました: {err}")

    top_k = getattr(config, "SAVE_TOP_K", 0)
    keep_files = {filename, latest_filename}
    meta_path = os.path.join(config.OUTPUT, "checkpoint_meta.json")

    if top_k > 0 and accuracy is not None:
        try:
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = []
        except (OSError, json.JSONDecodeError) as err:
            logger.warning(f"チェックポイントメタ情報の読み込みに失敗しました: {err}")
            meta = []

        # 正常なエントリのみ保持し、存在しないファイルは除去
        filtered_meta = []
        for entry in meta:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            acc = entry.get("accuracy")
            if path is None or acc is None:
                continue
            full_path = os.path.join(config.OUTPUT, path)
            if os.path.isfile(full_path) and path != filename:
                filtered_meta.append({"path": path, "accuracy": float(acc), "epoch": entry.get("epoch", 0)})

        filtered_meta.append({"path": filename, "accuracy": float(accuracy), "epoch": epoch})
        filtered_meta.sort(key=lambda e: (e["accuracy"], e.get("epoch", 0)), reverse=True)
        top_meta = filtered_meta[:top_k]
        keep_files.update(entry["path"] for entry in top_meta)

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(top_meta, f, indent=2, ensure_ascii=False)
        except OSError as err:
            logger.warning(f"チェックポイントメタ情報の書き込みに失敗しました: {err}")
    else:
        if os.path.isfile(meta_path) and top_k <= 0:
            try:
                os.remove(meta_path)
            except OSError as err:
                logger.warning(f"チェックポイントメタ情報の削除に失敗しました: {err}")

    for entry in os.listdir(config.OUTPUT):
        if not entry.endswith(".pth"):
            continue
        if entry in keep_files:
            continue
        if not entry.startswith("ckpt_epoch_"):
            continue
        file_path = os.path.join(config.OUTPUT, entry)
        try:
            os.remove(file_path)
            logger.info(f"不要なチェックポイントを削除しました: {file_path}")
        except OSError as err:
            logger.warning(f"チェックポイント {file_path} の削除に失敗しました: {err}")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def init_dist_slurm():
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if "MASTER_PORT" in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ["MASTER_PORT"] = "29500"
    # use MASTER_ADDR in the environment variable if it already exists
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr

    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)

    dist.init_process_group(backend="nccl")
