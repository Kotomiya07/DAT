# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ""
# Root of raw domain data (custom datasets)
# Split metadata json path
_C.DATA.SPLIT_FILE = ""
# Label mapping json path
_C.DATA.LABEL_MAP = ""
# Dataset name
_C.DATA.DATASET = "imagenet"
# Input image size
_C.DATA.IMG_SIZE = 224
# Number of classes for dataset specific overrides
_C.DATA.NUM_CLASSES = 1000
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = "bicubic"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "dat"
# Model name
_C.MODEL.NAME = "dat_tiny"
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ""
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

_C.MODEL.PRETRAINED = None

_C.MODEL.DAT = CN(new_allowed=True)
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Reference batch size for LR scaling
_C.TRAIN.REFERENCE_BATCH = 512.0
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# Freeze backbone parameters during training
_C.TRAIN.FREEZE_BACKBONE = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = "pixel"
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = "batch"

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# WandB settings
# -----------------------------------------------------------------------------
_C.WANDB = CN()
_C.WANDB.ENABLED = False
_C.WANDB.PROJECT = ""
_C.WANDB.ENTITY = ""
_C.WANDB.RUN_NAME = ""
_C.WANDB.TAGS = []
_C.WANDB.MODE = "online"
_C.WANDB.RESUME = ""
_C.WANDB.ID = ""
_C.WANDB.NOTES = ""

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

# overwritten by command line argument
_C.AMP = False
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ""
# Tag of experiment, overwritten by command line argument
_C.TAG = "default"
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 100
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.amp:
        config.AMP = args.amp
    if getattr(args, "freeze_backbone", None) is not None:
        config.TRAIN.FREEZE_BACKBONE = args.freeze_backbone
    if args.print_freq:
        config.PRINT_FREQ = args.print_freq
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if getattr(args, "wandb", False):
        config.WANDB.ENABLED = True
    if getattr(args, "wandb_project", None):
        config.WANDB.PROJECT = args.wandb_project
    if getattr(args, "wandb_entity", None):
        config.WANDB.ENTITY = args.wandb_entity
    if getattr(args, "wandb_run_name", None):
        config.WANDB.RUN_NAME = args.wandb_run_name
    if getattr(args, "wandb_tags", None):
        config.WANDB.TAGS = args.wandb_tags
    if getattr(args, "wandb_mode", None):
        config.WANDB.MODE = args.wandb_mode
    if getattr(args, "wandb_resume", None):
        config.WANDB.RESUME = args.wandb_resume
    if getattr(args, "wandb_id", None):
        config.WANDB.ID = args.wandb_id
    if getattr(args, "wandb_notes", None):
        config.WANDB.NOTES = args.wandb_notes

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    if config.DATA.NUM_CLASSES:
        config.MODEL.NUM_CLASSES = config.DATA.NUM_CLASSES
        if hasattr(config.MODEL, "DAT") and "num_classes" in config.MODEL.DAT:
            config.MODEL.DAT.num_classes = config.MODEL.NUM_CLASSES

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
