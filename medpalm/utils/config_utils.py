import argparse
from mmengine.config import Config

def parse_args(training=False):
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("config", help="model config file path")

    # ======================================================
    # General
    # ======================================================
    parser.add_argument("--seed", default=42, type=int, help="generation seed")
    parser.add_argument("--ckpt-path", type=str, help="path to model ckpt; will overwrite cfg.ckpt_path if specified")
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")

    # ======================================================
    # Inference
    # ======================================================
    if not training:
        # output
        parser.add_argument("--save-dir", default=None, type=str, help="path to save generated samples")
        parser.add_argument("--sample-name", default=None, type=str, help="sample name, default is sample_idx")
        parser.add_argument("--start-index", default=None, type=int, help="start index for sample name")
        parser.add_argument("--end-index", default=None, type=int, help="end index for sample name")
        parser.add_argument("--num-sample", default=None, type=int, help="number of samples to generate for one prompt")
        parser.add_argument("--prompt-as-path", action="store_true", help="use prompt as path to save samples")

        # prompt
        parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
        parser.add_argument("--prompt", default=None, type=str, nargs="+", help="prompt list")

        # image/video
        parser.add_argument("--num-frames", default=None, type=int, help="number of frames")
        parser.add_argument("--fps", default=None, type=int, help="fps")
        parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")

        # hyperparameters
        parser.add_argument("--num-sampling-steps", default=None, type=int, help="sampling steps")
        parser.add_argument("--cfg-scale", default=None, type=float, help="balance between cond & uncond")

        # reference
        parser.add_argument("--loop", default=None, type=int, help="loop")
        parser.add_argument("--condition-frame-length", default=None, type=int, help="condition frame length")
        parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
        parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")
    # ======================================================
    # Training
    # ======================================================
    else:
        parser.add_argument("--wandb", default=None, type=bool, help="enable wandb")
        parser.add_argument("--load", default=None, type=str, help="path to continue training")
        parser.add_argument("--data-path", default=None, type=str, help="path to data csv")
        parser.add_argument("--start-from-scratch", action="store_true", help="start training from scratch")

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    return cfg
