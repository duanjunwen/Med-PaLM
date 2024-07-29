
import torch
import torch.distributed as dist
from torch.optim import Adam
from calflops import calculate_flops
import os
from datetime import timedelta, datetime
from medpalm.utils.train_utils import set_seed, to_torch_dtype, get_model_numel, format_numel
from medpalm.utils.config_utils import (
    parse_args,
)
from medpalm.models.model import MedPalm
from medpalm.acceleration.checkpoint import set_grad_checkpoint
from medpalm.models.transformer import Attention, AttentionLayers
import functools
from functools import partial
from profiler.performance_evaluator import PerformanceEvaluator

def main():
    cfg = parse_args(training=True)
    os.environ['RANK']='0'
    os.environ['WORLD_SIZE']='1'
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='8080'
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    # dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    set_seed(8192)
    # device = torch.cuda.current_device  # device cuda:0
    dtype = to_torch_dtype(cfg.dtype)
    torch._dynamo.config.optimize_ddp = False

    model = MedPalm()

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
    )

    criterion = lambda x, *arg, **kwargs: (x * x).mean()

    model_numel, model_numel_trainable = get_model_numel(model)
    Evaluator = functools.partial(
        PerformanceEvaluator,
        model_numel=model_numel,
        num_layers=8,
        hidden_size=512,
        vocab_size=20000,
        max_seq_length=1024,
        ignore_steps=1,
        # num_steps=cfg.benchmark_num_steps, # epoch * steps
        num_steps=cfg.epochs * cfg.steps, # epoch * steps
        use_torch_profiler=False,
        cfg=cfg,
        # use_torch_profiler=True,
        # torch_profiler_path=f"./profiler/{plugin}",
    )

    model = model.to(device='cuda', dtype=dtype)

    # # test flops
    # img = torch.randn(cfg.batch_size, 3, 256, 256).to(device='cuda', dtype=dtype)
    # caption = torch.randint(0, 20000, (cfg.batch_size, 1024)).to(device='cuda')
    # model_flops, model_macs, model_params = calculate_flops(model=model,args=[img, caption])
    # print(f"model flops {model_flops}, model_macs {model_macs}, model_params {model_params}")

        # print(f"num_ckpt_blocks {num_ckpt_blocks}")
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model, Attention)
        num_ckpt_blocks = 0
        for module in model.modules():
            if isinstance(module, Attention):
                module.grad_checkpointing = module.grad_checkpointing and num_ckpt_blocks < cfg.num_ckpt_blocks
                num_ckpt_blocks += module.grad_checkpointing
                print(f"module.grad_checkpointing {module.grad_checkpointing}")
        print(f"num_ckpt_blocks {num_ckpt_blocks}")

    model.train()
    
    performance_evaluator = Evaluator(stdit_weight_memory=format_numel(model_numel*2),total_weight_memory=format_numel(model_numel*2))
    performance_evaluator.on_fit_start()
    for epoch in range(0, cfg.epochs):
        performance_evaluator.start_new_iter()
        for step in range(0, cfg.steps):
            # input
            img = torch.randn(cfg.batch_size, 3, 256, 256).to(device='cuda', dtype=dtype)
            caption = torch.randint(0, 20000, (cfg.batch_size, 1024)).to(device='cuda')
            img.requires_grad=True
            performance_evaluator.before_forward()
            output = model(img, caption)

            loss = criterion(output[0]).mean()
            performance_evaluator.before_backward()
            loss.backward()
            performance_evaluator.before_optimizer_update()
            optimizer.step()

            performance_evaluator.end_iter(input_ids=torch.empty(cfg.batch_size, cfg.epochs))
            performance_evaluator.start_new_iter()
    performance_evaluator.on_fit_end()
#   ENV: Sus
# C:\Users\duanj\miniconda3\envs\SustainableLLM\Lib\site-packages
# python setup.py install
# python .\script\train_benchmark.py .\configs\medpalm2\train\train-v1.py
if __name__ == "__main__":
    main()
