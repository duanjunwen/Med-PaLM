# import resource
import psutil
import inspect
from time import time
from typing import Optional
import multiprocessing as mp

import torch
from torch.nn import ModuleList
import torch.distributed as dist
from torch import Tensor
from torch.utils.checkpoint import checkpoint

_pipe = None


def divide(x: float, y: float) -> float:
    if y == 0:
        return float("inf")
    elif y == float("inf"):
        return float("nan")
    return x / y


@torch.no_grad()
def all_reduce_mean(x: float, world_size: int, local_only: bool = True) -> float:
    if local_only:
        return x

    if world_size == 1:
        return x
    tensor = torch.tensor([x], device=torch.cuda.current_device(), dtype=torch.float)
    # dist.all_reduce(tensor)
    tensor = tensor / world_size
    return tensor.item()


def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = mp.Pipe()
    conn_in, conn_out = _pipe
    return conn_in, conn_out

def run(conn, models, x, t, y, t_tmp, kwarg, recompute):
    """
    Run a single forward test in a subprocess.
    Recomputation is a controllable option.
    Memory results are returned via pipe.
    """
    batch_size, seq_len = x.shape[:2]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    empty_mem = torch.cuda.memory_allocated()
    # assert empty_mem == 0
    if not isinstance(models, ModuleList):
        models = ModuleList([models])
    models = models.cuda()
    x = x.cuda()
    models.train()
    layer_name = ["norm1", "attn", "attn.qkv", "attn.q_norm", "attn.k_norm", "attn.attn_drop", "attn.proj", "attn.proj_drop",
                  "cross_attn", "mlp.fc1", "mlp.act", "mlp.drop1", "mlp.norm", "mlp.fc2", "mlp.drop2",
                  "attn_temp", "attn_temp.qkv", "attn_temp.q_norm", "attn_temp.k_norm", "attn_temp.attn_drop", "attn_temp.proj", "attn_temp.proj_drop"]
    for module in models:
        for name, layer in module.named_modules():
            weight_mem, grad_mem, adam_mem, master_weight_mem = get_mem_stats(layer, torch.bfloat16)
            print(f"{name}: weight_mem {weight_mem/1024**3:.2f} GB; grad_mem {grad_mem/1024**3:.2f} GB; adam_mem {adam_mem/1024**3:.2f} GB;")
        kwargs = {}
        forward_params = inspect.signature(module.forward).parameters
        # if 'attention_mask' in forward_params:
        #     kwargs['attention_mask'] = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        # if 'position_ids' in forward_params:
        #     kwargs['position_ids'] = torch.arange(seq_len, dtype=torch.long,
        #                                           device=x.device).unsqueeze(0).view(-1, seq_len)
        # if 'timestep' in forward_params:
        #     kwargs['timestep'] = torch.randint(0, 1000, (x.shape[0],), device=x.device)
        # if 'y' in forward_params:
        #     kwargs['y'] = torch.rand(batch_size, 1, 120, 4096, device=x.device)
        if 'mask' in forward_params:
            kwargs['mask'] = kwarg['mask']
        if 'x_mask' in forward_params:
            kwargs['x_mask'] = kwarg['x_mask']
        if 't0' in forward_params:
            kwargs['t0'] = kwarg['t0']
        if 't0_tmp' in forward_params:
            kwargs['t0_tmp'] = kwarg['t0_tmp']
        if 'T' in forward_params:
            kwargs['T'] = 16
        if 'S' in forward_params:
            kwargs['S'] = 256


        # t_tmp for stdit2
        if recompute:
            # y = checkpoint(module, x, y, t, use_reentrant=False)
            y = checkpoint(module, x, y, t, t_tmp, kwargs['mask'], kwargs['x_mask'], kwargs['t0'], kwargs['t0_tmp'],kwargs['T'], kwargs['S'],  use_reentrant=False)
        else:
            # y = module(x, y, t, **kwargs)
            y = module(x, y, t, t_tmp, kwargs['mask'], kwargs['x_mask'], kwargs['t0'], kwargs['t0_tmp'],kwargs['T'], kwargs['S'])

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    peak_mem = torch.cuda.max_memory_allocated() - empty_mem
    final_mem = torch.cuda.memory_allocated()
    used_mem = final_mem - empty_mem
    # conn.send((used_mem, peak_mem))
    return used_mem, peak_mem

def get_mem_stats(model, dtype):
    """
    Calculate memory statistics for the given module under given precision.
    """
    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_mem = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    adam_mem = sum(p.numel() * 4 for p in model.parameters()) * 2
    master_weight_mem = sum(p.numel() * 4 for p in model.parameters()) if dtype != torch.float32 else 0
    return weight_mem, grad_mem, adam_mem, master_weight_mem


def profile(
    name,
    model,
    x,
    t,
    y,
    t_tmp,
    kwargs,
    dtype=torch.float32,
    zero_size=1,
    recompute=False,
    cpu_offload=False,
    verbose=False,
):
    """
    Profile a pytorch module.
    Controllable options:
        dtype: Training precision
        zero_size: the scale of zero data parallelism
        recompute: run forward with no_grad and recompute it right before backward to save activation memory
        cpu_offload: move optimizer memory away from GPU
    Returns:
        model_mem: total memory usage in regard to the parameters of given module, including weights, gradients, Adam momemtums
        act_mem: peak activation memory during training
    """
    if not isinstance(name, (list, tuple)):
        name = [name]
    conn_in, conn_out = get_pipe()
    weight_mem, grad_mem, adam_mem, master_weight_mem = get_mem_stats(model, dtype)
    optimizer_mem = adam_mem + master_weight_mem
    model_mem = weight_mem + grad_mem
    if not cpu_offload:
        model_mem += optimizer_mem / zero_size
    used_mem, peak_mem = run(conn_in, model, x, t, y, t_tmp, kwargs, recompute)
    extra_mem = peak_mem - used_mem
    act_mem = used_mem - weight_mem
    msg = f"[{', '.join(name)}] -- "
    msg += f"weight memory: {weight_mem/1024**3:.2f} GB, "
    msg += "checkpoint" if recompute else "activation"
    msg += f" memory: {act_mem/1024**3:.2f} GB, "
    msg += f"extra memory: {extra_mem/1024**3:.2f} GB\n"
    if verbose:
        print(msg)
    return model_mem, act_mem

class Timer:
    def __init__(self, use_pp=False, grad_accum=False) -> None:
        # self.state: "before_iter" -> "before_forward" -> "before_backward" -> "before_optimizer_update" -> "before_iter"
        # When using pipeline parallel, forward pass and backward pass are entangled, thus skipping calling before_backward()

        self.start_time: Optional[float] = None
        self.last_time_checkpoint: Optional[float] = None
        self.state = "before_iter"
        self.torch_profiler_duration: float = 0.0
        self.data_load_duration: float = 0.0
        self.forward_duration: float = 0.0
        self.backward_duration: float = 0.0
        self.forward_backward_duration: float = 0.0 # Only used when pp is enabled.
        self.optimizer_update_duration: float = 0.0
        self.iter_duration: float = 0.0
        self.use_pp = use_pp
        self.grad_accum = grad_accum

    def start(self) -> None:
        assert self.state == "before_iter"
        self.start_time = time()
        self.last_time_checkpoint = self.start_time

    def before_forward(self, torch_profiler_duration: float) -> None:
        assert self.state == "before_iter"
        self.state = "before_forward"
        self.torch_profiler_duration = torch_profiler_duration # The time of torch.profiler.step() shouldn't be considered

        current_time = time()
        self.data_load_duration += current_time - self.last_time_checkpoint - self.torch_profiler_duration
        self.last_time_checkpoint = current_time

    def before_backward(self) -> None:
        assert self.state == "before_forward"
        self.state = "before_backward"
        current_time = time()
        self.forward_duration += current_time - self.last_time_checkpoint
        self.last_time_checkpoint = current_time

    def before_optimizer_update(self) -> None:
        if not self.use_pp: # In pipeline parallel, forward and backward are entangled together.
            assert self.state == "before_backward"
            self.state = "before_optimizer_update"
            current_time = time()
            self.backward_duration += current_time - self.last_time_checkpoint
        else:
            assert self.state == "before_forward"
            self.state = "before_optimizer_update"
            current_time = time()
            self.forward_backward_duration += current_time - self.last_time_checkpoint
        self.last_time_checkpoint = current_time

    def end(self) -> None:
        assert self.start_time is not None

        # When using grad accum, optimizer.step might be skipped.
        # TODO: This assertion should be fixed when implementing benchmarking on pipeline + grad_accum
        assert (self.state ==  "before_optimizer_update") or (self.grad_accum and self.state == "before_backward")

        current_time = time()
        if self.state == "before_optimizer_update":
            self.optimizer_update_duration += current_time - self.last_time_checkpoint
        elif self.grad_accum and self.state == "before_backward":
            self.backward_duration += current_time - self.last_time_checkpoint

        self.state = "before_iter"

        current_iter_duration = current_time - self.start_time - self.torch_profiler_duration
        self.iter_duration += current_iter_duration

        self.start_time = None
        self.last_time_checkpoint = None
        self.torch_profiler_duration = 0.0

        return current_iter_duration

    def reset(self) -> None:
        self.data_load_duration = 0.0
        self.forward_duration = 0.0
        self.backward_duration = 0.0
        self.forward_backward_duration = 0.0
        self.optimizer_update_duration = 0.0
        self.iter_duration = 0.0
        self.torch_profiler_duration = 0.0
        self.state = "before_iter"

class PerformanceEvaluator:
    """
        Callback for valuate the performance of the model.
    Args:
        actor_num_params: The number of parameters of the actor model.
        critic_num_params: The number of parameters of the critic model.
        initial_model_num_params: The number of parameters of the initial model.
        reward_model_num_params: The number of parameters of the reward model.
        enable_grad_checkpoint: Whether to enable gradient checkpointing.
        ignore_episodes: The number of episodes to ignore when calculating the performance.
    """

    def __init__(
        self,
        model_numel: int,
        stdit_weight_memory: int,
        total_weight_memory: int,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        max_seq_length: int,
        num_steps: int,
        use_torch_profiler: bool,
        torch_profiler_path: Optional[str] = None,
        enable_grad_checkpoint: bool = False,
        grad_checkpoint_ratio: float = 1.0,
        ignore_steps: int = 0,
        grad_accum: int = 1,
        include_optimizer_time: bool = False,
        include_data_gen_time: bool = False,
        disable_internal_sync: bool = False,
        dp_size: int = 1,
        tp_size: int = 1,
        pp_size: int = 1,
        cfg: dict = None,
    ) -> None:
        self.world_size = 1
        self.model_numel = model_numel
        self.max_seq_length = max_seq_length
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.grad_checkpoint_ratio = grad_checkpoint_ratio
        self.num_steps = num_steps
        self.ignore_steps = ignore_steps
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        print(
            "Incorrect grad_checkpoint_ratio might lead to misleading TFLOPS"
        )
        self.grad_accum = grad_accum
        self.include_optimizer_time = include_optimizer_time
        self.include_data_gen_time = include_data_gen_time
        self.disable_internal_sync = disable_internal_sync
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.mp_world_size = tp_size * pp_size
        self.timer = Timer(use_pp=self.pp_size > 1, grad_accum=(grad_accum > 1))
        self.num_samples: int = 0
        self.flop_megatron = 0
        self.flop: int = 0
        self.flop_hfu: int = 0
        self.stdit_flop: float = 1.66
        self.skip_record = True
        self.step_cnt = 0 # The number of benchmarked iterations, should be args.num_steps - args.ignore_steps
        # When opening grad accum, the number of calling optimizer.step might be smaller than self.step_cnt
        self.optimizer_update_cnt = 0
        self.stdit_weight_memory = stdit_weight_memory
        self.total_weight_memory = total_weight_memory
        self.grad_memory = stdit_weight_memory

        self.stdit_flop_hfu = self.stdit_flop * 3 + self.stdit_flop * (cfg.num_ckpt_blocks/28) #  1 for fwd + 1 for grad ckpt + 2 for bwd
        self.stdit_flop = self.stdit_flop * 3 #  1 for fwd + 2 for bwd


        # Sanity Check
        self.torch_profiler = None
        self.torch_profiler_path = torch_profiler_path
        self.use_torch_profiler = use_torch_profiler
        if self.use_torch_profiler:
            assert self.torch_profiler_path is not None

    def on_fit_start(self) -> None:

        # Check Memory Usage before training
        self.optim_init_memory = torch.cuda.memory_allocated() - self.total_weight_memory * 1024**3
        # HACK: we assume that the memory occupied by gradients is the same as the memory occupied by weights
        # self.grad_memory = self.total_weight_memory
        print(f"Allocated CUDA memory before training: {torch.cuda.memory_allocated()/1024**3:.3f} GB")
        print(f"Peak CUDA memory before training: {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")
        print(
            # f"Peak CPU memory before training: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
            f"Peak CPU memory before training: {dict(psutil.virtual_memory()._asdict())['used'] /1024**3:.3f} GB"
        )

        # Create a torch profiler
        if self.use_torch_profiler:
            assert self.ignore_steps > 1
            wait_steps = 1
            warmup_steps = self.ignore_steps - wait_steps
            active_steps = self.num_steps - warmup_steps - wait_steps
            self.torch_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.torch_profiler_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
            self.torch_profiler.start()

    def step_torch_profiler(self):
        assert self.use_torch_profiler
        self.torch_profiler.step()

    def start_new_iter(self) -> None:
        self.skip_record = (self.ignore_steps > 0 and self.step_cnt < self.ignore_steps)
        if self.skip_record:
            return
        torch.cuda.synchronize()
        self.timer.start()

    def before_forward(self) -> None:
        if self.skip_record:
            return
        if not self.disable_internal_sync:
            torch.cuda.synchronize()
        self.timer.before_forward(torch_profiler_duration=0)

    def before_backward(self) -> None:
        assert self.pp_size == 1, "PerformanceEvaluator.before_backward shouldn't be called when pipeline is enabled"
        if self.skip_record:
            return
        if not self.disable_internal_sync:
            torch.cuda.synchronize()
        self.timer.before_backward()

    def before_optimizer_update(self) -> None:
        if self.skip_record:
            return
        if not self.disable_internal_sync:
            torch.cuda.synchronize()
        self.timer.before_optimizer_update()
        self.optimizer_update_cnt += 1

    def end_iter(self, input_ids: Tensor, **kwargs) -> None:
        self.step_cnt += 1
        print(
            f"\n"
            f"Step: {self.step_cnt - 1}, Is warming up: {self.skip_record}, "
            f"Peak Memory: {torch.cuda.max_memory_allocated()/1024**3:.3f} GB, "
            f"Allocated Memory: {torch.cuda.memory_allocated()/1024**3:.3f} GB, "
            # f"CPU Memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
            f"CPU Memory: {dict(psutil.virtual_memory()._asdict())['used'] /1024**3:.3f} GB"
        )

        if self.skip_record:
            if self.use_torch_profiler:
                self.step_torch_profiler()
            return

        torch.cuda.synchronize()
        current_iter_duration = self.timer.end()

        batch_size, seq_len = input_ids.shape
        self.num_samples += batch_size
        checkpoint_activations_factor = (3 + int(self.enable_grad_checkpoint) * self.grad_checkpoint_ratio)
        self.flop_megatron += (24 * checkpoint_activations_factor * batch_size * seq_len * self.num_layers * (self.hidden_size**2)) * (1. + (seq_len / (6. * self.hidden_size)) + (self.vocab_size / (16. * self.num_layers * self.hidden_size)))
        flop = self.stdit_flop # only stidt
        flop_hfu = self.stdit_flop_hfu
        self.flop += flop
        self.flop_hfu += flop_hfu
        # Reporting speed performance, using statistics on master rank for convenience.
        print(
            # f"TGS of last iteration: {batch_size * seq_len / (current_iter_duration + 1e-12) / self.mp_world_size:.3f} tokens/s, "
            f"TGS of last iteration: {batch_size / (current_iter_duration + 1e-12):.3f} samples/s, "
            f"TFLOPS of last iteration: {flop / (current_iter_duration + 1e-12):.3f}"
        )

        if self.use_torch_profiler:
            self.step_torch_profiler()

    def on_fit_end(self) -> None:

        # End torch profiler
        if self.use_torch_profiler:
            self.torch_profiler.stop()

        with open(f"memory_cuda_0.log", "w") as f:
            f.write(torch.cuda.memory_summary(device=torch.cuda.current_device()))

        # if dist.get_rank() != 0:
        #     return

        # Overall Stats
        num_record_steps = self.step_cnt - self.ignore_steps if (self.step_cnt - self.ignore_steps ) > 0 else 1
        iter_duration = self.timer.iter_duration
        if not self.include_optimizer_time:
            iter_duration -= self.timer.optimizer_update_duration
        if not self.include_data_gen_time:
            iter_duration -= self.timer.data_load_duration

        # rm random data generate in train with random dataset
        avg_duration = all_reduce_mean(iter_duration, self.world_size)
        avg_latency = avg_duration / num_record_steps
        avg_throughput = self.num_samples * self.dp_size / (avg_duration + 1e-12)
        avg_stdit_tflops_per_gpu = self.stdit_flop / (self.timer.forward_duration + self.timer.backward_duration + 1e-12)
        avg_tflops_per_gpu = self.flop /num_record_steps/ (avg_latency + 1e-12)
        avg_tflops_hfu_per_gpu = self.flop_hfu /num_record_steps/ (avg_latency + 1e-12)
        print(
            f"Overall Stats: "
            f"batch_size_per_device: {self.num_samples / num_record_steps}, sequence_length: {self.max_seq_length}, dp_size: {self.dp_size}, "
            f"Latency: {avg_latency:.3f} s, Throughput: {avg_throughput:.3f} samples/sec, "
            # f"Latency: {avg_latency:.3f} s, Throughput: {avg_throughput:.3f} samples/sec, TGS: {tokens_per_gpu_per_second:.3f} tokens/s, "
            f"Model TFLOPS: {avg_stdit_tflops_per_gpu:.3f}, "
            f"TFLOPS per GPU (MFU): {avg_tflops_per_gpu:.3f}, TFLOPS per GPU (HFU): {avg_tflops_hfu_per_gpu:.3f},"
            f"\n"
        )

        # Time Breakdown Stats
        data_load_duration = all_reduce_mean(self.timer.data_load_duration, self.world_size)
        if self.pp_size == 1:
            forward_duration = all_reduce_mean(self.timer.forward_duration, self.world_size)
            backward_duration = all_reduce_mean(self.timer.backward_duration, self.world_size)
        else:
            forward_backward_duration = all_reduce_mean(self.timer.forward_backward_duration, self.world_size)
        optimizer_update_duration = all_reduce_mean(self.timer.optimizer_update_duration, self.world_size)

        time_usage_log = f"Time Usage Breakdown: "
        time_usage_log += f"Avg Dataload Latency: {1000 * data_load_duration / num_record_steps:.2f} ms, "
        if self.pp_size == 1:
            time_usage_log += f"Avg Forward Latency: {1000 * forward_duration / num_record_steps:.2f} ms, "
            time_usage_log += f"Avg Backward Latency: {1000 * backward_duration / num_record_steps:.2f} ms, "
        else:
            time_usage_log += f"Avg Forward Backward Latency: {1000 * forward_backward_duration / num_record_steps:.2f} ms, "
        if self.optimizer_update_cnt > 0:
            time_usage_log += f"Avg Optimizer Update Latency: {1000 * optimizer_update_duration / self.optimizer_update_cnt:.2f} ms, "
        time_usage_log += f"Avg Step Latency: {1000 * avg_latency:.2f} ms\n"
        print(time_usage_log)

        peak_memory = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        memory_fragmentation = torch.cuda.max_memory_reserved() - peak_memory
        final_allocated_memory = torch.cuda.memory_allocated()
        # print(f"final_allocated_memory {final_allocated_memory /1024**3:.3f} peak_memory {peak_memory /1024**3:.3f}")
        optimizer_memory = final_allocated_memory - self.total_weight_memory * 1024**3
        assert optimizer_memory >= self.optim_init_memory, "Optimizer memory should be larger than the initial memory"
        activation_memory = peak_memory - final_allocated_memory - self.grad_memory
        print(
            f"Memory Usage Breakdown: "
            f"Stdit Weight {self.stdit_weight_memory :.3f} GB,"
            f"Total Weight {self.total_weight_memory :.3f} GB, "
            f"Grad {self.grad_memory:.3f} GB, "
            f"Optimizer {optimizer_memory/1024**3:.3f} GB, "
            f"Activation {activation_memory/1024**3:.3f} GB, "
            f"Peak {peak_memory/1024**3:.3f} GB, "
            f"Frag {memory_fragmentation/1024**3:.3f} GB, "
            f"Final {final_allocated_memory/1024**3:.3f} GB, "
            # f"CPU {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
            f"CPU {dict(psutil.virtual_memory()._asdict())['used'] /1024**3:.3f} GB"
            F"\n"
        )
        # print(
        #     "Notice: Sometimes the weight and optimizer are initialized together (e.g. booster.boost/deepspeed.initialize), "
        #     "in such cases the calculated Weight memory is the sum of model weight memory and part of optimizer memory, please refer to other logs for model weight memory information."
        # )
        print(torch.cuda.memory_summary(device=torch.cuda.current_device()))
