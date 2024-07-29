# Define acceleration
num_workers = 4
dtype = "fp32"
# dtype = "fp16"
# dtype = "bf16"
grad_checkpoint = True


# Others
seed = 42
outputs = "outputs"
wandb = False

epochs =  1
log_every = 10
ckpt_every = 300
load = None

batch_size = 4
lr = 1e-5
grad_clip = 1.0
grad_accm = 2

steps = 4
num_ckpt_blocks = 0 # 0-18
