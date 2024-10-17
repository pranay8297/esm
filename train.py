import numpy as np
import os
import wandb

from torch.optim import AdamW

from esm_model import *
from utils import *
from settings import *

from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

train_entire_model = True # After inital finetune with LoRA, train the entire model for an epoch to finalize the results
checkpoint_path = None

tokens_for_grad_update = 30000 
epochs = 1
batch_size = 48
grad_accum_steps = 3

steps_processed_after_gradstep = 0
total_tokens_trained = 0
loss = None
losses = []

weight_decay = 0.1
starting_lr = 1e-06

# Epochs to skip
skip_step_percentage = 0.0#91068
# Saving related stuff
save_after_every = 250

torch.manual_seed(22) # Signature Number
if torch.cuda.is_available():
    torch.cuda.manual_seed(22)

# DDP
ddp = torch.cuda.device_count() > 1
ddp_args = {}

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{ddp_local_rank}')
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    ddp_args = {'ddp_world_size': ddp_world_size, 'ddp_rank': ddp_rank, 'is_ddp': True}
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = torch.device(f'cuda:{ddp_local_rank}') if torch.cuda.is_available() else torch.device('cpu') 
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.    

print(f'This is process: {ddp_rank}, Im the master process: {master_process}')
# This should only be on master process - WandB logging and checkpointing should be done on master process only
if master_process:
    wandb.login(key = WandB_API_KEY)
    run = wandb.init(
        project = 'esm_run_1', 
    )

# GPU Config
use_bfloat_16 = False
gpu_details = torch.cuda.get_device_name(0)

if 'A100' in gpu_details or 'A6000' in gpu_details: # bfloat16 is available only in Ampere series
    use_bfloat_16 = True

print(f'Using Bfloat16: {use_bfloat_16}, GPU name: {gpu_details}')

# Data and few Hyper Parameters
data_obj = get_dls(batch_size = batch_size, ddp_args = ddp_args)
train_dl = data_obj['train_dl']
valid_dl = data_obj['valid_dl']

# Model 
torch.set_float32_matmul_precision('high')

# Load the pre trained model if you have any
if checkpoint_path is not None and os.path.exists(checkpoint_path):
    if master_process: print(f'Loading from the checkpoint: {checkpoint_path}')
    model = torch.load('ep2_checkpoints/finetune_model_ckpt_36.pt')
else: # Else fork the model
    if master_process: print(f'Forking the model from the main ESM model')
    model = ESM.from_pretrained(LoRAConfig(lora_r = 32, lora_key = True, lora_mlp = True, lora_projection = True, lora_alpha = 16), 'esm2_t30_150M_UR50D')

if train_entire_model: 
    if master_process: print(f'Training the entire model')
    for name, param in model.named_parameters():
        param.requires_grad = True

model = model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# opt = AdamW(model.parameters(), lr = starting_lr, betas = (0.9, 0.95), eps = 1e-08, fused = True)
opt = raw_model.configure_optimizers(weight_decay = weight_decay, learning_rate = starting_lr)

 # This is unwrapped model used for checkpointing
iterations = epochs * len(train_dl) + 5

vl_losses_track = {0: -np.log(1/384)} # Ideal loss at epoch 0 before any finetuning - Equal probability for all the tokens in the vocab
vl_losses_all = []

# Use cosine anneling learning rate because the model is partially finetuned for this task and warmup phase is over
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations) 

# Use this if you are finetuning from direct ESM model else use Cosine Anneling LRS
# lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr = starting_lr, total_steps = iterations, final_div_factor=10.0)

checkpoint_counter = 1
save_counter = 0

wandb.config = {"epochs": epochs, "learning_rate": starting_lr, "batch_size": batch_size}
valid_iter = iter(valid_dl)

for i in range(epochs):
    c = 0
    for it, batch in enumerate(train_dl):
        
        progress = it/len(train_dl)
        if c >= 5: # approximately for every 400k tokens trained on, lets calculate the validation loss
            c = 0
            vl_losses = []
            model.eval()
            with torch.no_grad():
                for i in range(5):
                    try:
                        vl_batch = next(valid_iter)
                    except Exception as e:
                        valid_iter = iter(valid_dl)
                        vl_batch = next(valid_iter)
                    if use_bfloat_16:
                        with torch.autocast(device_type='cuda', dtype = torch.bfloat16):
                            vl_outputs = model(vl_batch['input_ids'].to(device), y = vl_batch['labels'].to(device), attention_mask = vl_batch['attention_mask'].to(device))
                    else:
                        vl_outputs = model(vl_batch['input_ids'].to(device), y = vl_batch['labels'].to(device), attention_mask = vl_batch['attention_mask'].to(device))
                    vl_loss = vl_outputs['loss'].detach()
                    if ddp:
                        dist.all_reduce(vl_loss, op=dist.ReduceOp.AVG)
                    vl_losses.append(vl_loss.item())
            # Set the model back to training
            model.train()

            if master_process:
                vl_losses_all += vl_losses
                vl_losses_track[total_tokens_trained.item()*2] = np.mean(vl_losses)
                print(f"valid loss: {vl_losses_all[-1]:.6f}")
                wandb.log({"valid_loss": np.mean(vl_losses)})

        if master_process and save_counter >= save_after_every:
            # saves the checkpoint for every 1000 grad updates
            torch.save(raw_model, f'checkpoints/finetune_model_ckpt_{checkpoint_counter}.pt')
            checkpoint_counter += 1
            save_counter = 0
        
        # Training Code
        if progress <= skip_step_percentage: continue # ensures that it does not train on data it already trained on

        if use_bfloat_16:
            with torch.autocast(device_type='cuda', dtype = torch.bfloat16): # if using an Ampere series GPU else use float32 - Ideally work on A100
                outputs = model(batch['input_ids'].to(device), y = batch['labels'].to(device), attention_mask = batch['attention_mask'].to(device))

        else: # Else do the training using fp32 precision only
            outputs = model(batch['input_ids'].to(device), y = batch['labels'].to(device), attention_mask = batch['attention_mask'].to(device))
        
        total_tokens_trained += batch['attention_mask'].sum()
        loss = outputs['loss'].detach()
        if ddp:
            #dist.all_reduce(total_tokens_trained, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        if master_process:
            losses.append(loss.item())
        
        outputs['loss'] = outputs['loss']/grad_accum_steps

        if ddp and steps_processed_after_gradstep == grad_accum_steps: # For gradiant accumulation
            model.require_backward_grad_sync = True

        outputs['loss'].backward()
        # scaler.scale(outputs['loss']).backward()
        steps_processed_after_gradstep += 1

        if steps_processed_after_gradstep == grad_accum_steps:
            # Do a backward pass and optimizer step
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # scaler.step(opt)
            # scaler.update()
            lr_scheduler.step()
            opt.zero_grad() # zero out the gradiants

            steps_processed_after_gradstep = 0
            c += 1
            save_counter += 1
            if master_process:
                print(f"Train Progress: {progress*100:.6f}%, train loss: {losses[-1]:.6f}, norm: {norm:.4f}")
                wandb.log({"train_loss": np.mean(losses[-grad_accum_steps:]), "progress": progress*100})
            model.require_backward_grad_sync = False # Avoid gradiant accumulation until we do next gradiant update

        if torch.cuda.is_available():
            torch.cuda.synchronize() # wait for the GPU to finish work

if master_process:
    torch.save(raw_model, f'checkpoints/esm_final_adopted.pt')

if ddp:
    dist.destroy_process_group()
# Next: TODO 
# Load the pretrained weights to this model - Done
# Add required Activation functions and all... make sure its the same forward as of original esm model's
# Implement Rotary Embeddings - Done
# Do a forward pass - verification process - Done
# Then work on Embeddings - Add new tokens - Keep the existing tokens - Turn on requires grad - Done, Verification - Done
# Get the training data - Done
# Test one forward pass, calculate loss, calculate gradiants, update parameters - Done
# Set up Lora for the model - Done
# write training script - Grad accumulation, batches, generate - Done
# Finetune - Hope for the best - Snowflake - Done - Results look promising

# setup topk = 10 (For a target vocab size of 20 + 7 + 1 -> 10 is a good topk) # This is for generate - Not for training - Load the pretrained model and do this  - Done
# Do benchmarking with prot GPT - Done

# set up WandB - Done
# Initialize the LoRA matrecis so that they match the activations of attention layers, intermediate, output layers - Done
# finally also set up ddp