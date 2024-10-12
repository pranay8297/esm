import numpy as np
import wandb

from torch.optim import AdamW

from esm_model import *
from utils import *
from settings import *

wandb.login(key = WandB_API_KEY)
run = wandb.init(
    project = 'esm_run_1', 
)

model = ESM.from_pretrained(LoRAConfig(lora_r = 32, lora_key = True, lora_mlp = True, lora_projection = True, lora_alpha = 16), 'esm2_t30_150M_UR50D')

# Load the pre trained model if you have any
# model = torch.load('model_finetune_v1.pt')

use_bfloat_16 = False
gpu_details = torch.cuda.get_device_name(0)
if 'A100' in gpu_details or 'A6000' in gpu_details: # bfloat16 is available only in Ampere series
    use_bfloat_16 = True

data_obj = get_dls()
train_dl = data_obj['train_dl']
valid_dl = data_obj['valid_dl']

tokens_for_grad_update = 30000 
epochs = 1
batch_size = 8
grad_accum_steps = 5
iterations = epochs * len(train_dl) + 5

steps_processed_after_gradstep = 0
total_tokens_trained = 0
loss = None
losses = []

vl_losses_track = {0: -np.log(1/384)} # Ideal loss at epoch 0 before any finetuning - Equal probability for all the tokens in the vocab
vl_losses_all = []

starting_lr = 1e-05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt = AdamW(model.parameters(), lr = starting_lr, betas = (0.9, 0.95), eps = 1e-05)
# Use cosine anneling learning rate because the model is partially finetuned for this task and warmup phase is over
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations) 

# Use this if you are finetuning from direct ESM model else use Cosine Anneling LRS
# lr_scheduler = lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr = 6e-04, total_steps = iterations, final_div_factor=10.0) 

skip_step_percentage = 0.091068
# scaler = torch.amp.GradScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = torch.compile(model)

# Saving related stuff
save_after_every = 1000
checkpoint_counter = 1
save_counter = 0

wandb.config = {"epochs": epochs, "learning_rate": starting_lr, "batch_size": batch_size}

for i in range(epochs):
    c = 0
    for iter, batch in enumerate(train_dl):

        progress = iter/len(train_dl)
        if progress < skip_step_percentage: continue # ensures that it does not train on data it already trained on

        if use_bfloat_16:
            with torch.autocast(device_type='cuda', dtype = torch.bfloat16): # if using an Ampere series GPU else use float32 - Ideally work on A100
                outputs = model(batch['input_ids'].to(device), y = batch['labels'].to(device), attention_mask = batch['attention_mask'].to(device))
        else: # Else do the training using fp32 precision only
            outputs = model(batch['input_ids'].to(device), y = batch['labels'].to(device), attention_mask = batch['attention_mask'].to(device))
        
        total_tokens_trained += batch['attention_mask'].sum()

        losses.append(outputs['loss'].item())
        outputs['loss'] = outputs['loss']/grad_accum_steps
        outputs['loss'].backward()
        # scaler.scale(outputs['loss']).backward()
        steps_processed_after_gradstep += 1
        losses.append(outputs['loss'].item())

        if steps_processed_after_gradstep == grad_accum_steps:
            # Do a backward pass and optimizer step
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # scaler.step(opt)
            # scaler.update()
            lr_scheduler.step()
            opt.zero_grad() # zero out the gradiants

            steps_processed_after_gradstep = 0
            print(f"Train Progress: {progress*100:.6f}%, train loss: {losses[-1]:.6f}, norm: {norm:.4f}")
            c += 1
            save_counter += 1
            wandb.log({"train_loss": np.mean(losses[-grad_accum_steps:]), "progress": progress*100})

        if c >= 5: # approximately for every 400k tokens trained on, lets calculate the validation loss
            c = 0
            vl_losses = []
            v_iter = 0
            with torch.no_grad():
                for vl_batch in valid_dl: # TODO: Verify weather the validation batches are different for every validation run

                    vl_outputs = model(vl_batch['input_ids'].to(device), y = vl_batch['labels'].to(device), attention_mask = vl_batch['attention_mask'].to(device))
                    vl_losses.append(vl_outputs['loss'].item())
                    v_iter += 1

                    if v_iter%10 == 0: break

            vl_losses_all += vl_losses
            vl_losses_track[total_tokens_trained.item()] = np.mean(vl_losses)
            print(f"valid loss: {vl_losses_all[-1]:.6f}")
            wandb.log({"valid_loss": np.mean(vl_losses)})

        if save_counter >= save_after_every:
            # saves the checkpoint for every 1000 grad updates
            torch.save(f'checkpoints/finetune_model_ckpt_{checkpoint_counter}.pt')
            checkpoint_counter += 1
            save_counter = 0

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
# finally also set up ddp