from torch.optim import AdamW

from esm_model import *
from utils import *

tokens_for_grad_update = 10000
epochs = 2
batch_size = 8
grad_accum_steps = 10

data_obj = get_dls()
train_dl = data_obj['train_dl']
valid_dl = data_obj['valid_dl']

tokenizer = get_tokenizer()
model = ESM.from_pretrained(LoRAConfig(lora_r = 32, lora_key = True, lora_mlp = True, lora_projection = True, lora_alpha = 16), 'esm2_t30_150M_UR50D')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

opt = AdamW(model.parameters(), lr = 3e-04, betas = (0.9, 0.95), eps = 1e-05)
iterations = epochs * len(train_dl) + 5
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr = 6e-04, total_steps = iterations, final_div_factor=10.0)

steps_processed_after_gradstep = 0
loss = None
losses = []
for i in range(epochs):
    for iter, batch in enumerate(train_dl):

        outputs = model(batch['input_ids'].to(device), y = batch['labels'].to(device), attention_mask = batch['attention_mask'].to(device))
        losses.append(outputs['loss'].item())
        outputs['loss'] = outputs['loss']/grad_accum_steps
        outputs['loss'].backward()
        steps_processed_after_gradstep += 1

        if steps_processed_after_gradstep == grad_accum_steps:
            # Do a backward pass and optimizer step
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr_scheduler.step()
            steps_processed_after_gradstep = 0
            print(f"Train Loss: {losses[-1]}")

    with torch.no_grad():
        # Do a validation pass for 20 epochs and calculate loss
        vl_loss = None
        for vl_batch in valid_dl:
            outputs = model(vl_batch['input_ids'].to(device), y = vl_batch['labels'].to(device), attention_mask = vl_batch['attention_mask'].to(device))
            
            if vl_loss is not None: vl_loss += outputs['loss']
            else: vl_loss = outputs['loss']
        print(f'Validation Loss: {vl_loss.item()//len(valid_dl)}')

torch.save(model, './model_finetune_v1.pt')


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

# setup topk = 10 (For a target vocab size of 20 + 7 + 1 -> 10 is a good topk) # This is for generate - Not for training - Load the pretrained model and do this 
# Do benchmarking with prot GPT