import math
import torch

from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from einops import rearrange, repeat

from esm_embeddings import ESMEmbeddings
from rotary_utils import *
from utils import get_tokenizer

@dataclass
class LoRAConfig:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: int = 0.05
    lora_query: bool = True
    lora_key: bool = False
    lora_value: bool = True
    lora_projection: bool = False
    lora_mlp: bool = False
    lora_head: bool = False

class LoRALinear(nn.Linear):
    def __init__(self, nin, nout, lora_config):
        super().__init__(nin, nout)
        std_dev = 1 / torch.sqrt(torch.tensor(lora_config.lora_r).float())
        self.lora_A = torch.nn.Parameter(torch.randn(nin, lora_config.lora_r) * std_dev)
        self.lora_B = torch.nn.Parameter(torch.zeros(lora_config.lora_r, nout))
        self.alpha = lora_config.lora_alpha
    
    def forward(self, x):
        lora_x = self.alpha * (x @ self.lora_A @ self.lora_B)
        x = super().forward(x)
        return x + lora_x

@dataclass
class ESMConfig(): 
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    hidden_size: int = 4096 # 4 * block_size
    dropout: float = 0.0
    pre_trained_model_name: str = ''

class ESMIntermediateLayer(nn.Module):
    def __init__(self, nin, nout, lora_config, dropout = 0.0, ):
        super().__init__()
        
        self.dense = nn.Linear(nin, nout) if not lora_config.lora_mlp else LoRALinear(nin, nout, lora_config)
        self.act = nn.GELU(approximate = 'tanh')

    def forward(self, x):
        return self.act(self.dense(x))

class ESMOutLayer(nn.Module):
    def __init__(self, nin, nout, lora_config, dropout = 0.0, inside_attention = False):
        super().__init__()
        
        # 2 places used - 1. inisde the attention block  and inside the MLP
        # if used inside attention and lora_config.lora_projection is true then dense is a LoRALInear
        # elif used in mlp and lora_config.lora_mlp is true then dens is a LoRALinear again
        # else its a Linear

        if inside_attention == True and lora_config.lora_projection:
            self.dense = LoRALinear(nin, nout, lora_config)
        elif inside_attention == False and lora_config.lora_mlp: 
            self.dense = LoRALinear(nin, nout, lora_config)
        else:
            self.dense = nn.Linear(nin, nout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_scores):
        x = self.dense(x)
        x = self.dropout(x)
        x = x + attn_scores
        return x

class ESMSelfAttn(nn.Module): # Verified
    
    def __init__(self, config, lora_config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.query = nn.Linear(config.n_embd, config.n_embd) if not lora_config.lora_query else LoRALinear(config.n_embd, config.n_embd, lora_config)
        self.key = nn.Linear(config.n_embd, config.n_embd) if not lora_config.lora_key else LoRALinear(config.n_embd, config.n_embd, lora_config)
        self.value = nn.Linear(config.n_embd, config.n_embd) if not lora_config.lora_value else LoRALinear(config.n_embd, config.n_embd, lora_config)
        self.n_head = config.n_head

        attention_head_size = config.n_embd//config.n_head

        # Add a rotary embeddings here
        self.rotary_embeddings = RotaryEmbedding(dim = attention_head_size)
     
    def forward(self, x, attention_mask): 

        # x -> (b, s, e) -> (b s, h, e/h)
        k, q, v = self.key(x), self.query(x), self.value(x)

        k = rearrange(k, 'b s (h e) -> b h s e', h = self.n_head)
        q = rearrange(q, 'b s (h e) -> b h s e', h = self.n_head)
        v = rearrange(v, 'b s (h e) -> b h s e', h = self.n_head)

        # Add rotary embeddings here for k and q tensors
        q, k = self.rotary_embeddings(q, k)

        # Attention claculation - # TODO: make is_casual true in case of finetuning - Very important
        y = F.scaled_dot_product_attention(q, k, v, attn_mask = attention_mask, is_causal = True) # flash attention 
        y = rearrange(y, 'b h s e -> b s (h e)', h = self.n_head)
        return y

class ESMAttn(nn.Module): # Verified

    def __init__(self, config, lora_config):
        super().__init__() # No activation function at this level
        self.self = ESMSelfAttn(config, lora_config)
        self.output = ESMOutLayer(config.n_embd, config.n_embd, lora_config, dropout = getattr(config, 'dropout', 0.), inside_attention = True)
        self.LayerNorm = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, attention_mask):
        inter_x = self.LayerNorm(x)
        attn = self.self(inter_x, attention_mask)
        out = self.output(attn, x)
        return out

class ESMLayers(nn.Module): # Both Init and Forward Verified - Done and Dusted

    def __init__(self, config, lora_config):
        super().__init__()
        self.attention = ESMAttn(config, lora_config) 
        self.intermediate = ESMIntermediateLayer(config.n_embd, config.hidden_size, lora_config) # 
        self.output = ESMOutLayer(config.hidden_size, config.n_embd, lora_config) #
        self.LayerNorm = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, attention_mask):
        attention_op = self.attention(x, attention_mask)
        attention_op_ln = self.LayerNorm(attention_op) # This will keep the activations in check - Lets see
        inter = self.intermediate(attention_op_ln)
        out = self.output(inter, attention_op)
        return out

class ESMEncoder(nn.Module):

    def __init__(self, config, lora_config):
        super().__init__() 
        
        # No activation functions here as well

        self.layer = nn.ModuleList([ESMLayers(config, lora_config) for _ in range(config.n_layer)])
        self.emb_layer_norm_after = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, attention_mask = None): 

        for layer in self.layer:
            x = layer(x, attention_mask)

        return self.emb_layer_norm_after(x)

class ESM(nn.Module):
    
    def __init__(self, config, lora_config):
        
        super().__init__()
        self.config = config
        self.esm = nn.ModuleDict(dict(
            embeddings = ESMEmbeddings(config),
            encoder = ESMEncoder(config, lora_config), # Done, forward - here
            final_layer = nn.Linear(config.n_embd, config.vocab_size) if not lora_config.lora_head else 
                              LoRALinear(config.n_embd, config.vocab_size, lora_config)
        ))
        self.esm.final_layer.weight = self.esm.embeddings.word_embeddings.weight
        
        # Final Layer bias initializtion
        torch.nn.init.zeros_(self.esm.final_layer.bias) # Set the bias to 0
        # Finally one small thing is to decide weather to add an intermediate layer or not? - Thats a future discussion
    
    @classmethod
    def get_pretrained_config(cls, model_type = 'esm2_t33_650M_UR50D'):

        '''
        name                n_layers    n_params  
        esm2_t48_15B_UR50D	48	        15B
        esm2_t36_3B_UR50D	36	        3B
        esm2_t33_650M_UR50D	33	        650M
        esm2_t30_150M_UR50D	30	        150M
        esm2_t12_35M_UR50D	12	        35M
        esm2_t6_8M_UR50D
        '''

        assert model_type in {'esm2_t36_3B_UR50D', 'esm2_t33_650M_UR50D', 'esm2_t30_150M_UR50D'}

        config_args = {
            'esm2_t36_3B_UR50D': dict(n_layer=36, n_head = 40, n_embd=2560, hidden_size=10240), # 3B params
            'esm2_t33_650M_UR50D': dict(n_layer=33, n_head = 20, n_embd=1280, hidden_size=5120), # 650M params
            'esm2_t30_150M_UR50D': dict(n_layer=30, n_head = 20, n_embd=640, hidden_size=2560), # 150M params
        }[model_type]

        config_args['vocab_size'] = 33 # always 33 for ESM Models
        config_args['block_size'] = 1026 # Always constant for ESM Models
        config_args['pre_trained_model_name'] = f"facebook/{model_type}"

        config = ESMConfig(**config_args)
        return config

    @classmethod
    def from_pretrained(cls, lora_config, model_type = 'esm2_t33_650M_UR50D', embedding_post_init = True):

        config = cls.get_pretrained_config(model_type)
        print("loading weights from pretrained gpt: %s" % model_type)

        # create a from-scratch initialized minGPT model
        model = cls(config, lora_config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        from transformers import AutoModelForSequenceClassification
        num_labels = 33
        model_hf = AutoModelForSequenceClassification.from_pretrained(config.pre_trained_model_name, num_labels = num_labels)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        sd_keys_hf = [k for k in sd_keys_hf if 'inv_freq' not in k]
        sd_keys_hf = [k for k in sd_keys_hf if 'classifier' not in k]

        ignore_keys = ['esm.contact_head.regression.weight', 'esm.contact_head.regression.bias']
        for k in sd_keys_hf:
            
            if k in ignore_keys: continue

            # vanilla copy over the other parameters
            try: assert sd_hf[k].shape == sd[k].shape
            except Exception as e: 
              print(k)
              print(f"Mismatch in the shape of tensor while loading weights - Key: {k}, expected shape: {sd_hf[k].shape}, actual shape: {sd[k].shape if k in sd else k}")
            
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        # Set the final layers bias as 0 so that it does not affect weight tying scheme
        with torch.no_grad():
            model.esm.final_layer.bias.zero_()
        
        # Freeze the model
        for name, param in model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

        if embedding_post_init: 
            model.esm.embeddings.post_model_init()
            del model.esm.final_layer

            #IMP: Here we are assuming that embeddings will never have LoRA attached to it, hence we are going with Linear
            model.esm.final_layer = nn.Linear(config.n_embd, model.esm.embeddings.word_embeddings.weight.shape[0])
            
            model.esm.final_layer.weight = model.esm.embeddings.word_embeddings.weight

        return model

    def get_extended_attn_mask(self, attention_mask, input_shape):
        
        if attention_mask == None: return None
        b, s = attention_mask.shape
        # Make the attention mask braodcastable for [batch_size, n_heads, seq_len, seq_len]
        attention_mask = attention_mask[:, None, None, :]      

        # Now make sure that it has negetive infinity for all the padded tokens and 
        # 0 for all attention tokens as we add this mask to attention scores
        attn_mask = attention_mask.to(torch.float32)
        attn_mask = (1 - attn_mask) * (torch.finfo(torch.float32).min)
        attn_mask = attn_mask.expand(b, 1, s, s)
        return attn_mask

    def forward(self, x, y = None, attention_mask = None, output_encoder_states = True):

        # Calculate Embeddings
        x = self.esm.embeddings(x, attention_mask) # TODO: Verify the new embeddings function without doing post init and after doing post model init - Ideally both should stay the same

        # compute attention_mask for attention scores
        extended_attention_mask = self.get_extended_attn_mask(attention_mask, x.shape)

        #Do the forward pass
        x = self.esm.encoder(x, attention_mask = extended_attention_mask)
        logits = self.esm.final_layer(x)
        output = {'logits': logits}

        if output_encoder_states:
            output['encoder_output'] = x
        if y is not None: 
            # Calculate loss and send it in output
            outputs = logits.view(-1, logits.size(-1))  # (bs*seq_len, 384)
            targets = y.view(-1)  # (bs*seq_len)
            
            # Flatten the attention mask
            attention_mask = attention_mask.view(-1)  # (bs*seq_len)
            
            # Calculate cross entropy loss
            loss = F.cross_entropy(outputs, targets, reduction='none')
            
            # Apply the mask to the loss
            masked_loss = loss * attention_mask
            
            # Calculate the mean loss over the actual tokens (excluding padding)
            total_loss = masked_loss.sum()
            num_tokens = attention_mask.sum()
            
            actual_loss = total_loss / num_tokens
            output['loss'] = actual_loss

        return output