import torch
import torch.nn as nn
from transformers import AutoTokenizer
PFGPT_VOCAB_SIZE = 384
PFGPT_HF_MODEL_PATH = 'lamm-mit/ProteinForceGPT'

class ESMEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)
    
    def post_model_init(self):
        # Merge both the tokenizer vocabs - Battle of Tokenizers
        # That is create a new word embedding of pf_gpts vocab size and configs n_embd
        # Get pf GPT Tokenizer
        pfgpt_tokenizer = AutoTokenizer.from_pretrained(PFGPT_HF_MODEL_PATH, trust_remote_code=True)
        pfgpt_tokenizer.pad_token = pfgpt_tokenizer.eos_token
        pfgpt_vocab = pfgpt_tokenizer.get_vocab()

        # Get ESM Tokenizer
        esm_tokenizer = AutoTokenizer.from_pretrained(self.config.pre_trained_model_name, padding='max_length', max_length=1026)
        esm_vocab = esm_tokenizer.get_vocab()
        new_word_embeddings = nn.Embedding(PFGPT_VOCAB_SIZE, self.config.n_embd)
        torch.nn.init.normal_(new_word_embeddings.weight, std = 0.1263) # This is the std of esm embeddings - As we are using weights of that model, its better to use this std for effective training

        # Find all the common keys tokens between esm tokenizer and pf_gpt tokenizer
        pfgpt_keys = set(pfgpt_vocab.keys())
        esm_keys = set(esm_vocab.keys())
        common_keys = list(pfgpt_keys.intersection(esm_keys))

        # now, copy a particular tokens embedding from ems_embedding to the new embedding that we create here
        with torch.no_grad():
            indices = []
            for key in common_keys:
                esm_embd_index = esm_tokenizer.convert_tokens_to_ids(key)
                pfg_embd_index = pfgpt_tokenizer.convert_tokens_to_ids(key)
                indices.append(pfg_embd_index)
                new_word_embeddings.weight[pfg_embd_index] = self.word_embeddings.weight[esm_embd_index]

            # Check for embedding equivalance
            assert torch.equal(new_word_embeddings.weight[pfgpt_tokenizer.convert_tokens_to_ids(common_keys)], 
                               self.word_embeddings.weight[esm_tokenizer.convert_tokens_to_ids(common_keys)])

        # Create a mask for all the indecis we have copied pretrained embeddings 
        # and turn requires_grad off to those embeddings that we have copied - This is
        # not possible, so instead we store the indecis and zero out the grads before optim.step()
        # hence we do not update these embeddings
        self.indices = indices

        with torch.no_grad():
            self.word_embeddings = new_word_embeddings
        self.word_embeddings.requires_grad_(True)

    def forward(self, x, attention_mask = None):
        token_embs = self.word_embeddings(x)
        position_embs = self.position_embeddings
        pos = torch.arange(0, x.shape[1], dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.position_embeddings(pos)
        token_embs = token_embs + pos_emb
        # Not required as we are use rotary embeddings - Hence we do not require absolute position embeddings
        # position_embs = self.esm.embeddings.position_embeddings(torch.arange(0, x.shape[1], 1, dtype = torch.long)) 
        if attention_mask is not None: 
            token_embs = (token_embs * attention_mask.unsqueeze(-1)).to(token_embs.dtype)
        return token_embs