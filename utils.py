import torch

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from esm_embeddings import PFGPT_HF_MODEL_PATH

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(PFGPT_HF_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

tokenizer = get_tokenizer()

class ProtDS(Dataset):
    def __init__(self, sequences, max_len = 1026):

        self.sequences = sequences # list object
        self.max_len = max_len
        self.eos_token = '</s>' # PFGPT's eos token - End of sequence
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        label = seq[1:] + self.eos_token
        
        return seq, label

def seq_collate_fn(data: list) -> dict:
    # data is a list of tuples
    x, y = zip(*data)
    train_tokenized = tokenizer(x)
    labels_tokenized = tokenizer(y)

    padded_train_tokenized = tokenizer.pad(train_tokenized, padding = 'max_length', max_length = 1026)
    padded_labels_tokenized = tokenizer.pad(labels_tokenized, padding = 'max_length', max_length = 1026)
    padded_train_tokenized['labels'] = padded_labels_tokenized['input_ids']

    for k, v in padded_train_tokenized.items():
        padded_train_tokenized[k] = torch.tensor(v, dtype = torch.long)

    return padded_train_tokenized

def get_dls(batch_size: int = 2) -> dict:
    ds = load_dataset("lamm-mit/GPTProteinPretrained")
    sequences = ds['train']['text'] # list
    train_seqs, valid_seqs = train_test_split(sequences, test_size = 0.1, shuffle = True)
    train_ds = ProtDS(train_seqs); valid_ds = ProtDS(valid_seqs)
    train_dl = DataLoader(train_ds, batch_size = batch_size, collate_fn = seq_collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size = batch_size, collate_fn = seq_collate_fn)

    return {"train_ds": train_ds, 
            "valid_ds": valid_ds, 
            "train_dl": train_dl, 
            "valid_dl": valid_dl}
