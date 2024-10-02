
model_checkpoint = 'facebook/esm2_t30_150M_UR50D'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, padding = 'max_length', max_length = 1026)
# out = tokenizer('ATGCGTCATGS')
# print(out)


ForceGPT_model_name='lamm-mit/ProteinForceGPT'

seq = 'ATGCGTCATGS'

pfg_tokenizer = AutoTokenizer.from_pretrained(ForceGPT_model_name, trust_remote_code=True)
out = pfg_tokenizer(seq)
print(out)
print(pfg_tokenizer.tokenize(seq))
print(tokenizer.tokenize(seq))
