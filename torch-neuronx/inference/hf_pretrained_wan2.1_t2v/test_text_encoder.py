import torch
from transformers.models.umt5 import UMT5EncoderModel
from transformers import AutoTokenizer

model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DTYPE = torch.bfloat16
text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=DTYPE, cache_dir="wan2.1_t2v_14b_hf_cache_dir")
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir="wan2.1_t2v_14b_hf_cache_dir")

# print(text_encoder)

article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
input_ids = tokenizer(article, 
                    return_tensors="pt",
                    max_length=512,        # 最大长度
                    padding="max_length",  # 填充到最大长度
                    truncation=True        # 如果超过最大长度则截断
                    ).input_ids

# 创建 attention mask（如果需要）
attention_mask = (input_ids != tokenizer.pad_token_id).long()

outputs = text_encoder(input_ids)
hidden_state = outputs.last_hidden_state

print('hidden_state:', hidden_state.shape, hidden_state)
print('attention_mask:', attention_mask.shape, attention_mask)

# hidden_state: torch.Size([1, 512, 4096]) tensor([[[ 0.0018, -0.0620,  0.1611,  ...,  0.0005, -0.0391, -0.0635],
#          [ 0.0020,  0.0184, -0.0253,  ...,  0.0007, -0.0376, -0.0126],
#          [ 0.0021,  0.0247,  0.1240,  ...,  0.0004, -0.0232, -0.0032],
#          ...,
#          [ 0.0002, -0.0034, -0.0040,  ..., -0.0006,  0.0031, -0.0032],
#          [ 0.0014, -0.1621, -0.0304,  ..., -0.0009, -0.1055, -0.0405],
#          [ 0.0012,  0.0040,  0.0067,  ..., -0.0007,  0.0425, -0.0172]]],
#        dtype=torch.bfloat16, grad_fn=<MulBackward0>)

# attention_mask: torch.Size([1, 512]) tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0]])