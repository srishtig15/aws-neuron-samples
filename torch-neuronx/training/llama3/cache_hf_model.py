import transformers

tokenizer_path="~/llama3_tokenizer"
model_weights_path="~/llama3-8B_hf_weights"
model_id = "meta-llama/Meta-Llama-3-8B"

t = transformers.AutoTokenizer.from_pretrained(model_id)
t.save_pretrained(tokenizer_path)

m = transformers.AutoModelForCausalLM.from_pretrained(model_id)
m.save_pretrained(model_weights_path)
