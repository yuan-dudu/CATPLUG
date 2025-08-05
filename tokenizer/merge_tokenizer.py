# -*- coding: utf-8 -*-

from transformers import AutoTokenizer

pre_add_vocab_file = "/mnt/data/zhenmengyuan/LLaMA-Factory/saves/qwen2.5/ko-en/ko_pre_32000.txt"


tokenizer = AutoTokenizer.from_pretrained("/mnt/data/zhenmengyuan/LLaMA-Factory/saves/qwen2.5/qwen2.5-3b/",local_files_only=True, trust_remote_code=True)
print(f"before add vocab vocab length {len(tokenizer)}")

with open(pre_add_vocab_file, "r", encoding="utf-8") as f:
    new_tokens = [line.strip() for line in f.readlines()]  


existing_tokens = set(tokenizer.vocab.keys())
new_tokens = set(new_tokens) - existing_tokens

num_add_tokens = tokenizer.add_tokens(list(new_tokens))

print(f"after add vocab tokenizer_length:{len(tokenizer)}")

tokenizer.save_pretrained("/mnt/data/zhenmengyuan/LLaMA-Factory/saves/qwen2.5/ko-en/qwen-ko/")

print(f"new add  {num_add_tokens} commen Token to vocab ")