# # -*- coding: utf-8 -*-

import sentencepiece as spm
import os
import re

# 定义输入文件路径
input_file = "/mnt/data/zhenmengyuan/LLaMA-Factory/data/ko_data/ko.txt"

# 过滤后的临时文件路径
filtered_input_file = "/mnt/data/zhenmengyuan/LLaMA-Factory/data/ko_data/ko_filtered.txt"

# 设置不同的 vocab_size（示例）
vocab_sizes = [32000]  

# 确保输入文件存在且不为空
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file {input_file} does not exist.")
if os.path.getsize(input_file) == 0:
    raise ValueError(f"Input file {input_file} is empty.")

# 预处理函数：过滤英文和数字
def filter_non_chinese_or_other_chars(line):
    # 使用正则保留非英文、非数字字符（包括藏文、标点、空格等）
    return re.sub(r'[a-zA-Z0-9]', '', line)

# 执行预处理，生成干净数据
with open(input_file, "r", encoding="utf-8") as fin, \
     open(filtered_input_file, "w", encoding="utf-8") as fout:

    for line in fin:
        filtered_line = filter_non_chinese_or_other_chars(line)
        if filtered_line.strip():  # 如果不是空行再写入
            fout.write(filtered_line)

print(f"Filtered input saved to {filtered_input_file}")

# 遍历每个 vocab_size 执行训练和处理
for vocab_size in vocab_sizes:
    print(f"\nProcessing with vocab_size = {vocab_size} ...")

    # 构建 model_prefix 和 output_vocab_file
    model_prefix = f"LLaMA-Factory/saves/qwen2.5/ko-en/ko_{vocab_size}"
    output_vocab_file = f"LLaMA-Factory/saves/qwen2.5/ko-en/ko_pre_{vocab_size}.txt"
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    os.makedirs(os.path.dirname(output_vocab_file), exist_ok=True)

    # 1. 使用过滤后的语料训练BPE模型
    spm.SentencePieceTrainer.train(
        input=filtered_input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.995,
        model_type="bpe",
        bos_id=-1,
        eos_id=-1,
        unk_id=0,
        pad_id=1,
        user_defined_symbols=["<pad>"],
        max_sentence_length=4096,

        # 关键参数：避免数字被单独切分
        split_by_number=False,
        split_digits=False,
    )

    # 2. 加载训练好的SentencePiece模型
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    # 3. 读取生成的.vocab文件并处理
    vocab_path = f"{model_prefix}.vocab"
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file {vocab_path} does not exist.")

    tokens = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            token, _ = line.strip().split("\t")
            tokens.append(token)

    # 4. 将处理后的词表写入新的文件
    with open(output_vocab_file, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(f"{token}\n")

    print(f"Saved vocab to {output_vocab_file}")

print("All vocab sizes processed.")

