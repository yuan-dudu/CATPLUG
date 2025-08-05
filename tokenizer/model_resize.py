# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch.nn as nn


# model_name_or_path = "/mnt/data/zhenmengyuan/LLaMA-Factory/saves/qwen2.5/qwen2.5-3b/"  
# new_tokenizer_path = "/mnt/data/zhenmengyuan/LLaMA-Factory/saves/qwen2.5/ko-en/qwen-ko/"  
# output_model_path = "/mnt/data/zhenmengyuan/LLaMA-Factory/saves/qwen2.5/ko-en/qwen-ko/"  


# model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_path)


# new_vocab_size = len(tokenizer)

# #调整模型的词向量层（Embedding Layer）大小，以匹配新的词表
# model.resize_token_embeddings(new_vocab_size)


# #语言模型的输出层（线性映射）需要与新词表大小一致，因此重新初始化该层。
# model.lm_head = nn.Linear(
#     in_features=model.lm_head.in_features,
#     out_features=new_vocab_size,
#     bias=False
# )

# #使用 Xavier 均匀分布初始化新添加的 lm_head 权重，确保模型训练稳定性。
# nn.init.xavier_uniform_(model.lm_head.weight)


# model.save_pretrained(output_model_path)
# tokenizer.save_pretrained(output_model_path)

# print(f"Modified model saved to {output_model_path}")


# model_path = output_model_path
# model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)


# print("\nModel Parameters:")
# print(model)


# embedding_layer = model.get_input_embeddings()
# print("\nEmbedding Layer:")
# print(f"Embedding Layer Shape: {embedding_layer.weight.shape}")


# lm_head_layer = model.lm_head
# print("\nlm_head Layer:")
# print(f"lm_head Layer Shape: {lm_head_layer.weight.shape}")


# print(f"\nVocabulary Size: {new_vocab_size}")



#####测试分词效果
from transformers import AutoModelForCausalLM, AutoTokenizer

# 定义模型路径
original_model_path = "/mnt/data/zhenmengyuan/LLaMA-Factory/saves/qwen2.5/qwen2.5-3b/"
expanded_model_path = "/mnt/data/zhenmengyuan/LLaMA-Factory/saves/qwen2.5/ti-zh/qwen-ti/"

# 定义测试用的藏文句子（示例句子，可替换为你需要的藏文文本）
test_tibetan_text = "ང་དེ་རིང་ལས་ཀ་བྱེད་རྒྱུ་མིན་པས་ཧ་ཅང་དགའ་པོ་བྱུང་།"  # 可替换为任意藏文句子

def test_tokenization(model_path, tokenizer_path, label):
    """加载模型和分词器，测试对藏文文本的分词效果"""
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # 对测试文本进行分词
    tokens = tokenizer.tokenize(test_tibetan_text)
    token_ids = tokenizer.encode(test_tibetan_text)
    
    # 打印结果
    print(f"\n{label} 模型分词结果:")
    print(f"原始文本: {test_tibetan_text}")
    print(f"分词结果: {tokens}")
    print(f"分词ID: {token_ids}")
    print(f"分词数量: {len(tokens)}")
    print(f"词表大小: {len(tokenizer)}")
    
    return tokens, token_ids

# 测试原始模型
original_tokens, original_ids = test_tokenization(
    original_model_path, 
    original_model_path, 
    "原始词表"
)

# 测试扩充词表后的模型
expanded_tokens, expanded_ids = test_tokenization(
    expanded_model_path, 
    expanded_model_path, 
    "扩充词表"
)

# 对比分词结果
print("\n===== 分词结果对比 =====")
print(f"原始词表分词数量: {len(original_tokens)}")
print(f"扩充词表分词数量: {len(expanded_tokens)}")

if len(original_tokens) > len(expanded_tokens):
    print("→ 扩充词表后分词数量减少，说明分词更高效（更少的token表示相同内容）")
elif len(original_tokens) < len(expanded_tokens):
    print("→ 扩充词表后分词数量增加，可能需要检查分词器配置")
else:
    print("→ 分词数量相同，可能需要进一步分析分词质量")

# 打印详细对比（只显示前10个token）
print("\n详细对比（前10个token）:")
print("{:<20} {:<20}".format("原始词表", "扩充词表"))
print("-" * 40)
for i in range(max(len(original_tokens), len(expanded_tokens))):
    orig_token = original_tokens[i] if i < len(original_tokens) else ""
    exp_token = expanded_tokens[i] if i < len(expanded_tokens) else ""
    print("{:<20} {:<20}".format(orig_token, exp_token))