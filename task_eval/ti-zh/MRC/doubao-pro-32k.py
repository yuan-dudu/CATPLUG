import json
import openai
import os
import time
import unicodedata
import re
from tqdm import tqdm
from collections import Counter
from openai import OpenAI # 导入OpenAI
from sentence_transformers import SentenceTransformer, util  # 语义相似度计算
import botok
from botok import WordTokenizer
from botok.config import Config
from pathlib import Path

# ✅ 加载 LaBSE 模型
labse_model = SentenceTransformer("/home/zhenmengyuan/.cache/modelscope/hub/models/sentence-transformers/LaBSE/")

# 文件路径
data_file = "task_MRC.json"
output_results_file = "doubao-pro-32k_outputs.json"
eval_results_file = "doubao-pro-32k_results.json"

# 加载数据集
with open(data_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 读取已有结果（支持断点续跑）
if os.path.exists(output_results_file):
    with open(output_results_file, "r", encoding="utf-8") as f:
        existing_results = json.load(f)
else:
    existing_results = []

# **去重：跳过已处理的数据**
processed_inputs = {item["input"] for item in existing_results}
dataset = [item for item in dataset if item["input"] not in processed_inputs]

# ✅ OpenAI API 客户端（适配新版 OpenAI SDK）
client = OpenAI(base_url = "https://ark.cn-beijing.volces.com/api/v3",
                api_key  = "aa1ce03a-6b61-46fc-b914-73fae89921f3")
# **OpenAI API 调用**
def get_doubao_response(instruction, input_text, max_retries=3):
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "你是一个擅长文本阅读理解的助手"},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"}
        ]
        try:
            response = client.chat.completions.create(
                model="doubao-pro-32k-241215",
                messages=messages,
                temperature=0.8  # 低温度，保证稳定回答
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
            if attempt == max_retries - 1:
                return "ERROR"
            time.sleep(2)  # 等待2秒后重试

# ✅ 计算语义相似度
def compute_semantic_similarity(pred, gold, model):
    embeddings = labse_model.encode([pred, gold], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

# ✅ 清理 & 规范化文本
def clean_tibetan_text(text):
    text = unicodedata.normalize("NFC", text)  # 规范化 Unicode
    text = re.sub(r"\s+", "", text)  # 仅移除空格
    return text
# ✅ 使用botok进行分词
config = Config(dialect_name="general", base_path=Path.home())
tokenizer = WordTokenizer(config=config)
def tibetan_syllable_segment(text):
    """使用 Botok 进行藏文分词"""
    tokens = tokenizer.tokenize(text, split_affixes=False)
    return " ".join([token.text for token in tokens])  # 返回空格分隔的分词结果

# ✅ 计算 EM（完全匹配 + 语义相似度优化）
def compute_em(pred, gold):
    # 如果预测文本不含藏文字符，直接判定为不匹配
    if not re.search(r"[ༀ-ྼ0-9]", pred) or pred == "ERROR":
        return 0.0
    pred_clean = clean_tibetan_text(pred)
    gold_clean = clean_tibetan_text(gold)

    # 1️⃣ **完全匹配**
    if pred_clean == gold_clean:
        return 1.0

    # 2️⃣ 包含关系（需确保都不为空）
    if pred_clean and gold_clean and (gold_clean in pred_clean or pred_clean in gold_clean):
        return 1.0

    # 3️⃣ **计算语义相似度**
    similarity_score = compute_semantic_similarity(pred, gold, "labse")

    # 4️⃣ **如果语义相似度 > 0.95 也算匹配**
    if similarity_score > 0.9:
        return 1.0

    return 0.0  # ❌ 其他情况不匹配


# ✅ 计算 F1 Score
def compute_f1(pred, gold):
    pred_tokens = tibetan_syllable_segment(pred).split()
    gold_tokens = tibetan_syllable_segment(gold).split()

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # 计算 TP（预测正确的词数）
    common_tokens = pred_counter & gold_counter
    N_TP = sum(common_tokens.values())

    # 计算 FP（错误预测的词数）
    N_FP = sum(pred_counter.values()) - N_TP

    # 计算 FN（标准答案中遗漏的词数）
    N_FN = sum(gold_counter.values()) - N_TP

    # 避免除以零
    if N_TP == 0:
        return 0.0

    # 计算 Precision 和 Recall
    precision = N_TP / (N_TP + N_FP)
    recall = N_TP / (N_TP + N_FN)
    if precision + recall == 0:
        return 0.0
    # 计算 F1 Score
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# 🚀 **运行模型 & 评估**
results = existing_results  # 加载已完成的结果
eval_scores = []

# **进度条**
for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
    instruction = item["instruction"]
    input_text = item["input"]
    gold_answer = item["output"]

    # **调用 GPT-4o 生成答案**
    model_output = get_doubao_response(instruction, input_text)

    # **计算评估指标**
    em_score = compute_em(model_output, gold_answer)
    f1_score = compute_f1(model_output, gold_answer)
    similarity = compute_semantic_similarity(model_output, gold_answer, "labse")

    # **保存结果**
    results.append({
        "instruction": instruction,
        "input": input_text,
        "gold_answer": gold_answer,
        "model_output": model_output,
        "similarity": similarity,
        "em": em_score,
        "f1": f1_score
    })

    eval_scores.append({
        "similarity": similarity,
        "em": em_score,
        "f1": f1_score
    })

    # **每 50 条数据实时保存一次**
    if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
        with open(output_results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"✅ {i+1}/{len(dataset)} 条数据已保存至 {output_results_file}")

# **计算平均评估指标**
eval_scores = [item for item in results if item["model_output"] != "ERROR"]
num_samples = len(eval_scores)
avg_scores = {
    "EM": sum(item["em"] for item in eval_scores) / num_samples,
    "F1": sum(item["f1"] for item in eval_scores) / num_samples
}

# **保存评估结果**
with open(eval_results_file, "w", encoding="utf-8") as f:
    json.dump(avg_scores, f, ensure_ascii=False, indent=4)

print(f"📊 评估结果已保存至: {eval_results_file}")



###统计中文回答数据
# import json
# import re
#
# # 加载 JSON 文件
# with open('doubao-pro-32k_outputs.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 统计包含中文的 model_output 数据条数
# chinese_count = 0
#
# for item in data:
#     model_output = item['model_output']
#     # 检查是否包含中文（Unicode 范围 \u4e00-\u9fff）
#     if re.search(r"[\u4e00-\u9fff]", model_output):
#         chinese_count += 1
#         print(model_output)
#
# # 打印结果
# print(f"包含中文的数据条数: {chinese_count}")



#
# import json
#
# # 文件路径
# input_file = "doubao-pro-32k_outputs.json"
# output_file = "doubao-pro-32k__results2.json"
#
# # 读取 gpt-4o_outputs.json
# try:
#     with open(input_file, "r", encoding="utf-8") as f:
#         data = json.load(f)
# except FileNotFoundError:
#     print(f"❌ 输入文件 {input_file} 不存在")
#     exit(1)
# except json.JSONDecodeError:
#     print(f"❌ 输入文件 {input_file} 格式错误")
#     exit(1)
#
# # 提取 em 和 f1 分数
# eval_scores = [{"em": item["em"], "f1": item["f1"]} for item in data]
#
# # 计算数据量和均值
# num_samples = len(eval_scores)
# avg_scores = {
#     "EM": sum(item["em"] for item in eval_scores) / num_samples if num_samples > 0 else 0.0,
#     "F1": sum(item["f1"] for item in eval_scores) / num_samples if num_samples > 0 else 0.0
# }
#
# # 保存平均评估结果
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(avg_scores, f, ensure_ascii=False, indent=4)
#
# print(f"📊 评估结果已保存至: {output_file}")