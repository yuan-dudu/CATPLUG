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
# import botok
# from botok import WordTokenizer
# from botok.config import Config
from pathlib import Path

# ✅ 加载 LaBSE 模型
labse_model = SentenceTransformer("/mnt/data/zhenmengyuan/labse/")

# 文件路径
data_file = "task_MRC.json"
output_results_file = "task_outputs.json"
eval_results_file = "task_results.json"

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
client = OpenAI(api_key="0", base_url="http://0.0.0.0:8004/v1")

# **OpenAI API 调用**
def get_task_response(instruction, input_text, max_retries=3):
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "你是一个擅长文本阅读理解的助手"},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"}
        ]
        try:
            response = client.chat.completions.create(
                model="saves/qwen2.5/ti-zh/qwen-ti",
                messages=messages,
                temperature=0.6,
                top_p=0.6,
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

def tibetan_syllable_segment(text):
    """使用藏语音节符་进行分词"""
    # 在每个音节符后添加空格，合并多余空格，并去除首尾空格
    segmented = text.replace('་', '་ ')
    segmented = re.sub(r' +', ' ', segmented)  # 合并多个连续空格
    return segmented.strip()

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
    model_output = get_task_response(instruction, input_text)

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