###一、谷歌翻译
# import json
# from openpyxl import Workbook
#
# # 读取JSON文件
# with open('task_MRC.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 创建两个Excel工作簿
# wb_zh = Workbook()
# ws_zh = wb_zh.active
#
#
# # 处理并写入数据
# for item in data:
#     input_text = item.get('input', '')
#     ws_zh.append([input_text])
#
# # 保存Excel文件
# wb_zh.save('en.xlsx')
#
# print("文件生成完成：en.xlsx")

##二、整理数据
# import json
# from openpyxl import load_workbook
#
# # 读取原始JSON文件
# with open('task_MRC.json', 'r', encoding='utf-8') as f:
#     mrc_data = json.load(f)
#
# # 读取zh.xlsx文件
# wb_zh = load_workbook('en.xlsx')
# ws_zh = wb_zh.active
# zh_list = [row[0].value if row[0].value is not None else "" for row in ws_zh.iter_rows(min_row=1)]
#
# # 验证数据长度一致性
# if len(mrc_data) != len(zh_list):
#     raise ValueError("JSON和Excel数据条数不一致")
#
# # 构建新数据结构
# result = []
# for mrc_item, en_part in zip(mrc_data, zh_list):
#     input_en = en_part
#
#     new_item = {
#         "instruction": "Please complete the reading comprehension task based on the following text. You only need to output concise answers based on the last question.",
#         "input": mrc_item['input'],
#         "output": mrc_item['output'],
#         "input_en": input_en
#     }
#     result.append(new_item)
#
# # 写入新JSON文件
# with open('task_MRC_google.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)
#
# print("文件生成完成：task_MRC_google.json")

###三、模型推理
# import os
# import json
# import openai
# import time
# import unicodedata
# import re
# import torch
# from tqdm import tqdm
# from rouge import Rouge
# from collections import Counter
# from openai import OpenAI # 导入OpenAI
# from sacrebleu.metrics import BLEU
# import botok
# from botok import WordTokenizer
# from botok.config import Config
# from pathlib import Path
# import requests
# from tenacity import retry, stop_after_attempt, wait_exponential
# from openpyxl import Workbook
#
# # 文件路径
# data_file = "task_MRC_google.json"
# output_results_file = "google_gpt_outputs.json"
#
# # 加载数据集
# with open(data_file, "r", encoding="utf-8") as f:
#     dataset = json.load(f)
#
# # 读取已有结果（支持断点续跑）
# if os.path.exists(output_results_file):
#     with open(output_results_file, "r", encoding="utf-8") as f:
#         existing_results = json.load(f)
# else:
#     existing_results = []
#
# # **去重：跳过已处理的数据**
# processed_inputs = {item["input_en"] for item in existing_results}
# dataset = [item for item in dataset if item["input_en"] not in processed_inputs]
#
# # ✅ OpenAI API 客户端（适配新版 OpenAI SDK）
# client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
#                 api_key="sk-RQDnurboWQHgLHqMCvnn7ADAKpJdgPVTluLdL5fr6WM7Habm")
# # client = OpenAI(base_url = "https://ark.cn-beijing.volces.com/api/v3",
# #                 api_key  = "aa1ce03a-6b61-46fc-b914-73fae89921f3")
#
# # **OpenAI API 调用**
# def get_openai_response(instruction, input_text, max_retries=3):
#     for attempt in range(max_retries):
#         messages = [
#             {"role": "system", "content": "You are an assistant who is good at text reading comprehension"},
#             {"role": "user", "content": f"{instruction}\n\n{input_text}"}
#         ]
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.6,  # 低温度，保证稳定回答
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
#             if attempt == max_retries - 1:
#                 return "ERROR"
#             time.sleep(2)  # 等待2秒后重试
#
# # def get_doubao_response(instruction, input_text, max_retries=3):
# #     for attempt in range(max_retries):
# #         messages = [
# #             {"role": "system", "content": "You are an assistant who is good at text reading comprehension"},
# #             {"role": "user", "content": f"{instruction}\n\n{input_text}"}
# #         ]
# #         try:
# #             response = client.chat.completions.create(
# #                 model="doubao-pro-32k-241215",
# #                 messages=messages,
# #                 temperature=0.6  # 低温度，保证稳定回答
# #             )
# #             return response.choices[0].message.content.strip()
# #         except Exception as e:
# #             print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
# #             if attempt == max_retries - 1:
# #                 return "ERROR"
# #             time.sleep(2)  # 等待2秒后重试
#
# # 🚀 **运行模型 & 评估**
# results = existing_results  # 加载已完成的结果
# eval_scores = []
# # **进度条**
# for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
#     instruction = item["instruction"]
#     input_text = item["input"]
#     input_en = item["input_en"]
#     gold_answer = item["output"]
#
#     # **调用 GPT-4o 生成答案**
#     output_en = get_openai_response(instruction, input_en)
#
#     # **保存结果**
#     results.append({
#         "instruction": instruction,
#         "input": input_text,
#         "input_en": input_en,
#         "output_en": output_en,
#         "gold_answer": gold_answer
#     })
#     # **每 50 条数据实时保存一次**
#     if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
#         with open(output_results_file, "w", encoding="utf-8") as f:
#             json.dump(results, f, ensure_ascii=False, indent=4)
#         print(f"✅ {i+1}/{len(dataset)} 条数据已保存至 {output_results_file}")
#
#
# # 读取JSON文件
# with open('google_gpt_outputs.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 创建两个Excel工作簿
# wb_zh = Workbook()
# ws_zh = wb_zh.active
#
#
# # 处理并写入数据
# for item in data:
#     input_text = item.get('output_en', '')
#     ws_zh.append([input_text])
#
# # 保存Excel文件
# wb_zh.save('ko_gpt.xlsx')
#
# print("文件生成完成：ko_gpt.xlsx")


###整理数据
# import json
# from openpyxl import load_workbook
#
# # 读取原始JSON文件
# with open('google_gpt_outputs.json', 'r', encoding='utf-8') as f:
#     mrc_data = json.load(f)
#
# # 读取zh.xlsx文件
# wb_eh = load_workbook('ko_gpt.xlsx')
# ws_eh = wb_eh.active
# eh_list = [row[0].value if row[0].value is not None else "" for row in ws_eh.iter_rows(min_row=1)]
#
#
# # 验证数据长度一致性
# if len(mrc_data) != len(eh_list):
#     raise ValueError("JSON和Excel数据条数不一致")
#
# # 构建新数据结构
# result = []
# for mrc_item, eh_part in zip(mrc_data, eh_list):
#     input_en = eh_part
#
#     new_item = {
#         "instruction": mrc_item['instruction'],
#         "input": mrc_item['input'],
#         "gold_answer": mrc_item['gold_answer'],
#         "input_en": mrc_item["input_en"],
#         "output_en": mrc_item["output_en"],
#         "model_output": input_en
#     }
#     result.append(new_item)
#
# # 写入新JSON文件
# with open('google_gpt_outputs2.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)
#
# print("文件生成完成：google_gpt_outputs2.json")



####五、评估
import os
import json
import openai
import time
import unicodedata
import re
import torch
from tqdm import tqdm
from rouge import Rouge
from collections import Counter
from openai import OpenAI # 导入OpenAI
from sacrebleu.metrics import BLEU
import botok
from botok import WordTokenizer
from botok.config import Config
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers.generation.utils import GenerationMixin
from mecab import MeCab

# 文件路径
input_file = "google_gpt_outputs.json"  # 输入文件
output_file = "google_gpt_outputs.json"  # 结果写回原文件
eval_results_file = "google_gpt_results.json"  # 最终评估结果


# 读取已有数据（支持断点续跑）
if os.path.exists(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
else:
    raise FileNotFoundError(f"输入文件 {input_file} 不存在")

# ✅ 加载 LaBSE 模型
labse_model = SentenceTransformer("/home/zhenmengyuan/.cache/modelscope/hub/models/sentence-transformers/LaBSE/")
# ✅ 初始化 MeCab
mecab = MeCab(dictionary_path="/data/zhenmengyuan/miniconda3/envs/LLM/lib/mecab/dic/mecab-ko-dic/")
# ✅ 计算语义相似度
def compute_semantic_similarity(pred, gold, model):
    embeddings = labse_model.encode([pred, gold], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

# ✅ 清理 & 规范化韩语文本
def clean_korean_text(text):
    text = unicodedata.normalize("NFC", text)  # 规范化 Unicode
    text = re.sub(r"\s+", " ", text).strip()  # 移除多余空格，保留单个空格
    return text

# ✅ 韩语分词
def korean_tokenize(text):
    return " ".join(mecab.morphs(text))

# ✅ 计算 EM（完全匹配 + 语义相似度优化）
def compute_em(pred, gold):
    # 如果预测文本不含藏文字符，直接判定为不匹配
    if pred == "ERROR":
        return 0.0
    pred_clean = clean_korean_text(pred)
    gold_clean = clean_korean_text(gold)

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
    pred_tokens = korean_tokenize(clean_korean_text(pred)).split()
    gold_tokens = korean_tokenize(clean_korean_text(gold)).split()

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


processed_count = 0
for item in tqdm(dataset, desc="🚀 计算实体F1", unit="条"):
    # 跳过已计算过的条目
    if "em" in item:
        continue

    gold_answer = item["gold_answer"]
    model_output = item["model_output"]

    # 提取实体并计算F1
    em_score = compute_em(model_output, gold_answer)
    f1_score = compute_f1(model_output, gold_answer)
    similarity = compute_semantic_similarity(model_output, gold_answer, labse_model)

    # 将结果写入原数据
    item["similarity"] = similarity
    item["em"]= em_score
    item["f1"]= f1_score
    processed_count += 1

    # 每处理10条保存一次
    if processed_count % 10 == 0:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        print(f"✅ 已保存 {processed_count} 条计算结果")

# 最终保存完整数据
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
print(f"✅ 所有 {len(dataset)} 条数据已保存至 {output_file}")

# 计算平均F1
valid_items = [item for item in dataset if "em" in item]
avg_em = sum(item["em"] for item in valid_items) / len(valid_items)
avg_f1 = sum(item["f1"] for item in valid_items) / len(valid_items)

# 保存评估结果
results = {
    "em": avg_em,
    "f1": avg_f1,
    "total_samples": len(valid_items)
}
with open(eval_results_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"""
📊 评估结果：
- 平均em：{avg_em:.4f}
- 平均F1：{avg_f1:.4f}
- 有效样本数：{len(valid_items)}
- 结果文件已保存至：{eval_results_file}
""")