####一、谷歌翻译
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
# wb_q = Workbook()
# ws_q = wb_q.active
#
# # 处理并写入数据
# for item in data:
#     input_text = item.get('input', '')
#
#     # 从后往前查找第一个换行符
#     last_newline = input_text.rfind('\n')
#
#     if last_newline != -1:
#         # 分隔文本
#         text_part = input_text[:last_newline]  # \n前面的部分
#         question_part = input_text[last_newline + 1:]  # \n后面的部分
#
#         # 分别写入两个Excel文件
#         ws_zh.append([text_part])
#         ws_q.append([question_part])
#     else:
#         # 如果没有换行符，全文写入zh.xlsx，Q.xlsx写入空行
#         ws_zh.append([input_text])
#         ws_q.append([''])
#
# # 保存Excel文件
# wb_zh.save('zh.xlsx')
# wb_q.save('Q.xlsx')
#
# print("文件生成完成：zh.xlsx 和 Q.xlsx")

###二、整理数据
# import json
# from openpyxl import load_workbook
#
# # 读取原始JSON文件
# with open('task_MRC.json', 'r', encoding='utf-8') as f:
#     mrc_data = json.load(f)
#
# # 读取zh.xlsx文件
# wb_zh = load_workbook('zh.xlsx')
# ws_zh = wb_zh.active
# zh_list = [row[0].value if row[0].value is not None else "" for row in ws_zh.iter_rows(min_row=1)]
#
# # 读取Q.xlsx文件
# wb_q = load_workbook('Q.xlsx')
# ws_q = wb_q.active
# q_list = [row[0].value if row[0].value is not None else "" for row in ws_q.iter_rows(min_row=1)]
#
# # 验证数据长度一致性
# if len(mrc_data) != len(zh_list) or len(mrc_data) != len(q_list):
#     raise ValueError("JSON和Excel数据条数不一致")
#
# # 构建新数据结构
# result = []
# for mrc_item, zh_part, q_part in zip(mrc_data, zh_list, q_list):
#     # 拼接zh.xlsx和Q.xlsx的内容，中间用\n连接
#     input_zh = f"{zh_part}\n{q_part}" if q_part else zh_part
#
#     new_item = {
#         "instruction": "请根据以下文本完成阅读理解任务，只需根据最后的问题输出简洁的答案。",
#         "input": mrc_item['input'],
#         "output": mrc_item['output'],
#         "input_zh": input_zh
#     }
#     result.append(new_item)
#
# # 写入新JSON文件
# with open('task_MRC_zh.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)
#
# print("文件生成完成：task_MRC_zh.json")


####d调用谷歌进行翻译
# import json
# import requests
# from tqdm import tqdm
# import os
# from tenacity import retry, stop_after_attempt, wait_exponential
# # API 配置
# API_KEY = "ymg9YDPBCofY1DxJqxy6"
# API_URL = "http://www.trans-home.com/api/index/translate"
#
# # 重试机制：最多重试3次，等待时间指数增长
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def translate_to_tibetan(text):
#     """将中文文本翻译为藏文"""
#     if not text or not isinstance(text, str):
#         return ""
#     try:
#         params = {"token": API_KEY}
#         data = {
#             "keywords": text,
#             "sourceLanguage": "bo",
#             "targetLanguage": "zh-cn"
#         }
#         response = requests.post(API_URL, params=params, json=data, timeout=10)
#         response.raise_for_status()
#         result = response.json()
#
#         if result.get("code") == 1 and "data" in result and "text" in result["data"]:
#             return result["data"]["text"]
#         else:
#             print(f"❌ 翻译失败：{text}，错误：{result.get('info', '未知错误')}")
#             return ""
#     except Exception as e:
#         print(f"❌ 翻译异常：{text}，错误：{str(e)}")
#         raise  # 抛出异常以触发重试
#
# # 文件路径配置
# input_file = "task_MRC.json"
# output_file = "task_google_MRC.json"
#
# # 读取原始数据
# try:
#     with open(input_file, "r", encoding="utf-8") as f:
#         original_data = json.load(f)
# except FileNotFoundError:
#     print(f"❌ 输入文件 {input_file} 不存在")
#     exit(1)
# except json.JSONDecodeError:
#     print(f"❌ 输入文件 {input_file} 格式错误")
#     exit(1)
#
# # 检查输出文件是否存在并确定起始点
# if os.path.exists(output_file):
#     try:
#         with open(output_file, "r", encoding="utf-8") as f:
#             processed_data = json.load(f)
#         start_index = len(processed_data)
#         print(f"ℹ️ 继续从第 {start_index} 条开始翻译")
#     except json.JSONDecodeError:
#         print(f"❌ 输出文件 {output_file} 格式错误，将重新开始")
#         processed_data = []
#         start_index = 0
# else:
#     processed_data = []
#     start_index = 0
#
# # 处理并翻译数据
# for i in tqdm(range(start_index, len(original_data)), desc="🔄 翻译input字段", unit="条"):
#     item = original_data[i]
#     if "input" not in item:
#         print(f"⚠️ 缺失input字段：{item.get('id', '未知ID')}")
#         processed_data.append(item)
#         continue
#
#     # 执行翻译
#     tibetan_input = translate_to_tibetan(item["input"])
#
#     # 保留原始数据并添加翻译结果
#     new_item = item.copy()
#     new_item["input_zh"] = tibetan_input
#     processed_data.append(new_item)
#
#     # 每翻译10条保存一次
#     if (i + 1) % 10 == 0:
#         try:
#             with open(output_file, "w", encoding="utf-8") as f:
#                 json.dump(processed_data, f, ensure_ascii=False, indent=4)
#             print(f"💾 已保存 {i + 1} 条数据")
#         except Exception as e:
#             print(f"❌ 文件保存失败：{str(e)}")
#             exit(1)
#
# # 翻译完成后保存最终结果
# try:
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(processed_data, f, ensure_ascii=False, indent=4)
#     print(f"\n✅ 翻译完成！共处理 {len(processed_data)} 条数据")
#     print(f"📁 输出文件：{output_file}")
# except Exception as e:
#     print(f"❌ 文件保存失败：{str(e)}")
#     exit(1)




####三、使用gpt-4o进行评估
import json
import openai
import os
import time
import unicodedata
import re
from tqdm import tqdm
from collections import Counter
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import botok
from botok import WordTokenizer
from botok.config import Config
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Load LaBSE model for semantic similarity
labse_model = SentenceTransformer("/home/zhenmengyuan/.cache/modelscope/hub/models/sentence-transformers/LaBSE/")

# File paths
data_file = "task_google_MRC.json"
output_results_file = "google_deepseek_outputs.json"
eval_results_file = "google_deepseek_results.json"

# Load dataset
with open(data_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Load existing results for checkpointing
if os.path.exists(output_results_file):
    with open(output_results_file, "r", encoding="utf-8") as f:
        existing_results = json.load(f)
else:
    existing_results = []

# Deduplicate: Skip already processed data
processed_inputs = {item["input_zh"] for item in existing_results}
dataset = [item for item in dataset if item["input_zh"] not in processed_inputs]

# OpenAI API client
client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
                api_key="sk-p4oPXDT8fchQaVROS85cRlsGy9vd8zq6511QfSSgzUVUiBPo")

# client = OpenAI(base_url = "https://ark.cn-beijing.volces.com/api/v3",
#                 api_key  = "aa1ce03a-6b61-46fc-b914-73fae89921f3")

# GPT-4o inference function
def get_openai_response(instruction, input_text, max_retries=3):
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "你是一个擅长文本阅读理解的助手"},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"}
        ]
        try:
            response = client.chat.completions.create(
                model="deepseek-r1",
                messages=messages,
                temperature=0.6,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
            if attempt == max_retries - 1:
                return "ERROR"
            time.sleep(2)


# def get_doubao_response(instruction, input_text, max_retries=3):
#     for attempt in range(max_retries):
#         messages = [
#             {"role": "system", "content": "你是一个擅长文本阅读理解的助手"},
#             {"role": "user", "content": f"{instruction}\n\n{input_text}"}
#         ]
#         try:
#             response = client.chat.completions.create(
#                 model="doubao-pro-32k-241215",
#                 messages=messages,
#                 temperature=0.6  # 低温度，保证稳定回答
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
#             if attempt == max_retries - 1:
#                 return "ERROR"
#             time.sleep(2)  # 等待2秒后重试

# ### 计算语义相似度
def compute_semantic_similarity(pred, gold, model):
    embeddings = labse_model.encode([pred, gold], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

# Text cleaning and normalization
def clean_tibetan_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", "", text)
    return text

# Tibetan syllable segmentation with botok
config = Config(dialect_name="general", base_path=Path.home())
tokenizer = WordTokenizer(config=config)

def tibetan_syllable_segment(text):
    tokens = tokenizer.tokenize(text, split_affixes=False)
    return " ".join([token.text for token in tokens])

# EM score computation
def compute_em(pred, gold):
    if not re.search(r"[ༀ-ྼ0-9]", pred) or pred == "ERROR":
        return 0.0
    pred_clean = clean_tibetan_text(pred)
    gold_clean = clean_tibetan_text(gold)
    if pred_clean == gold_clean:
        return 1.0
    if pred_clean and gold_clean and (gold_clean in pred_clean or pred_clean in gold_clean):
        return 1.0
    similarity_score = compute_semantic_similarity(pred, gold, labse_model)
    if similarity_score > 0.9:
        return 1.0
    return 0.0

# F1 score computation
def compute_f1(pred, gold):
    pred_tokens = tibetan_syllable_segment(pred).split()
    gold_tokens = tibetan_syllable_segment(gold).split()
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common_tokens = pred_counter & gold_counter
    N_TP = sum(common_tokens.values())
    N_FP = sum(pred_counter.values()) - N_TP
    N_FN = sum(gold_counter.values()) - N_TP
    if N_TP == 0:
        return 0.0
    precision = N_TP / (N_TP + N_FP)
    recall = N_TP / (N_TP + N_FN)
    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Translation function
API_KEY = "ymg9YDPBCofY1DxJqxy6"
API_URL = "http://www.trans-home.com/api/index/translate"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def translate_to_tibetan(text):
    if not text or not isinstance(text, str):
        return ""
    try:
        params = {"token": API_KEY}
        data = {
            "keywords": text,
            "sourceLanguage": "zh-cn",  # Fixed to translate from Chinese to Tibetan
            "targetLanguage": "bo"
        }
        response = requests.post(API_URL, params=params, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("code") == 1 and "data" in result and "text" in result["data"]:
            return result["data"]["text"]
        else:
            print(f"❌ 翻译失败：{text}，错误：{result.get('info', '未知错误')}")
            return ""
    except Exception as e:
        print(f"❌ 翻译异常：{text}，错误：{str(e)}")
        raise

# Main processing loop
results = existing_results
eval_scores = []

for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
    instruction = item["instruction"]
    input_zh = item["input_zh"]
    input_text = item["input"]
    gold_answer = item["output"]

    # Inference with GPT-4o
    output_zh = get_openai_response(instruction, input_zh)

    # Translate to Tibetan
    model_output = translate_to_tibetan(output_zh)

    # Compute evaluation metrics
    em_score = compute_em(model_output, gold_answer)
    f1_score = compute_f1(model_output, gold_answer)
    similarity = compute_semantic_similarity(model_output, gold_answer, labse_model)

    # Save results
    results.append({
        "instruction": instruction,
        "input": input_text,
        "input_zh": input_zh,
        "output_zh": output_zh,
        "model_output": model_output,
        "gold_answer": gold_answer,
        "similarity": similarity,
        "em": em_score,
        "f1": f1_score
    })

    eval_scores.append({
        "similarity": similarity,
        "em": em_score,
        "f1": f1_score
    })

    # Save every 10 items or at the end
    if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
        with open(output_results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"✅ {i+1}/{len(dataset)} 条数据已保存至 {output_results_file}")

# Compute average scores
eval_scores = [item for item in results if item["model_output"] != "ERROR"]
num_samples = len(eval_scores)
avg_scores = {
    "EM": sum(item["em"] for item in eval_scores) / num_samples if num_samples > 0 else 0.0,
    "F1": sum(item["f1"] for item in eval_scores) / num_samples if num_samples > 0 else 0.0
}

# Save evaluation results
with open(eval_results_file, "w", encoding="utf-8") as f:
    json.dump(avg_scores, f, ensure_ascii=False, indent=4)

print(f"📊 评估结果已保存至: {eval_results_file}")


