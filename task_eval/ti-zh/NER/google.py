####一、谷歌翻译
# import json
# from openpyxl import Workbook
#
# # 读取JSON文件
# with open('task_NER.json', 'r', encoding='utf-8') as f:
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
# wb_zh.save('zh.xlsx')
#
# print("文件生成完成：zh.xlsx")

###二、整理数据
# import json
# from openpyxl import load_workbook
#
# # 读取原始JSON文件
# with open('task_NER.json', 'r', encoding='utf-8') as f:
#     mrc_data = json.load(f)
#
# # 读取zh.xlsx文件
# wb_zh = load_workbook('zh.xlsx')
# ws_zh = wb_zh.active
# zh_list = [row[0].value if row[0].value is not None else "" for row in ws_zh.iter_rows(min_row=1)]
#
#
# # 验证数据长度一致性
# if len(mrc_data) != len(zh_list):
#     raise ValueError("JSON和Excel数据条数不一致")
#
# # 构建新数据结构
# result = []
# for mrc_item, zh_part in zip(mrc_data, zh_list):
#     # 拼接zh.xlsx和Q.xlsx的内容，中间用\n连接
#     input_zh = zh_part
#
#     new_item = {
#         "instruction": mrc_item['instruction'],
#         "input": mrc_item['input'],
#         "output": mrc_item['output'],
#         "input_zh": input_zh
#     }
#     result.append(new_item)
#
# # 写入新JSON文件
# with open('task_NER_google.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)
#
# print("文件生成完成：task_NER_google.json")



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
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# 文件路径
data_file = "task_NER_google.json"
output_results_file = "google_doubao_outputs.json"
eval_results_file = "google_doubao_results.json"

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
processed_inputs = {item["input_zh"] for item in existing_results}
dataset = [item for item in dataset if item["input_zh"] not in processed_inputs]

# ✅ OpenAI API 客户端（适配新版 OpenAI SDK）
# client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
#                 api_key="sk-RQDnurboWQHgLHqMCvnn7ADAKpJdgPVTluLdL5fr6WM7Habm")
client = OpenAI(base_url = "https://ark.cn-beijing.volces.com/api/v3",
                api_key  = "aa1ce03a-6b61-46fc-b914-73fae89921f3")

str1="(只需以PER: LOC: ORG: 的格式输出存在的实体，不存在的类别不用输出)"
# **OpenAI API 调用**
# def get_openai_response(instruction, input_text, max_retries=3):
#     for attempt in range(max_retries):
#         messages = [
#             {"role": "system", "content": "你是一个擅长文本命名实体识别的助手，不用做多余解释"},
#             {"role": "user", "content": f"{instruction}+{str1}\n\n{input_text}"}
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

def get_doubao_response(instruction, input_text, max_retries=3):
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "你是一个擅长文本命名实体识别的助手，不用做多余解释"},
            {"role": "user", "content": f"{instruction}+{str1}\n\n{input_text}"}
        ]
        try:
            response = client.chat.completions.create(
                model="doubao-pro-32k-241215",
                messages=messages,
                temperature=0.6  # 低温度，保证稳定回答
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
            if attempt == max_retries - 1:
                return "ERROR"
            time.sleep(2)  # 等待2秒后重试

#新增实体提取函数
def extract_entities(text):
    """改进版实体提取函数，正确处理空格分隔的实体类型"""
    entities = {'PER': set(), 'LOC': set(), 'ORG': set()}

    # 改进正则表达式，捕获实体内容直到下一个实体类型或字符串末尾
    pattern = r"""
        (?i)                    # 忽略大小写
        \b(PER|LOC|ORG)\b       # 实体类型作为独立单词
        \s*:\s*                 # 冒号前后可能有空格
        (                       # 捕获实体内容
            (?:                 # 非捕获组，确保不跨实体类型
                (?!\b(?:PER|LOC|ORG)\b\s*:)  # 负向前瞻，排除下一个实体类型
                .               # 匹配任意字符（包括空格）
            )*?                 # 非贪婪匹配，直到下一个实体类型或末尾
        )
        (?=\s*\b(?:PER|LOC|ORG)\b\s*:|$|\s*$)  # 正向断言，确保停在下一个实体类型或字符串末尾
    """

    matches = re.findall(pattern, text, re.VERBOSE)

    for ent_type, ent_text in matches:
        ent_type = ent_type.upper()
        if not ent_text or ent_text.strip().lower() in ['none', '无', '（没有组织名）']:
            continue

        # 清洗内容：去引号、归一化、去空格
        cleaned = ent_text.strip().replace("'", "").replace('"', '')
        cleaned = unicodedata.normalize('NFC', cleaned)
        cleaned = re.sub(r'\s+', '', cleaned)  # 去除所有空格

        # 使用藏文逗号和中英文逗号分割
        parts = [p for p in re.split(r'[,\u0f0d]', cleaned) if p]

        # 过滤掉可能误匹配的实体类型关键词
        filtered_parts = []
        for part in parts:
            if part.strip().upper() not in {'PER', 'LOC', 'ORG'}:
                filtered_parts.append(part)

        if ent_type in entities and filtered_parts:
            entities[ent_type].update(filtered_parts)

    return entities


def compute_entity_f1(gold_entities, pred_entities):
    """
    改进版F1计算，精确处理空值情况：
    1. 当且仅当黄金标准或预测标准中某类型存在实体时才计算该类型
    2. 当两者都为空时，该类型不参与计算
    3. 宏平均只计算实际参与的类型
    """
    f1_scores = []
    entity_types = ['PER', 'LOC', 'ORG']

    for ent_type in entity_types:
        gold_set = gold_entities.get(ent_type, set())
        pred_set = pred_entities.get(ent_type, set())

        # 判断是否需要计算该类型
        gold_has_entities = len(gold_set) > 0
        pred_has_entities = len(pred_set) > 0

        if not gold_has_entities and not pred_has_entities:
            continue  # 双方都为空时不参与计算

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        # 处理分母为零的情况
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1)

    # 当所有类型都不参与计算时返回1
    return sum(f1_scores) / len(f1_scores) if f1_scores else 1.0

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

# 🚀 **运行模型 & 评估**
results = existing_results  # 加载已完成的结果
eval_scores = []
# **进度条**
for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
    instruction = item["instruction"]
    input_text = item["input"]
    input_zh = item["input_zh"]
    gold_answer = item["output"]

    # **调用 GPT-4o 生成答案**
    output_zh = get_doubao_response(instruction, input_zh)

    # Translate to Tibetan
    model_output = translate_to_tibetan(output_zh)
    # 提取实体
    gold_entities = extract_entities(gold_answer)
    pred_entities = extract_entities(model_output)

    # 计算实体F1
    macro_f1 = compute_entity_f1(gold_entities, pred_entities)


    # **保存结果**
    results.append({
        "instruction": instruction,
        "input": input_text,
        "input_zh": input_zh,
        "output_zh": output_zh,
        "gold_answer": gold_answer,
        "model_output": model_output,
        "entity_f1": macro_f1
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
    "entity_f1": sum(item["entity_f1"] for item in eval_scores) / num_samples
}

# **保存评估结果**
with open(eval_results_file, "w", encoding="utf-8") as f:
    json.dump(avg_scores, f, ensure_ascii=False, indent=4)

print(f"📊 评估结果已保存至: {eval_results_file}")