####一、谷歌翻译
# import json
# from openpyxl import Workbook
#
# # 读取JSON文件
# with open('task_NTG.json', 'r', encoding='utf-8') as f:
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
# with open('task_NTG.json', 'r', encoding='utf-8') as f:
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
# with open('task_NTG_google.json', 'w', encoding='utf-8') as f:
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
from transformers.generation.utils import GenerationMixin
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
data_file = "task_NTG_google.json"
output_results_file = "google_gpt_outputs.json"
eval_results_file = "google_gpt_results.json"

# 加载数据集
with open(data_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 读取已有结果（支持断点续跑）
if os.path.exists(output_results_file):
    with open(output_results_file, "r", encoding="utf-8") as f:
        existing_results = json.load(f)
else:
    existing_results = []

# # **去重：跳过已处理的数据**
# processed_inputs = {item["input_zh"] for item in existing_results}
# dataset = [item for item in dataset if item["input_zh"] not in processed_inputs]

# # ✅ OpenAI API 客户端（适配新版 OpenAI SDK）
# client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
#                 api_key="sk-RQDnurboWQHgLHqMCvnn7ADAKpJdgPVTluLdL5fr6WM7Habm")
# # client = OpenAI(base_url = "https://ark.cn-beijing.volces.com/api/v3",
# #                 api_key  = "aa1ce03a-6b61-46fc-b914-73fae89921f3")
#
# str1="只需输出简短新闻标题)"
# # **OpenAI API 调用**
# def get_openai_response(instruction, input_text, max_retries=3):
#     for attempt in range(max_retries):
#         messages = [
#             {"role": "system", "content": "你是一个擅长新闻标题生成的助手"},
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
#
# # def get_doubao_response(instruction, input_text, max_retries=3):
# #     for attempt in range(max_retries):
# #         messages = [
# #             {"role": "system", "content": "你是一个擅长新闻标题生成的助手"},
# #             {"role": "user", "content": f"{instruction}+{str1}\n\n{input_text}"}
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
# # ✅ 使用botok进行分词
# config = Config(dialect_name="general", base_path=Path.home())
# tokenizer = WordTokenizer(config=config)
# def segment_tibetan_text(text):
#     """使用 Botok 进行藏文分词"""
#     tokens = tokenizer.tokenize(text, split_affixes=False)
#     return " ".join([token.text for token in tokens])  # 返回空格分隔的分词结果
#
# def tibetan_syllable_segment(text):
#     """使用藏语音节符་进行分词"""
#     # 在每个音节符后添加空格，合并多余空格，并去除首尾空格
#     segmented = text.replace('་', '་ ')
#     segmented = re.sub(r' +', ' ', segmented)  # 合并多个连续空格
#     return segmented.strip()
#
# # ✅ 计算 BLEU
# def compute_bleu1(pred: str, gold: str) -> float:
#     """基于 SacreBLEU 源码的 BLEU 计算函数 (适配藏文分词)"""
#
#     # ✅ 初始化 BLEU 参数（与源码默认参数对齐）
#     bleu = BLEU(
#         smooth_method='add-k',  # 平滑方法
#         smooth_value=1,  # add-k 的 k 值
#         max_ngram_order=4,  # 显式指定 BLEU-4
#         effective_order=True,  # 禁用动态 n-gram 阶数
#         tokenize='none',  # 禁用内置分词（已手动分词）
#         lowercase=False  # 不转小写
#     )
#
#     # ✅ 藏文分词（假设 tibetan_syllable_segment 已正确定义）
#     pred_tok = segment_tibetan_text(pred)
#     gold_tok = segment_tibetan_text(gold)
#
#     # ✅ 调用 sentence_score（参考 SacreBLEU 源码逻辑）
#     # 注意：references 必须是列表形式，即使只有一个参考
#     bleu_score = bleu.sentence_score(pred_tok, [gold_tok])
#
#     return bleu_score.score
#
# # ✅ 计算 BLEU
# def compute_bleu2(pred: str, gold: str) -> float:
#     """基于 SacreBLEU 源码的 BLEU 计算函数 (适配藏文分词)"""
#
#     # ✅ 初始化 BLEU 参数（与源码默认参数对齐）
#     bleu = BLEU(
#         smooth_method='add-k',  # 平滑方法
#         smooth_value=1,  # add-k 的 k 值
#         max_ngram_order=4,  # 显式指定 BLEU-4
#         effective_order=True,  # 禁用动态 n-gram 阶数
#         tokenize='none',  # 禁用内置分词（已手动分词）
#         lowercase=False  # 不转小写
#     )
#
#     # ✅ 藏文分词（假设 tibetan_syllable_segment 已正确定义）
#     pred_tok = tibetan_syllable_segment(pred)
#     gold_tok = tibetan_syllable_segment(gold)
#
#     # ✅ 调用 sentence_score（参考 SacreBLEU 源码逻辑）
#     # 注意：references 必须是列表形式，即使只有一个参考
#     bleu_score = bleu.sentence_score(pred_tok, [gold_tok])
#
#     return bleu_score.score
# # ✅ 计算 ROUGE
# rouge_evaluator = Rouge()
# def compute_rouge(pred, gold):
#     # 空值过滤
#     pred = __builtins__.str(pred).strip() or " "
#     gold = __builtins__.str(gold).strip() or " "
#
#     pred_clean = segment_tibetan_text(pred)#需要分词
#     gold_clean = segment_tibetan_text(gold)
#     # 二次验证
#     if not pred_clean.strip() or not gold_clean.strip():
#         return {
#             "rouge-1": {"f": 0.0},
#             "rouge-2": {"f": 0.0},
#             "rouge-l": {"f": 0.0}
#         }
#
#     try:
#         scores = rouge_evaluator.get_scores(pred_clean, gold_clean)
#         return scores[0]
#     except Exception as e:
#         print(f"⚠️ ROUGE 计算失败: pred='{pred_clean}' gold='{gold_clean}'")
#         return {
#             "rouge-1": {"f": 0.0},
#             "rouge-2": {"f": 0.0},
#             "rouge-l": {"f": 0.0}
#         }
#
#
# # Translation function
# API_KEY = "ymg9YDPBCofY1DxJqxy6"
# API_URL = "http://www.trans-home.com/api/index/translate"
#
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def translate_to_tibetan(text):
#     if not text or not isinstance(text, str):
#         return ""
#     try:
#         params = {"token": API_KEY}
#         data = {
#             "keywords": text,
#             "sourceLanguage": "zh-cn",  # Fixed to translate from Chinese to Tibetan
#             "targetLanguage": "bo"
#         }
#         response = requests.post(API_URL, params=params, json=data, timeout=10)
#         response.raise_for_status()
#         result = response.json()
#         if result.get("code") == 1 and "data" in result and "text" in result["data"]:
#             return result["data"]["text"]
#         else:
#             print(f"❌ 翻译失败：{text}，错误：{result.get('info', '未知错误')}")
#             return ""
#     except Exception as e:
#         print(f"❌ 翻译异常：{text}，错误：{str(e)}")
#         raise

# 🚀 **运行模型 & 评估**
results = existing_results  # 加载已完成的结果
# eval_scores = []
# # **进度条**
# for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
#     instruction = item["instruction"]
#     input_text = item["input"]
#     input_zh = item["input_zh"]
#     gold_answer = item["output"]
#
#     # **调用 GPT-4o 生成答案**
#     output_zh = get_openai_response(instruction, input_zh)
#
#     # Translate to Tibetan
#     model_output = translate_to_tibetan(output_zh)
#
#     # 计算实体F1
#     rouge_scores = compute_rouge(model_output, gold_answer)
#     bleu_scores1 = compute_bleu1(model_output, gold_answer)
#     bleu_scores2 = compute_bleu2(model_output, gold_answer)
#
#
#     # **保存结果**
#     results.append({
#         "instruction": instruction,
#         "input": input_text,
#         "input_zh": input_zh,
#         "output_zh": output_zh,
#         "gold_answer": gold_answer,
#         "model_output": model_output,
#         "rouge-1": rouge_scores["rouge-1"]["f"],
#         "rouge-2": rouge_scores["rouge-2"]["f"],
#         "rouge-l": rouge_scores["rouge-l"]["f"],
#         "bleu1": bleu_scores1,
#         "bleu2": bleu_scores2
#     })
#     # **每 50 条数据实时保存一次**
#     if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
#         with open(output_results_file, "w", encoding="utf-8") as f:
#             json.dump(results, f, ensure_ascii=False, indent=4)
#         print(f"✅ {i+1}/{len(dataset)} 条数据已保存至 {output_results_file}")

# **计算平均评估指标**
eval_scores = [item for item in results if item["model_output"] != "ERROR"]
num_samples = len(eval_scores)
avg_scores = {
        "rouge-1": sum(item["rouge-1"] for item in eval_scores) / num_samples,
        "rouge-2": sum(item["rouge-2"] for item in eval_scores) / num_samples,
        "rouge-L": sum(item["rouge-l"] for item in eval_scores) / num_samples,
        "bleu1": sum(item["bleu1"] for item in eval_scores) / num_samples,
        "bleu2": sum(item["bleu2"] for item in eval_scores) / num_samples
}

# **保存评估结果**
with open(eval_results_file, "w", encoding="utf-8") as f:
    json.dump(avg_scores, f, ensure_ascii=False, indent=4)

print(f"📊 评估结果已保存至: {eval_results_file}")