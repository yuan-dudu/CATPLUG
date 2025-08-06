###一、谷歌翻译
# import json
# from openpyxl import Workbook
#
# # 读取JSON文件
# with open('task_SUM.json', 'r', encoding='utf-8') as f:
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
# with open('task_SUM.json', 'r', encoding='utf-8') as f:
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
#         "instruction": mrc_item["instruction"],
#         "input": mrc_item['input'],
#         "output": mrc_item['output'],
#         "input_en": input_en
#     }
#     result.append(new_item)
#
# # 写入新JSON文件
# with open('task_SUM_google.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)
#
# print("文件生成完成：task_SUM_google.json")

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
# data_file = "task_SUM_google.json"
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
#             {"role": "system", "content": "You are an assistant who is good at text summarization"},
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
# #             {"role": "system", "content": "You are an assistant who is good at text summarization"},
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


####整理数据
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



####评估
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

# ✅ 使用botok进行分词
config = Config(dialect_name="general", base_path=Path.home())
tokenizer = WordTokenizer(config=config)
def segment_tibetan_text(text):
    """使用 Botok 进行藏文分词"""
    tokens = tokenizer.tokenize(text, split_affixes=False)
    return " ".join([token.text for token in tokens])  # 返回空格分隔的分词结果

def tibetan_syllable_segment(text):
    """使用藏语音节符་进行分词"""
    # 在每个音节符后添加空格，合并多余空格，并去除首尾空格
    segmented = text.replace('་', '་ ')
    segmented = re.sub(r' +', ' ', segmented)  # 合并多个连续空格
    return segmented.strip()

# ✅ 计算 BLEU
def compute_bleu1(pred: str, gold: str) -> float:
    """基于 SacreBLEU 源码的 BLEU 计算函数 (适配藏文分词)"""

    # ✅ 初始化 BLEU 参数（与源码默认参数对齐）
    bleu = BLEU(
        smooth_method='add-k',  # 平滑方法
        smooth_value=1,  # add-k 的 k 值
        max_ngram_order=4,  # 显式指定 BLEU-4
        effective_order=True,  # 禁用动态 n-gram 阶数
        tokenize='none',  # 禁用内置分词（已手动分词）
        lowercase=False  # 不转小写
    )

    # ✅ 藏文分词（假设 tibetan_syllable_segment 已正确定义）
    pred_tok = segment_tibetan_text(pred)
    gold_tok = segment_tibetan_text(gold)

    # ✅ 调用 sentence_score（参考 SacreBLEU 源码逻辑）
    # 注意：references 必须是列表形式，即使只有一个参考
    bleu_score = bleu.sentence_score(pred_tok, [gold_tok])

    return bleu_score.score

# ✅ 计算 BLEU
def compute_bleu2(pred: str, gold: str) -> float:
    """基于 SacreBLEU 源码的 BLEU 计算函数 (适配藏文分词)"""

    # ✅ 初始化 BLEU 参数（与源码默认参数对齐）
    bleu = BLEU(
        smooth_method='add-k',  # 平滑方法
        smooth_value=1,  # add-k 的 k 值
        max_ngram_order=4,  # 显式指定 BLEU-4
        effective_order=True,  # 禁用动态 n-gram 阶数
        tokenize='none',  # 禁用内置分词（已手动分词）
        lowercase=False  # 不转小写
    )

    # ✅ 藏文分词（假设 tibetan_syllable_segment 已正确定义）
    pred_tok = tibetan_syllable_segment(pred)
    gold_tok = tibetan_syllable_segment(gold)

    # ✅ 调用 sentence_score（参考 SacreBLEU 源码逻辑）
    # 注意：references 必须是列表形式，即使只有一个参考
    bleu_score = bleu.sentence_score(pred_tok, [gold_tok])

    return bleu_score.score
# ✅ 计算 ROUGE
rouge_evaluator = Rouge()
def compute_rouge(pred, gold):
    # 空值过滤
    pred = __builtins__.str(pred).strip() or " "
    gold = __builtins__.str(gold).strip() or " "

    pred_clean = segment_tibetan_text(pred)#需要分词
    gold_clean = segment_tibetan_text(gold)
    # 二次验证
    if not pred_clean.strip() or not gold_clean.strip():
        return {
            "rouge-1": {"f": 0.0},
            "rouge-2": {"f": 0.0},
            "rouge-l": {"f": 0.0}
        }

    try:
        scores = rouge_evaluator.get_scores(pred_clean, gold_clean)
        return scores[0]
    except Exception as e:
        print(f"⚠️ ROUGE 计算失败: pred='{pred_clean}' gold='{gold_clean}'")
        return {
            "rouge-1": {"f": 0.0},
            "rouge-2": {"f": 0.0},
            "rouge-l": {"f": 0.0}
        }



processed_count = 0
for item in tqdm(dataset, desc="🚀 计算实体F1", unit="条"):
    # 跳过已计算过的条目
    if "rouge-1" in item:
        continue

    gold_answer = item["gold_answer"]
    model_output = item["model_output"]

    # 提取实体并计算F1
    rouge_scores = compute_rouge(model_output, gold_answer)
    bleu_scores1 = compute_bleu1(model_output, gold_answer)
    bleu_scores2 = compute_bleu2(model_output, gold_answer)

    # 将结果写入原数据
    item["rouge-1"] = rouge_scores["rouge-1"]["f"]
    item["rouge-2"]= rouge_scores["rouge-2"]["f"]
    item["rouge-L"]= rouge_scores["rouge-l"]["f"]
    item["bleu1"] =bleu_scores1
    item["bleu2"] = bleu_scores2
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
valid_items = [item for item in dataset if "rouge-1" in item]
avg_rouge1 = sum(item["rouge-1"] for item in valid_items) / len(valid_items)
avg_rouge2 = sum(item["rouge-2"] for item in valid_items) / len(valid_items)
avg_rougel = sum(item["rouge-L"] for item in valid_items) / len(valid_items)
avg_bleu1 = sum(item["bleu1"] for item in valid_items) / len(valid_items)
avg_bleu2 = sum(item["bleu2"] for item in valid_items) / len(valid_items)

# 保存评估结果
results = {
    "rouge-1": avg_rouge1,
    "rouge-2": avg_rouge2,
    "rouge-l": avg_rougel,
    "bleu1": avg_bleu1,
    "bleu2": avg_bleu2
}
with open(eval_results_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"""
📊 评估结果：
- rouge-1：{avg_rouge1:.4f}
- rouge-2：{avg_rouge2:.4f}
- rouge-l：{avg_rougel:.4f}
- bleu1：{avg_bleu1:.4f}
- bleu2：{avg_bleu2:.4f}
- 结果文件已保存至：{eval_results_file}
""")