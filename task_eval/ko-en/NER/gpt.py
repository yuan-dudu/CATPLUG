# ####一、处理数据
# import json
# import os
#
# # 📂 定义文件路径
# input_file = "/data/zhenmengyuan/LLaMA-Factory/data/ko_data/eval_NER_en2ko.json"
# output_dir = "/data/zhenmengyuan/LLaMA-Factory/saves/task_eval/ko-en/NER/"
# output_file = os.path.join(output_dir, "task_NER.json")
# demo_output_file = os.path.join(output_dir, "task_NER_demo.json")
#
# # 📂 确保输出目录存在
# os.makedirs(output_dir, exist_ok=True)
#
# # ✅ 读取 eval_NER_en2ko.json 文件
# with open(input_file, "r", encoding="utf-8") as f:
#     eval_data = json.load(f)
#
# # ✅ 需要过滤的实体类别列表（注意AFW可能笔误为AFM，已包含AFW）
# FILTER_ENTITIES = {"ANM", "MAT", "FLD", "PLT", "TRM", "AFW", "CVL"}  # AFM改为AFW
#
#
# def filter_entities(output_str):
#     """过滤不需要的实体类别"""
#     filtered = []
#     entities = [e.strip() for e in output_str.split(",")]
#
#     for entity in entities:
#         if ":" in entity:
#             # 分割实体类型和内容
#             entity_type, content = entity.split(":", 1)
#             entity_type = entity_type.strip()
#
#             # 保留不在过滤列表中的实体
#             if entity_type not in FILTER_ENTITIES:
#                 # 重构标准化格式
#                 filtered.append(f"{entity_type}: {content.strip()}")
#
#     return ", ".join(filtered)
#
#
# # ✅ 转换格式
# task_data = []
# for item in eval_data:
#     original_output = item.get("output", "")
#     cleaned_output = filter_entities(original_output)
#
#     task_data.append({
#         "instruction": "Please complete the named entity recognition task. You only need to output PER (person name), LOC (place name), ORG (organization name), DAT (date), TIM (time), NUM (number/quantity), and EVT (event) in the following text. The output format is as follows: PER: ,LOC: ,ORG: ... You only need to output the entity that exist, and do not output those that do not exist.",
#         "input": item["history"][0][1],
#         "output": cleaned_output
#     })
#
# demo_data = task_data[:10]
#
# # ✅ 保存文件
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(task_data, f, ensure_ascii=False, indent=4)
#
# with open(demo_output_file, "w", encoding="utf-8") as f:
#     json.dump(demo_data, f, ensure_ascii=False, indent=4)
#
# print(f"转换完成，文件已保存至: {output_file}")

#
import json

# 原始文件路径
input_file = "task_LCTW_NER.json"
#output_file = "task_nohistory_NER_demo.json"

# # 新的instruction内容
new_instruction = "Please complete the NER task. You only need to output PER (person name), LOC (place name), ORG (organization name), DAT (full date, such as in the first quarter, 6 years, this coming November), TIM (time, such as morning, 90 minutes, at 1:50 AM), NUM (full entity including the number, such as No. 1517, 25 wins and 1 loss, 10 ladles, 13 horses, step 30), and EVT (full event, such as the second round of the professional basketball playoffs, ●U-18 High School Club League (Incheon)) in the following text. The output format is as follows: PER: ,LOC: ,ORG: ... Only output the entity that exist, and do not output those that do not exist."

# 读取原始数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 修改前50条数据的instruction字段
for item in data:
    item["instruction"] = new_instruction

# 保存修改后的完整文件
with open(input_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
# # 只保留前50条数据
# demo_data = data[:50]
#
# # 保存修改后的数据
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(demo_data, f, ensure_ascii=False, indent=4)

print(f"1. 已更新原始文件 {input_file} 的所有instruction字段")
#print(f"2. 已生成包含前50条数据的demo文件 {output_file}")


#####模型评估
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
#
# # 文件路径
# data_file = "task_NER.json"
# output_results_file = "gpt-4o_outputs.json"
# eval_results_file = "gpt-4o_results.json"
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
# processed_inputs = {item["input"] for item in existing_results}
# dataset = [item for item in dataset if item["input"] not in processed_inputs]
#
# # ✅ OpenAI API 客户端（适配新版 OpenAI SDK）
# client = OpenAI(base_url = "http://chatapi.littlewheat.com/v1",
#                 api_key  = "sk-p4oPXDT8fchQaVROS85cRlsGy9vd8zq6511QfSSgzUVUiBPo")
#
# # **OpenAI API 调用**
# def get_gpt4o_response(instruction, input_text, max_retries=3):
#     for attempt in range(max_retries):
#         messages = [
#             {"role": "system", "content": "You are an assistant who is good at NER task, no need to explain anything."},
#             {"role": "user", "content": f"{instruction}\n\n{input_text}"}
#         ]
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.8,  # 低温度，保证稳定回答
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
#             if attempt == max_retries - 1:
#                 return "ERROR"
#             time.sleep(2)  # 等待2秒后重试
#
# def extract_entities(text):
#     """改进版实体提取函数，正确处理空格分隔的实体类型"""
#     entities = {'PER': set(), 'LOC': set(), 'ORG': set(),'DAT': set(), 'TIM': set(), 'NUM': set(), 'EVT': set()}
#
#     # 改进正则表达式，捕获实体内容直到下一个实体类型或字符串末尾
#     pattern = r"""
#         (?i)                    # 忽略大小写
#         \b(PER|LOC|ORG|DAT|TIM|NUM|EVT)\b       # 实体类型作为独立单词
#         \s*:\s*                 # 冒号前后可能有空格
#         (                       # 捕获实体内容
#             (?:                 # 非捕获组，确保不跨实体类型
#                 (?!\b(?:PER|LOC|ORG|DAT|TIM|NUM|EVT)\b\s*:)  # 负向前瞻，排除下一个实体类型
#                 .               # 匹配任意字符（包括空格）
#             )*?                 # 非贪婪匹配，直到下一个实体类型或末尾
#         )
#         (?=\s*\b(?:PER|LOC|ORG|DAT|TIM|NUM|EVT)\b\s*:|$|\s*$)  # 正向断言，确保停在下一个实体类型或字符串末尾
#     """
#
#     matches = re.findall(pattern, text, re.VERBOSE)
#
#     for ent_type, ent_text in matches:
#         ent_type = ent_type.upper()
#         if not ent_text or ent_text.strip().lower() in ['none']:
#             continue
#
#         # 清洗内容：去引号、归一化、去空格
#         cleaned = ent_text.strip().replace("'", "").replace('"', '')
#         cleaned = unicodedata.normalize('NFC', cleaned)
#         cleaned = re.sub(r'\s+', '', cleaned)  # 去除所有空格
#
#         # 使用藏文逗号和中英文逗号分割
#         parts = [p for p in re.split(r'[,\u0f0d]', cleaned) if p]
#
#         # 过滤掉可能误匹配的实体类型关键词
#         filtered_parts = []
#         for part in parts:
#             if part.strip().upper() not in {'PER', 'LOC', 'ORG', 'DAT', 'TIM', 'NUM', 'EVT'}:
#                 filtered_parts.append(part)
#
#         if ent_type in entities and filtered_parts:
#             entities[ent_type].update(filtered_parts)
#
#     return entities
#
#
# def compute_entity_f1(gold_entities, pred_entities):
#     """
#     改进版F1计算，精确处理空值情况：
#     1. 当且仅当黄金标准或预测标准中某类型存在实体时才计算该类型
#     2. 当两者都为空时，该类型不参与计算
#     3. 宏平均只计算实际参与的类型
#     """
#     f1_scores = []
#     entity_types = ['PER', 'LOC', 'ORG', 'DAT', 'TIM', 'NUM', 'EVT']
#
#     for ent_type in entity_types:
#         gold_set = gold_entities.get(ent_type, set())
#         pred_set = pred_entities.get(ent_type, set())
#
#         # 判断是否需要计算该类型
#         gold_has_entities = len(gold_set) > 0
#         pred_has_entities = len(pred_set) > 0
#
#         if not gold_has_entities and not pred_has_entities:
#             continue  # 双方都为空时不参与计算
#
#         tp = len(gold_set & pred_set)
#         fp = len(pred_set - gold_set)
#         fn = len(gold_set - pred_set)
#
#         # 处理分母为零的情况
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#
#         f1_scores.append(f1)
#
#     # 当所有类型都不参与计算时返回1
#     return sum(f1_scores) / len(f1_scores) if f1_scores else 1.0
#
#
# # 🚀 **运行模型 & 评估**
# results = existing_results  # 加载已完成的结果
# eval_scores = []
# total_tp = Counter()
# total_fp = Counter()
# total_fn = Counter()
# # **进度条**
# for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
#     instruction = item["instruction"]
#     input_text = item["input"]
#     gold_answer = item["output"]
#
#     # **调用 GPT-4o 生成答案**
#     model_output = get_gpt4o_response(instruction, input_text)
#
#     # 提取实体
#     gold_entities = extract_entities(gold_answer)
#     pred_entities = extract_entities(model_output)
#
#     # 计算实体F1
#     macro_f1 = compute_entity_f1(gold_entities, pred_entities)
#
#
#     # **保存结果**
#     results.append({
#         "instruction": instruction,
#         "input": input_text,
#         "gold_answer": gold_answer,
#         "model_output": model_output,
#         "entity_f1": macro_f1
#     })
#
#     eval_scores.append({
#         "entity_f1": macro_f1
#     })
#
#     # **每 50 条数据实时保存一次**
#     if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
#         with open(output_results_file, "w", encoding="utf-8") as f:
#             json.dump(results, f, ensure_ascii=False, indent=4)
#         print(f"✅ {i+1}/{len(dataset)} 条数据已保存至 {output_results_file}")
#
# # **计算平均评估指标**
# eval_scores = [item for item in results if item["model_output"] != "ERROR"]
# num_samples = len(eval_scores)
# avg_scores = {
#     "entity_f1": sum(item["entity_f1"] for item in eval_scores) / num_samples
# }
#
# # **保存评估结果**
# with open(eval_results_file, "w", encoding="utf-8") as f:
#     json.dump(avg_scores, f, ensure_ascii=False, indent=4)
#
# print(f"📊 评估结果已保存至: {eval_results_file}")


###统计中文输出
# import json
# import re
#
# # 加载 JSON 文件
# with open('deepseek_outputs.json', 'r', encoding='utf-8') as f:
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

# import os
# import json
# import unicodedata
# import re
# from tqdm import tqdm
# from collections import Counter
#
# # 文件路径
# input_results_file = "LCTW-gpt_outputs_demo.json"
# output_results_file = "LCTW-gpt_outputs_demo2.json"
# eval_results_file = "LCTW-gpt_results_demo2.json"
#
# # 加载已有的结果文件
# if not os.path.exists(input_results_file):
#     raise FileNotFoundError(f"❌ 输入文件 {input_results_file} 不存在")
# with open(input_results_file, "r", encoding="utf-8") as f:
#     results = json.load(f)
#
# def extract_entities(text):
#     """实体提取函数（保持不变）"""
#     entities = {'PER': set(), 'LOC': set(), 'ORG': set(), 'DAT': set(), 'TIM': set(), 'NUM': set(), 'EVT': set()}
#
#     pattern = r"""
#         (?i)                    # 忽略大小写
#         \b(PER|LOC|ORG|DAT|TIM|NUM|EVT)\b       # 实体类型作为独立单词
#         \s*:\s*                 # 冒号前后可能有空格
#         (                       # 捕获实体内容
#             (?:                 # 非捕获组，确保不跨实体类型
#                 (?!\b(?:PER|LOC|ORG|DAT|TIM|NUM|EVT)\b\s*:)  # 负向前瞻，排除下一个实体类型
#                 .               # 匹配任意字符（包括空格）
#             )*?                 # 非贪婪匹配，直到下一个实体类型或末尾
#         )
#         (?=\s*\b(?:PER|LOC|ORG|DAT|TIM|NUM|EVT)\b\s*:|$|\s*$)  # 正向断言，确保停在下一个实体类型或字符串末尾
#     """
#
#     matches = re.findall(pattern, text, re.VERBOSE)
#
#     for ent_type, ent_text in matches:
#         ent_type = ent_type.upper()
#         if not ent_text or ent_text.strip().lower() in ['none']:
#             continue
#
#         cleaned = ent_text.strip().replace("'", "").replace('"', '')
#         cleaned = unicodedata.normalize('NFC', cleaned)
#         cleaned = re.sub(r'\s+', '', cleaned)
#
#         parts = [p for p in re.split(r'[,\u0f0d]', cleaned) if p]
#         filtered_parts = []
#         for part in parts:
#             if part.strip().upper() not in {'PER', 'LOC', 'ORG', 'DAT', 'TIM', 'NUM', 'EVT'}:
#                 filtered_parts.append(part)
#
#         if ent_type in entities and filtered_parts:
#             entities[ent_type].update(filtered_parts)
#
#     return entities
#
# def is_substring_match(ent1, ent2):
#     """检查两个实体是否互为包含关系"""
#     ent1, ent2 = ent1.strip(), ent2.strip()
#     return ent1 in ent2 or ent2 in ent1
#
# def compute_entity_f1(gold_entities, pred_entities):
#     """
#     改进版F1计算，处理空值并支持互为包含关系的TP：
#     1. 当且仅当黄金标准或预测标准中某类型存在实体时才计算该类型
#     2. 当两者都为空时，该类型不参与计算
#     3. TP计算时，若预测实体和黄金实体互为包含关系，计为1
#     4. 宏平均只计算实际参与的类型
#     """
#     f1_scores = []
#     entity_types = ['PER', 'LOC', 'ORG', 'DAT', 'TIM', 'NUM', 'EVT']
#
#     for ent_type in entity_types:
#         gold_set = gold_entities.get(ent_type, set())
#         pred_set = pred_entities.get(ent_type, set())
#
#         gold_has_entities = len(gold_set) > 0
#         pred_has_entities = len(pred_set) > 0
#
#         if not gold_has_entities and not pred_has_entities:
#             continue
#
#         tp = 0
#         matched_gold = set()
#         matched_pred = set()
#
#         for gold_ent in gold_set:
#             for pred_ent in pred_set:
#                 if is_substring_match(gold_ent, pred_ent) and gold_ent not in matched_gold and pred_ent not in matched_pred:
#                     tp += 1
#                     matched_gold.add(gold_ent)
#                     matched_pred.add(pred_ent)
#                     break
#
#         fp = len(pred_set - matched_pred)
#         fn = len(gold_set - matched_gold)
#
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#
#         f1_scores.append(f1)
#
#     return sum(f1_scores) / len(f1_scores) if f1_scores else 1.0
#
# # 🚀 重新评估并更新F1分数
# eval_scores = []
# for i, item in enumerate(tqdm(results, desc="🚀 重新评估数据", unit="条")):
#     gold_answer = item["gold_answer"]
#     model_output = item["model_output"]
#
#     if model_output == "ERROR":
#         macro_f1 = 0.0
#     else:
#         gold_entities = extract_entities(gold_answer)
#         pred_entities = extract_entities(model_output)
#         macro_f1 = compute_entity_f1(gold_entities, pred_entities)
#
#     # 更新entity_f1字段
#     item["entity_f1"] = macro_f1
#     eval_scores.append({
#         "input": item["input"],
#         "entity_f1": macro_f1
#     })
#
# # 计算平均评估指标
# num_samples = len(eval_scores)
# avg_scores = {
#     "entity_f1": sum(item["entity_f1"] for item in eval_scores) / num_samples if num_samples > 0 else 0.0,
#     "num_samples": num_samples
# }
#
# # 保存更新后的结果
# with open(output_results_file, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)
#
# # 保存评估结果
# with open(eval_results_file, "w", encoding="utf-8") as f:
#     json.dump(avg_scores, f, ensure_ascii=False, indent=4)
#
# print(f"📊 更新后的结果已保存至: {output_results_file}")
# print(f"📊 评估结果已保存至: {eval_results_file}")
# print(f"📈 平均实体F1分数: {avg_scores['entity_f1']:.4f}")
