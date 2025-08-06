###翻译数据
# import json
# import os
# from openai import OpenAI
# from tqdm import tqdm

# # Data paths
# data_path = '/mnt/data/zhenmengyuan/LLaMA-Factory/saves/task_eval/ti-zh/NER/task_NER.json'
# output_file = '/mnt/data/zhenmengyuan/LLaMA-Factory/saves/task_eval/ti-zh/NER/task_nohistory_NER.json'


# # Load dataset
# try:
#     with open(data_path, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)
# except FileNotFoundError:
#     print(f"❌ 输入文件 {data_path} 不存在")
#     exit(1)
# except json.JSONDecodeError:
#     print(f"❌ 输入文件 {data_path} 格式错误")
#     exit(1)

# # Load existing translated results (if any)
# if os.path.exists(output_file):
#     try:
#         with open(output_file, 'r', encoding='utf-8') as f:
#             translated_data = json.load(f)
#         print(f"ℹ️ 继续从第 {len(translated_data)} 条开始翻译")
#     except json.JSONDecodeError:
#         print(f"❌ 输出文件 {output_file} 格式错误，将重新开始")
#         translated_data = []
# else:
#     translated_data = []

# # Determine the number of already translated items
# translated_count = len(translated_data)

# # Initialize OpenAI client
# client = OpenAI(api_key="0", base_url="http://0.0.0.0:8001/v1")

# # Translation function
# def translate(system_content, instruction, input_text):
#     messages = [
#         {"role": "system", "content": system_content},
#         {"role": "user", "content": f"{instruction}\n\n{input_text}"}
#     ]
#     try:
#         result = client.chat.completions.create(
#             messages=messages,
#             model="saves/qwen2.5/ti-zh/qwen-ti",
#             temperature=0.6,
#             top_p=0.6,
#         )
#         return result.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"❌ 翻译失败：{input_text}，错误：{str(e)}")
#         return ""

# # Translation process
# SAVE_INTERVAL = 20
# system_content = "You are a professional translator fluent in both Chinese and Tibetan."
# instruction = "Translate the following sentences from Tibetan to Chinese."

# for i, item in enumerate(tqdm(dataset[translated_count:], desc="🔄 翻译input字段", unit="条", initial=translated_count, total=len(dataset))):
#     input_text = item.get('input', '')
#     if not input_text:
#         print(f"⚠️ 缺失input字段：{item.get('id', '未知ID')}")
#         translated_data.append(item)
#         continue

#     # Translate Tibetan input to Chinese
#     input_zh = translate(system_content, instruction, input_text)

#     # Retain original data and add translated input_zh
#     new_item = item.copy()
#     new_item['input_zh'] = input_zh
#     translated_data.append(new_item)

#     # Display progress
#     print(f"Translated {i + 1}/{len(dataset)}: {input_zh}")

#     # Save every SAVE_INTERVAL items or at the end
#     if (i + 1) % SAVE_INTERVAL == 0 or (i + 1) == len(dataset):
#         try:
#             with open(output_file, 'w', encoding='utf-8') as f:
#                 json.dump(translated_data, f, ensure_ascii=False, indent=4)
#             print(f"💾 已保存 {i + 1} 条数据")
#         except Exception as e:
#             print(f"❌ 文件保存失败：{str(e)}")
#             exit(1)

# # Save final results
# try:
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(translated_data, f, ensure_ascii=False, indent=4)
#     print(f"\n✅ 翻译完成！共处理 {len(translated_data)} 条数据")
#     print(f"📁 输出文件：{output_file}")
# except Exception as e:
#     print(f"❌ 文件保存失败：{str(e)}")
#     exit(1)




####模型评估
import json
import os
from openai import OpenAI
from transformers.generation.utils import GenerationMixin
import re
import unicodedata
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import time
import torch
from rouge import Rouge
from sacrebleu.metrics import BLEU


# File paths
data_file = "task_nohistory_NER.json"
output_results_file = "nohistory-doubao_outputs.json"
eval_results_file = "nohistory-doubao_results.json"

# Load dataset
try:
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(f"❌ 输入文件 {data_file} 不存在")
    exit(1)
except json.JSONDecodeError:
    print(f"❌ 输入文件 {data_file} 格式错误")
    exit(1)

# Load existing results (for checkpointing)
if os.path.exists(output_results_file):
    with open(output_results_file, "r", encoding="utf-8") as f:
        existing_results = json.load(f)
else:
    existing_results = []

# Deduplicate: Skip already processed data
processed_inputs = {item["input_zh"] for item in existing_results if "input_zh" in item}
dataset = [item for item in dataset if "input_zh" in item and item["input_zh"] not in processed_inputs]

# Initialize translation client
translate_client = OpenAI(api_key="0", base_url="http://0.0.0.0:8001/v1")

# Initialize inference client
# inference_client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
#                           api_key="sk-eU6vJLbjn2cznR4sK1lV7f9zDSFARcxcbihirVfM2wOd2fiS")

client = OpenAI(base_url = "https://ark.cn-beijing.volces.com/api/v3",
                api_key  = "30f8d6d0-96d1-4334-8244-eb15bc951063")

# Translation function
def translate(system_content, instruction, input_text):
    messages = [
        {"role": "system", "content": system_content},
    ]
    messages.append({"role": "user", "content": f"{instruction}\n\n{input_text}"})
    try:
        result = translate_client.chat.completions.create(
            messages=messages,
            model="saves/qwen2.5/ti-zh/qwen-ti",
            temperature=0.6,
            top_p=0.6,
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 翻译失败：{input_text}，错误：{str(e)}")
        return "ERROR"
    
str="(只需以PER: LOC: ORG: 的格式输出存在的实体，不存在的类别不用输出)"
# Inference function with retry mechanism
# def get_openai_response(instruction, input_text, max_retries=3):
#     for attempt in range(max_retries):
#         messages = [
#             {"role": "system", "content": "你是一个擅长文本命名实体识别的助手，不用做多余解释"},
#             {"role": "user", "content": f"{instruction}+{str}\n\n{input_text}"}
#         ]
#         try:
#             response = inference_client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.6,
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
#             if attempt == max_retries - 1:
#                 return "ERROR"
#             time.sleep(2)

def get_doubao_response(instruction, input_text, max_retries=3):
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "你是一个擅长文本命名实体识别的助手，不用做多余解释"},
            {"role": "user", "content": f"{instruction}+{str}\n\n{input_text}"}
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



# Main processing loop
results = existing_results
eval_scores = []

# System prompt and translation instructions
system_content = "You are a professional translator fluent in both Chinese and Tibetan."
instruction_ti = "Translate the following sentences from Chinese to Tibetan. "

for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
    instruction = item["instruction"]
    input_zh = item["input_zh"]
    input_text = item["input"]  # Original Tibetan input for history
    gold_answer = item["output"]

    # 1. Use GPT-4o for inference on instruction + input_zh
    output_zh = get_doubao_response(instruction, input_zh)

    # 2. Translate output_zh back to Tibetan with history
    model_output = translate(system_content, instruction_ti, output_zh)

    # 3. Compute evaluation metrics
    # 提取实体
    gold_entities = extract_entities(gold_answer)
    pred_entities = extract_entities(model_output)

    # 计算实体F1
    macro_f1 = compute_entity_f1(gold_entities, pred_entities)

    # Save results
    results.append({
        "instruction": instruction,
        "input": input_text,
        "input_zh": input_zh,
        "output_zh": output_zh,
        "model_output": model_output,
        "gold_answer": gold_answer,
        "entity_f1": macro_f1

    })

    # Save every 10 items or at the end
    if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
        try:
            with open(output_results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"✅ {i + 1}/{len(dataset)} 条数据已保存至 {output_results_file}")
        except Exception as e:
            print(f"❌ 文件保存失败：{str(e)}")


# Compute average evaluation scores
eval_scores = [item for item in results if item["model_output"] != "ERROR"]
num_samples = len(eval_scores)
if num_samples > 0:
    avg_scores = {
        "entity_f1": sum(item["entity_f1"] for item in eval_scores) / num_samples
    }
else:
    avg_scores = {"entity_f1": 0.0}

# Save evaluation results
try:
    with open(eval_results_file, "w", encoding="utf-8") as f:
        json.dump(avg_scores, f, ensure_ascii=False, indent=4)
    print(f"📊 评估结果已保存至: {eval_results_file}")
except Exception as e:
    print(f"❌ 评估结果保存失败：{str(e)}")

