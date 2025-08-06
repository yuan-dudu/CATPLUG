###翻译数据
# import json
# import os
# from openai import OpenAI
# from tqdm import tqdm

# # Data paths
# data_path = 'task_QA.json'
# output_file = 'task_nohistory_QA.json'


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
#             model="saves/qwen2.5/ko-en/qwen-ko",
#             temperature=0.6,
#             top_p=0.6,
#         )
#         return result.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"❌ 翻译失败：{input_text}，错误：{str(e)}")
#         return ""

# # Translation process
# SAVE_INTERVAL = 20
# system_content = "You are a professional translator fluent in both English and Korean."
# instruction = "Translate the following sentences from Korean to English."

# for i, item in enumerate(tqdm(dataset[translated_count:], desc="🔄 翻译input字段", unit="条", initial=translated_count, total=len(dataset))):
#     input_text = item.get('input', '')
#     if not input_text:
#         print(f"⚠️ 缺失input字段：{item.get('id', '未知ID')}")
#         translated_data.append(item)
#         continue

#     # Translate Tibetan input to Chinese
#     input_en = translate(system_content, instruction, input_text)

#     # Retain original data and add translated input_zh
#     new_item = item.copy()
#     new_item['input_en'] = input_en
#     translated_data.append(new_item)

#     # Display progress
#     #print(f"Translated {i + 1}/{len(dataset)}: {input_en}")

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



#####评估
import json
import os
from openai import OpenAI
from transformers.generation.utils import GenerationMixin
import re
import unicodedata
from collections import Counter
from tqdm import tqdm
from mecab import MeCab
from pathlib import Path
import time
import torch
from rouge import Rouge
from sacrebleu.metrics import BLEU
# from nltk.translate.meteor_score import meteor_score


# File paths
data_file = "task_nohistory_QA.json"
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
processed_inputs = {item["input_en"] for item in existing_results if "input_en" in item}
dataset = [item for item in dataset if "input_en" in item and item["input_en"] not in processed_inputs]

# Initialize translation client
translate_client = OpenAI(api_key="0", base_url="http://0.0.0.0:8001/v1")

# Initialize inference client
# client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
#                 api_key="sk-eU6vJLbjn2cznR4sK1lV7f9zDSFARcxcbihirVfM2wOd2fiS")

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
            model="saves/qwen2.5/ko-en/qwen-ko",
            temperature=0.6,
            top_p=0.6,
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 翻译失败：{input_text}，错误：{str(e)}")
        return "ERROR"

# Inference function with retry mechanism
str1="(You only need to output the answer, no extra explanation is needed)"
# **OpenAI API 调用**
# def get_openai_response(instruction, input_text, max_retries=3):
#     for attempt in range(max_retries):
#         messages = [
#             {"role": "system", "content": "You are an assistant who is good at answering questions"},
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
            {"role": "system", "content": "You are an assistant who is good at answering questions"},
            {"role": "user", "content": f"{instruction}+{str1}\n\n{input_text}"}
        ]
        try:
            response = client.chat.completions.create(
                model="doubao-pro-32k-241215",
                messages=messages,
                temperature=0.6,  # 低温度，保证稳定回答
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
            if attempt == max_retries - 1:
                return "ERROR"
            time.sleep(2)  # 等待2秒后重试


# ✅ 使用mecab进行分词
# ✅ 初始化 MeCab
mecab = MeCab(dictionary_path="/root/anaconda3/envs/eval/lib/python3.10/site-packages/mecab_ko_dic/dictionary")
def korean_tokenize(text):
    return " ".join(mecab.morphs(text))
# ✅ 计算 BLEU
def compute_bleu1(pred: str, gold: str) -> float:
    """基于 SacreBLEU 源码的 BLEU 计算函数 (适配藏文分词)"""

    # ✅ 初始化 BLEU 参数（与源码默认参数对齐）
    bleu = BLEU(
        smooth_method='add-k',  # 平滑方法
        smooth_value=1,  # add-k 的 k 值
        max_ngram_order=4,  # 显式指定 BLEU-4
        effective_order=True,  # 禁用动态 n-gram 阶数
        tokenize='ko-mecab',  # 禁用内置分词（已手动分词）
        lowercase=False  # 不转小写
    )

    # ✅ 调用 sentence_score（参考 SacreBLEU 源码逻辑）
    # 注意：references 必须是列表形式，即使只有一个参考
    bleu_score = bleu.sentence_score(pred, [gold])

    return bleu_score.score

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
    pred_tok=korean_tokenize(pred)
    gold_tok=korean_tokenize(gold)

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

    pred_clean = korean_tokenize(pred)#需要分词
    gold_clean = korean_tokenize(gold)
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


# def compute_meteor(pred: str, gold: str) -> float:
#     # 使用 mecab 进行韩语分词
#     pred_tokens = mecab.morphs(pred)
#     gold_tokens = mecab.morphs(gold)

#     # 计算 METEOR 分数
#     score = meteor_score([gold_tokens], pred_tokens)
#     return score

# Main processing loop
results = existing_results
eval_scores = []

# System prompt and translation instructions
system_content = "You are a professional translator fluent in both English and Korean."
instruction_ti = "Translate the following sentences from English to Korean. "


for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
    instruction = item["instruction"]
    input_en = item["input_en"]
    input_text = item["input"]  # Original Tibetan input for history
    gold_answer = item["output"]

    # 1. Use GPT-4o for inference on instruction + input_zh
    output_en = get_doubao_response(instruction, input_en)

    # 2. Translate output_zh back to Tibetan with history

    model_output = translate(system_content, instruction_ti, output_en)

    # 3. Compute evaluation metrics
    rouge_scores = compute_rouge(model_output, gold_answer)
    bleu_scores1 = compute_bleu1(model_output, gold_answer)
    bleu_scores2 = compute_bleu2(model_output, gold_answer)
    # meteor_score_value = compute_meteor(model_output, gold_answer)

    # Save results
    results.append({
        "instruction": instruction,
        "input": input_text,
        "input_en": input_en,
        "output_en": output_en,
        "model_output": model_output,
        "gold_answer": gold_answer,
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "bleu1": bleu_scores1,
        "bleu2": bleu_scores2,
        # "meteor": meteor_score_value  # 新增

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
        "rouge-1": sum(item["rouge-1"] for item in eval_scores) / num_samples,
        "rouge-2": sum(item["rouge-2"] for item in eval_scores) / num_samples,
        "rouge-L": sum(item["rouge-l"] for item in eval_scores) / num_samples,
        "bleu1": sum(item["bleu1"] for item in eval_scores) / num_samples,
        "bleu2": sum(item["bleu2"] for item in eval_scores) / num_samples,
        # "METEOR": sum(item["meteor"] for item in eval_scores) / num_samples  # 新增
    }
else:
    avg_scores = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-L":0.0, "bleu1":0.0}

# Save evaluation results
try:
    with open(eval_results_file, "w", encoding="utf-8") as f:
        json.dump(avg_scores, f, ensure_ascii=False, indent=4)
    print(f"📊 评估结果已保存至: {eval_results_file}")
except Exception as e:
    print(f"❌ 评估结果保存失败：{str(e)}")