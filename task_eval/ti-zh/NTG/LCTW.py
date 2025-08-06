###翻译数据
# import json
# import os
# from openai import OpenAI
# from tqdm import tqdm

# # Data paths
# data_path = '/mnt/data/zhenmengyuan/LLaMA-Factory/saves/task_eval/ti-zh/NTG/task_NTG.json'
# output_file = '/mnt/data/zhenmengyuan/LLaMA-Factory/saves/task_eval/ti-zh/NTG/task_LCTW_NTG.json'


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
# client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")

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



#####模型评估
import json
import os
from openai import OpenAI
from transformers.generation.utils import GenerationMixin
import re
import unicodedata
from collections import Counter
from tqdm import tqdm
# import botok
# from botok import WordTokenizer
# from botok.config import Config
from pathlib import Path
import time
import torch
from rouge import Rouge
from sacrebleu.metrics import BLEU


# File paths
data_file = "task_LCTW_NTG.json"
output_results_file = "LCTW-gpt_outputs.json"
eval_results_file = "LCTW-gpt_results.json"

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
translate_client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")

# Initialize inference client
inference_client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
                          api_key="sk-eU6vJLbjn2cznR4sK1lV7f9zDSFARcxcbihirVfM2wOd2fiS")

# client = OpenAI(base_url = "https://ark.cn-beijing.volces.com/api/v3",
#                 api_key  = "30f8d6d0-96d1-4334-8244-eb15bc951063")

# Translation function
def translate(system_content, instruction, input_text, history=None):
    messages = [
        {"role": "system", "content": system_content},
    ]
    if history:
        for pair in history:
            messages.append({"role": "user", "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})
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
str="(只需输出简短新闻标题)"
# Inference function with retry mechanism
def get_openai_response(instruction, input_text, max_retries=3):
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "你是一个擅长新闻标题生成的助手"},
            {"role": "user", "content": f"{instruction}+{str}\n\n{input_text}"}
        ]
        try:
            response = inference_client.chat.completions.create(
                model="gpt-4o",
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
#             {"role": "user", "content": f"{instruction}+{str}\n\n{input_text}"}
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


# ✅ 使用botok进行分词
# config = Config(dialect_name="general", base_path=Path.home())
# tokenizer = WordTokenizer(config=config)
# def segment_tibetan_text(text):
#     """使用 Botok 进行藏文分词"""
#     tokens = tokenizer.tokenize(text, split_affixes=False)
#     return " ".join([token.text for token in tokens])  # 返回空格分隔的分词结果

def tibetan_syllable_segment(text):
    """使用藏语音节符་进行分词"""
    # 在每个音节符后添加空格，合并多余空格，并去除首尾空格
    segmented = text.replace('་', '་ ')
    segmented = re.sub(r' +', ' ', segmented)  # 合并多个连续空格
    return segmented.strip()

# ✅ 计算 BLEU
# def compute_bleu1(pred: str, gold: str) -> float:
#     """基于 SacreBLEU 源码的 BLEU 计算函数 (适配藏文分词)"""

#     # ✅ 初始化 BLEU 参数（与源码默认参数对齐）
#     bleu = BLEU(
#         smooth_method='add-k',  # 平滑方法
#         smooth_value=1,  # add-k 的 k 值
#         max_ngram_order=4,  # 显式指定 BLEU-4
#         effective_order=True,  # 禁用动态 n-gram 阶数
#         tokenize='none',  # 禁用内置分词（已手动分词）
#         lowercase=False  # 不转小写
#     )

#     # ✅ 藏文分词（假设 tibetan_syllable_segment 已正确定义）
#     pred_tok = segment_tibetan_text(pred)
#     gold_tok = segment_tibetan_text(gold)

#     # ✅ 调用 sentence_score（参考 SacreBLEU 源码逻辑）
#     # 注意：references 必须是列表形式，即使只有一个参考
#     bleu_score = bleu.sentence_score(pred_tok, [gold_tok])

#     return bleu_score.score

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

    pred_clean = tibetan_syllable_segment(pred)#需要分词
    gold_clean = tibetan_syllable_segment(gold)
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


# Main processing loop
results = existing_results
eval_scores = []

# System prompt and translation instructions
system_content = "You are a professional translator fluent in both Chinese and Tibetan."
instruction_ti = ("Translate the following sentences from Chinese to Tibetan. "
                 "Keep the following instructions in mind when translating: "
                 "The context contains a set of translations from Chinese to Tibetan. "
                 "Make full reference to the context to better understand how to translate from Chinese to Tibetan, "
                 "and ensure that the names of people, places, organizations, terminology and semantics in your translation "
                 "are consistent with the context, reduce the deviation of semantics and terminology during the translation process.")

for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
    instruction = item["instruction"]
    input_zh = item["input_zh"]
    input_text = item["input"]  # Original Tibetan input for history
    gold_answer = item["output"]

    # 1. Use GPT-4o for inference on instruction + input_zh
    output_zh = get_openai_response(instruction, input_zh)

    # 2. Translate output_zh back to Tibetan with history
    history = [[input_zh, input_text]]
    model_output = translate(system_content, instruction_ti, output_zh, history)

    # 3. Compute evaluation metrics
    rouge_scores = compute_rouge(model_output, gold_answer)
    # bleu_scores1 = compute_bleu1(model_output, gold_answer)
    bleu_scores2 = compute_bleu2(model_output, gold_answer)

    # Save results
    results.append({
        "instruction": instruction,
        "input": input_text,
        "input_zh": input_zh,
        "output_zh": output_zh,
        "model_output": model_output,
        "gold_answer": gold_answer,
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        # "bleu1": bleu_scores1,
        "bleu2": bleu_scores2

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
        # "bleu1": sum(item["bleu1"] for item in eval_scores) / num_samples,
        "bleu2": sum(item["bleu2"] for item in eval_scores) / num_samples
    }
else:
    avg_scores = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-L":0.0, "bleu2":0.0}

# Save evaluation results
try:
    with open(eval_results_file, "w", encoding="utf-8") as f:
        json.dump(avg_scores, f, ensure_ascii=False, indent=4)
    print(f"📊 评估结果已保存至: {eval_results_file}")
except Exception as e:
    print(f"❌ 评估结果保存失败：{str(e)}")