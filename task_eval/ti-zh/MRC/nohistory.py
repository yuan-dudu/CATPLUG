# ###翻译数据
# import json
# import os
# from openai import OpenAI
# from tqdm import tqdm

# # Data paths
# data_path = '/mnt/data/zhenmengyuan/LLaMA-Factory/saves/task_eval/ti-zh/MRC/task_MRC.json'
# output_file = '/mnt/data/zhenmengyuan/LLaMA-Factory/saves/task_eval/ti-zh/MRC/task_nohistory_MRC.json'


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




#####评估
import json
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import re
import unicodedata
from collections import Counter
from tqdm import tqdm
# import botok
# from botok import WordTokenizer
# from botok.config import Config
from pathlib import Path
import time

# File paths
data_file = "task_nohistory_MRC.json"
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
processed_inputs = {item["input_zh"] for item in existing_results}
dataset = [item for item in dataset if item["input_zh"] not in processed_inputs]

# Initialize translation client
translate_client = OpenAI(api_key="0", base_url="http://0.0.0.0:8001/v1")

# Initialize inference client
# inference_client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
#                           api_key="sk-eU6vJLbjn2cznR4sK1lV7f9zDSFARcxcbihirVfM2wOd2fiS")

client = OpenAI(base_url = "https://ark.cn-beijing.volces.com/api/v3",
                api_key  = "30f8d6d0-96d1-4334-8244-eb15bc951063")

# Load LaBSE model for semantic similarity
labse_model = SentenceTransformer("/mnt/data/zhenmengyuan/labse")


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

# Inference function with retry mechanism
# def get_openai_response(instruction, input_text, max_retries=3):
#     for attempt in range(max_retries):
#         messages = [
#             {"role": "system", "content": "你是一个擅长文本阅读理解的助手"},
#             {"role": "user", "content": f"{instruction}\n\n{input_text}"}
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
            {"role": "system", "content": "你是一个擅长文本阅读理解的助手"},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"}
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

# Clean and normalize Tibetan text
def clean_tibetan_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", "", text)
    return text

# Compute semantic similarity
def compute_semantic_similarity(pred, gold, model):
    embeddings = labse_model.encode([pred, gold], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

# Tibetan syllable segmentation with botok
# config = Config(dialect_name="general", base_path=Path.home())
# tokenizer = WordTokenizer(config=config)

# def tibetan_syllable_segment(text):
#     """使用 Botok 进行藏文分词"""
#     tokens = tokenizer.tokenize(text, split_affixes=False)
#     return " ".join([token.text for token in tokens])  # 返回空格分隔的分词结果

def tibetan_syllable_segment(text):
    """使用藏语音节符་进行分词"""
    # 在每个音节符后添加空格，合并多余空格，并去除首尾空格
    segmented = text.replace('་', '་ ')
    segmented = re.sub(r' +', ' ', segmented)  # 合并多个连续空格
    return segmented.strip()

# Compute EM score
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

# Compute F1 score
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

    # 2. Translate output_zh back to Tibetan

    model_output = translate(system_content, instruction_ti, output_zh)

    # 3. Compute evaluation metrics
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

    # Save every 10 items or at the end
    if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
        try:
            with open(output_results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"✅ {i + 1}/{len(dataset)} 条数据已保存至 {output_results_file}")
        except Exception as e:
            print(f"❌ 文件保存失败：{str(e)}")

# # Retry error items (uncomment if needed)
# retry_error_items(results)

# Compute average evaluation scores
eval_scores = [item for item in results if item["model_output"] != "ERROR"]
num_samples = len(eval_scores)
if num_samples > 0:
    avg_scores = {
        "EM": sum(item["em"] for item in eval_scores) / num_samples,
        "F1": sum(item["f1"] for item in eval_scores) / num_samples
    }
else:
    avg_scores = {"EM": 0.0, "F1": 0.0}

# Save evaluation results
try:
    with open(eval_results_file, "w", encoding="utf-8") as f:
        json.dump(avg_scores, f, ensure_ascii=False, indent=4)
    print(f"📊 评估结果已保存至: {eval_results_file}")
except Exception as e:
    print(f"❌ 评估结果保存失败：{str(e)}")
