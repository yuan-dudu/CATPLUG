# ###翻译数据
# import json
# import os
# from openai import OpenAI
# from tqdm import tqdm

# # Data paths
# data_path = 'task_MRC.json'
# output_file = 'task_nohistory_MRC.json'


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
from sentence_transformers import SentenceTransformer, util
from transformers.generation.utils import GenerationMixin
import re
import unicodedata
from collections import Counter
from tqdm import tqdm
from mecab import MeCab
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
processed_inputs = {item["input_en"] for item in existing_results}
dataset = [item for item in dataset if item["input_en"] not in processed_inputs]

# Initialize translation client
translate_client = OpenAI(api_key="0", base_url="http://0.0.0.0:8001/v1")

# Initialize inference client
# client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
#                           api_key="sk-eU6vJLbjn2cznR4sK1lV7f9zDSFARcxcbihirVfM2wOd2fiS")

client = OpenAI(base_url = "https://ark.cn-beijing.volces.com/api/v3",
                api_key  = "30f8d6d0-96d1-4334-8244-eb15bc951063")

# ✅ 加载 LaBSE 模型
labse_model = SentenceTransformer("/mnt/data/zhenmengyuan/labse")
# ✅ 初始化 MeCab
mecab = MeCab(dictionary_path="/root/anaconda3/envs/eval/lib/python3.10/site-packages/mecab_ko_dic/dictionary")


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
# def get_openai_response(instruction, input_text, max_retries=3):
#     for attempt in range(max_retries):
#         messages = [
#             {"role": "system",
#              "content": "You are an assistant who is good at text reading comprehension."},
#             {"role": "user", "content": f"{instruction}\n\n{input_text}"}
#         ]
#         try:
#             response = client.chat.completions.create(
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
            {"role": "system", "content": "You are an assistant who is good at text reading comprehension."},
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


# Main processing loop
results = existing_results
eval_scores = []

# System prompt and translation instructions
system_content = "You are a professional translator fluent in both English and Korean."
instruction_en = "Translate the following sentences from English to Korean."

for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
    instruction = "Please complete the reading comprehension task based on the following text, you only need to output concise answers based on the last question.."
    input_en = item["input_en"]
    input_text = item["input"]  # Original Korean input for history
    gold_answer = item["output"]

    # 1. Use GPT-4o for inference on instruction + input_zh
    output_en = get_doubao_response(instruction, input_en)

    # 2. Translate output_zh back to Tibetan with history
    model_output = translate(system_content, instruction_en, output_en)

    # 3. Compute evaluation metrics
    em_score = compute_em(model_output, gold_answer)
    f1_score = compute_f1(model_output, gold_answer)
    similarity = compute_semantic_similarity(model_output, gold_answer, labse_model)

    # Save results
    results.append({
        "instruction": instruction,
        "input": input_text,
        "input_en": input_en,
        "output_en": output_en,
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