# ####一、处理数据
# import json
# import os
#
# # 📂 定义文件路径
# input_file = "/data/zhenmengyuan/LLaMA-Factory/data/tib_data/eval_NER_zh2ti.json"
# output_dir = "/data/zhenmengyuan/LLaMA-Factory/saves/task_eval/NER/"
# output_file = os.path.join(output_dir, "task_NER.json")
# demo_output_file = os.path.join(output_dir, "task_NER_demo.json")  # 生成 demo 文件
#
# # 📂 确保输出目录存在
# os.makedirs(output_dir, exist_ok=True)
#
# # ✅ 读取 eval_SUM.json 文件
# with open(input_file, "r", encoding="utf-8") as f:
#     eval_data = json.load(f)
#
# # ✅ 转换格式
# task_data = []
# for item in eval_data:
#     task_data.append({
#         "instruction": "请完成命名实体识别任务，只需输出以下文本中存在的PER（人名）、LOC（地名）、ORG（组织名）：",
#         "input": item["history"][0][1],
#         "output": item["output"]
#     })
# demo_data=task_data[:10]
# # ✅ 生成 `task_SUM.json`
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(task_data, f, ensure_ascii=False, indent=4)
#
# with open(demo_output_file, "w", encoding="utf-8") as f:
#     json.dump(demo_data, f, ensure_ascii=False, indent=4)
# # 📢 打印完成信息
# print(f"转换完成，文件已保存至: {output_file}")


####处理实体单引号
# import json
# import re
#
#
# def remove_quotes_in_entities(input_file, output_file):
#     """
#     处理NER任务JSON文件，去除output字段中实体值的单引号
#     参数：
#         input_file: 原始文件路径（如"task_NER.json"）
#         output_file: 输出文件路径（如"task_NER_processed.json"）
#     """
#     # 读取原始文件
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     # 定义正则表达式模式（匹配 PER/LOC/ORG: 后的单引号内容）
#     pattern = re.compile(
#         r"(PER|LOC|ORG):\s*'(.*?)'",
#         flags=re.MULTILINE
#     )
#
#     # 处理每个条目
#     for item in data:
#         if "output" in item:
#             # 替换单引号并保留实体结构
#             processed = pattern.sub(r'\1: \2', item["output"])
#             item["output"] = processed
#
#     # 保存处理后的文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)
#
#
# # 使用示例
# remove_quotes_in_entities(
#     input_file="task_NER_demo.json",
#     output_file="task_NER_demo_processed.json"
# )


#####模型评估
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

# 文件路径
data_file = "task_NER.json"
output_results_file = "gpt-4o_outputs.json"
eval_results_file = "gpt-4o_results.json"

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
processed_inputs = {item["input"] for item in existing_results}
dataset = [item for item in dataset if item["input"] not in processed_inputs]

# ✅ OpenAI API 客户端（适配新版 OpenAI SDK）
client = OpenAI(base_url = "http://chatapi.littlewheat.com/v1",
                api_key  = "sk-p4oPXDT8fchQaVROS85cRlsGy9vd8zq6511QfSSgzUVUiBPo")

str="(只需以PER: LOC: ORG: 的格式输出存在的藏文实体，不存在的类别不用输出)"
# **OpenAI API 调用**
def get_gpt4o_response(instruction, input_text, max_retries=3):
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "你是一个擅长文本命名实体识别的助手，不用做多余解释"},
            {"role": "user", "content": f"{instruction}+{str}\n\n{input_text}"}
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.8,  # 低温度，保证稳定回答
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenAI API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
            if attempt == max_retries - 1:
                return "ERROR"
            time.sleep(2)  # 等待2秒后重试


#新增实体提取函数
def extract_entities(text):
    """增强版实体提取函数（处理None/无实体标记+多实体分割）"""
    entities = {'PER': set(), 'LOC': set(), 'ORG': set()}

    # 正则表达式（支持多格式）
    pattern = r"""
        (?i)                    # 忽略大小写
        (PER|LOC|ORG)           # 实体类型
        \s*:\s*                 # 冒号前后可能有空格
        (                       # 捕获实体内容
            (?!                 # 排除特定否定情况
                None\b          # 排除纯None
                |无\b          # 排除中文无
                |（没有组织名）  # 排除中文括号标记
            )
            (?:                 # 内容匹配组
                '[^']*'         # 单引号内容
                |               # 或
                "[^"]*"         # 双引号内容
                |               # 或
                [^\n:]+         # 无引号内容（直到换行或冒号）
            )
            (?:\s*,\s*[^\n:]+)* # 允许逗号分隔的多实体
        )?
    """

    matches = re.findall(pattern, text, re.VERBOSE)

    for ent_type, ent_text in matches:
        ent_type = ent_type.upper()
        # 处理空内容或特殊标记
        if not ent_text or ent_text.strip().lower() in ['none', '无', '（没有组织名）']:
            continue

        # 清洗内容：去引号+去空格+分割实体
        cleaned = ent_text.strip().replace("'", "").replace('"', "").strip()
        parts = [p.strip() for p in re.split(r"\s*,\s*", cleaned) if p.strip()]

        if ent_type in entities and parts:
            entities[ent_type].update(parts)

    return entities

# 新增F1计算函数
def compute_entity_f1(gold_entities, pred_entities):
    """计算实体级别的F1分数（处理全空情况）"""
    tp = {'PER': 0, 'LOC': 0, 'ORG': 0}
    fp = {'PER': 0, 'LOC': 0, 'ORG': 0}
    fn = {'PER': 0, 'LOC': 0, 'ORG': 0}

    # 统计每个类型的TP/FP/FN
    for ent_type in ['PER', 'LOC', 'ORG']:
        gold_set = gold_entities.get(ent_type, set())
        pred_set = pred_entities.get(ent_type, set())

        tp[ent_type] = len(gold_set & pred_set)
        fp[ent_type] = len(pred_set - gold_set)
        fn[ent_type] = len(gold_set - pred_set)

    # 计算宏平均F1（处理全空情况）
    f1_scores = []
    for ent_type in ['PER', 'LOC', 'ORG']:
        gold_has_ent = len(gold_entities.get(ent_type, set())) > 0
        pred_has_ent = len(pred_entities.get(ent_type, set())) > 0

        # 当黄金和预测都没有该类型实体时，记为完美匹配
        if not gold_has_ent and not pred_has_ent:
            f1_scores.append(1.0)
            continue

        # 正常计算
        precision = tp[ent_type] / (tp[ent_type] + fp[ent_type]) if (tp[ent_type] + fp[ent_type]) > 0 else 0
        recall = tp[ent_type] / (tp[ent_type] + fn[ent_type]) if (tp[ent_type] + fn[ent_type]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1


# 🚀 **运行模型 & 评估**
results = existing_results  # 加载已完成的结果
eval_scores = []
total_tp = Counter()
total_fp = Counter()
total_fn = Counter()
# **进度条**
for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
    instruction = item["instruction"]
    input_text = item["input"]
    gold_answer = item["output"]

    # **调用 GPT-4o 生成答案**
    model_output = get_gpt4o_response(instruction, input_text)

    # 提取实体
    gold_entities = extract_entities(gold_answer)
    pred_entities = extract_entities(model_output)

    # 计算实体F1
    macro_f1 = compute_entity_f1(gold_entities, pred_entities)


    # **保存结果**
    results.append({
        "instruction": instruction,
        "input": input_text,
        "gold_answer": gold_answer,
        "model_output": model_output,
        "entity_f1": macro_f1
    })

    eval_scores.append({
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



#####调整输出
import os
import json
import re
import unicodedata
from collections import defaultdict

# 文件路径
input_file="LCTW-deepseek_outputs.json"
#output_results_file = "deepseek_outputs.json"
eval_results_file = "LCTW-deepseek_results2.json"

# 加载已有的输出结果
if os.path.exists(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    raise FileNotFoundError(f"文件 {input_file} 不存在，请确保已生成该文件。")


# 实体提取函数

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


# 计算评估指标
eval_scores = []
for item in results:
    gold_answer = item["gold_answer"]
    model_output = item["model_output"]

    gold_entities = extract_entities(gold_answer)
    pred_entities = extract_entities(model_output)

    macro_f1 = compute_entity_f1(gold_entities, pred_entities)

    item["entity_f1"] = macro_f1
    eval_scores.append({"entity_f1": macro_f1})

# 计算平均评估指标
num_samples = len(results)
if num_samples > 0:
    avg_scores = {
        "entity_f1": sum(item["entity_f1"] for item in results) / num_samples
    }
else:
    avg_scores = {"entity_f1": 0.0}

# 保存结果
with open(eval_results_file, "w", encoding="utf-8") as f:
    json.dump(avg_scores, f, ensure_ascii=False, indent=4)


print(f"📊 评估结果已保存至: {eval_results_file}")