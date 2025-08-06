# ####一、处理数据
# import json
# import os
#
# # 📂 定义文件路径
# input_file = "/data/zhenmengyuan/LLaMA-Factory/data/ko_data/eval_QA.json"
# output_dir = "/data/zhenmengyuan/LLaMA-Factory/saves/task_eval/ko-en/QA/"
# output_file = os.path.join(output_dir, "task_QA.json")
# #demo_output_file = os.path.join(output_dir, "task_SUM_demo.json")  # 生成 demo 文件
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
#         "instruction": "Please answer the following questions or requests.",
#         "input": item["text_ko"],
#         "output": item["answer_ko"]
#     })
#
# # ✅ 生成 `task_SUM.json`
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(task_data, f, ensure_ascii=False, indent=2)
#
# # 📢 打印完成信息
# print(f"转换完成，文件已保存至: {output_file}")



####二、调用模型进行评估
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
from mecab import MeCab
from pathlib import Path
from nltk.translate.meteor_score import meteor_score
import nltk
#nltk.download('wordnet')

# 文件路径
data_file = "task_QA.json"
output_results_file = "gpt_outputs.json"
eval_results_file = "gpt_results.json"

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
                api_key  = "sk-RQDnurboWQHgLHqMCvnn7ADAKpJdgPVTluLdL5fr6WM7Habm")
str="(You only need to output the answer, no extra explanation is needed, answer in Korean)"
# **OpenAI API 调用**
def get_gpt4o_response(instruction, input_text, max_retries=3):
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "You are an assistant who is good at answering questions"},
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


# ✅ 使用mecab进行分词
# ✅ 初始化 MeCab
mecab = MeCab(dictionary_path="/data/zhenmengyuan/miniconda3/envs/LLM/lib/mecab/dic/mecab-ko-dic/")
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


def compute_meteor(pred: str, gold: str) -> float:
    # 使用 mecab 进行韩语分词
    pred_tokens = mecab.morphs(pred)
    gold_tokens = mecab.morphs(gold)

    # 计算 METEOR 分数
    score = meteor_score([gold_tokens], pred_tokens)
    return score

# 🚀 **运行模型 & 评估**
results = existing_results  # 加载已完成的结果
eval_scores = []

# **进度条**
for i, item in enumerate(tqdm(dataset, desc="🚀 处理数据", unit="条")):
    instruction = item["instruction"]
    input_text = item["input"]
    gold_answer = item["output"]

    # **调用 GPT-4o 生成答案**
    model_output = get_gpt4o_response(instruction, input_text)

    # **计算评估指标**
    rouge_scores = compute_rouge(model_output, gold_answer)
    bleu_scores1 = compute_bleu1(model_output, gold_answer)
    bleu_scores2 = compute_bleu2(model_output, gold_answer)
    meteor_score_value = compute_meteor(model_output, gold_answer)



    # **保存结果**
    results.append({
        "instruction": instruction,
        "input": input_text,
        "gold_answer": gold_answer,
        "model_output": model_output,
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "bleu1": bleu_scores1,
        "bleu2": bleu_scores2,
        "meteor": meteor_score_value  # 新增
    })

    eval_scores.append({
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "bleu1": bleu_scores1,
        "bleu2": bleu_scores2,
        "meteor": meteor_score_value  # 新增
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
    "ROUGE-1": sum(item["rouge-1"] for item in eval_scores) / num_samples,
    "ROUGE-2": sum(item["rouge-2"] for item in eval_scores) / num_samples,
    "ROUGE-L": sum(item["rouge-l"] for item in eval_scores) / num_samples,
    "BLEU1": sum(item["bleu1"] for item in eval_scores) / num_samples,
    "BLEU2": sum(item["bleu2"] for item in eval_scores) / num_samples,
    "METEOR": sum(item["meteor"] for item in eval_scores) / num_samples  # 新增
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


# import json
# from rouge import Rouge
# from sacrebleu.metrics import BLEU
# import botok
# import re
# from botok import WordTokenizer
# from botok.config import Config
# from pathlib import Path
# from tqdm import tqdm
#
# # 🔄 修改输出文件路径
# #output_results_file = "deepseek_outputs2.json"  # 新增输出文件
# eval_results_file = "LCTW-deepseek_results2.json"  # 新增评估文件
# #
# # # ✅ 使用botok进行分词
# # config = Config(dialect_name="general", base_path=Path.home())
# # tokenizer = WordTokenizer(config=config)
# #
# #
# # def segment_tibetan_text(text):
# #     """使用 Botok 进行藏文分词"""
# #     tokens = tokenizer.tokenize(text, split_affixes=False)
# #     return " ".join([token.text for token in tokens])
# #
# #
# # def tibetan_syllable_segment(text):
# #     """使用藏语音节符་进行分词"""
# #     # 在每个音节符后添加空格，合并多余空格，并去除首尾空格
# #     segmented = text.replace('་', '་ ')
# #     segmented = re.sub(r' +', ' ', segmented)  # 合并多个连续空格
# #     return segmented.strip()
# #
# # # ✅ 计算 BLEU
# # def compute_bleu1(pred: str, gold: str) -> float:
# #     """基于 SacreBLEU 源码的 BLEU 计算函数 (适配藏文分词)"""
# #
# #     # ✅ 初始化 BLEU 参数（与源码默认参数对齐）
# #     bleu = BLEU(
# #         smooth_method='add-k',  # 平滑方法
# #         smooth_value=1,  # add-k 的 k 值
# #         max_ngram_order=4,  # 显式指定 BLEU-4
# #         effective_order=True,  # 禁用动态 n-gram 阶数
# #         tokenize='none',  # 禁用内置分词（已手动分词）
# #         lowercase=False  # 不转小写
# #     )
# #
# #     # ✅ 藏文分词（假设 tibetan_syllable_segment 已正确定义）
# #     pred_tok = segment_tibetan_text(pred)
# #     gold_tok = segment_tibetan_text(gold)
# #
# #     # ✅ 调用 sentence_score（参考 SacreBLEU 源码逻辑）
# #     # 注意：references 必须是列表形式，即使只有一个参考
# #     bleu_score = bleu.sentence_score(pred_tok, [gold_tok])
# #
# #     return bleu_score.score
# #
# # # ✅ 计算 BLEU
# # def compute_bleu2(pred: str, gold: str) -> float:
# #     """基于 SacreBLEU 源码的 BLEU 计算函数 (适配藏文分词)"""
# #
# #     # ✅ 初始化 BLEU 参数（与源码默认参数对齐）
# #     bleu = BLEU(
# #         smooth_method='add-k',  # 平滑方法
# #         smooth_value=1,  # add-k 的 k 值
# #         max_ngram_order=4,  # 显式指定 BLEU-4
# #         effective_order=True,  # 禁用动态 n-gram 阶数
# #         tokenize='none',  # 禁用内置分词（已手动分词）
# #         lowercase=False  # 不转小写
# #     )
# #
# #     # ✅ 藏文分词（假设 tibetan_syllable_segment 已正确定义）
# #     pred_tok = tibetan_syllable_segment(pred)
# #     gold_tok = tibetan_syllable_segment(gold)
# #
# #     # ✅ 调用 sentence_score（参考 SacreBLEU 源码逻辑）
# #     # 注意：references 必须是列表形式，即使只有一个参考
# #     bleu_score = bleu.sentence_score(pred_tok, [gold_tok])
# #
# #     return bleu_score.score
# #
# #
# # # ✅ 计算 ROUGE（保持不变）
# # rouge_evaluator = Rouge()
# #
# # def compute_rouge(pred, gold):
# #     pred = __builtins__.str(pred).strip() or " "
# #     gold = __builtins__.str(gold).strip() or " "
# #
# #     pred_clean = segment_tibetan_text(pred)
# #     gold_clean = segment_tibetan_text(gold)
# #
# #     if not pred_clean.strip() or not gold_clean.strip():
# #         return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
# #
# #     try:
# #         return rouge_evaluator.get_scores(pred_clean, gold_clean)[0]
# #     except Exception as e:
# #         print(f"⚠️ ROUGE 计算失败: pred='{pred_clean}' gold='{gold_clean}'")
# #         return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
# #
# #
# # # 🚀 加载原始结果（不修改原文件）
# # with open("deepseek_outputs.json", "r", encoding="utf-8") as f:  # 读取原始文件
# #     original_results = json.load(f)
# #
# # # 创建新结果副本（避免修改原始数据）
# # new_results = [item.copy() for item in original_results]
# #
# # # 🔄 重新计算指标
# # print("🔄 开始重新计算评估指标...")
# # for item in tqdm(new_results, desc="📊 评估进度"):
# #     if item.get("model_output") == "ERROR":
# #         continue  # 跳过错误样本
# #
# #     # 重新计算指标
# #     rouge_scores = compute_rouge(item["model_output"], item["gold_answer"])
# #     bleu_scores1 = compute_bleu1(item["model_output"], item["gold_answer"])
# #     bleu_scores2 = compute_bleu2(item["model_output"], item["gold_answer"])
# #
# #     # 更新到新结果
# #     item.update({
# #         "rouge-1": rouge_scores["rouge-1"]["f"],
# #         "rouge-2": rouge_scores["rouge-2"]["f"],
# #         "rouge-l": rouge_scores["rouge-l"]["f"],
# #         "bleu1": bleu_scores1,
# #         "bleu2": bleu_scores2
# #     })
# #
# # # 💾 保存新输出文件
# # with open(output_results_file, "w", encoding="utf-8") as f:
# #     json.dump(new_results, f, ensure_ascii=False, indent=4)
# # print(f"✅ 新输出已保存至: {output_results_file}")
#
# # 🚀 加载原始结果（不修改原文件）
# with open("LCTW-deepseek_outputs.json", "r", encoding="utf-8") as f:  # 读取原始文件
#     new_results = json.load(f)
# # 📈 计算平均指标
# valid_items = [item for item in new_results if item.get("model_output") != "ERROR"]
# avg_scores = {
#     "ROUGE-1": sum(i["rouge-1"] for i in valid_items) / len(valid_items),
#     "ROUGE-2": sum(i["rouge-2"] for i in valid_items) / len(valid_items),
#     "ROUGE-L": sum(i["rouge-l"] for i in valid_items) / len(valid_items),
#     "bleu1": sum(i["bleu1"] for i in valid_items) / len(valid_items),
#     "bleu2": sum(i["bleu2"] for i in valid_items) / len(valid_items)
# }
#
# # 💾 保存新评估结果
# with open(eval_results_file, "w", encoding="utf-8") as f:
#     json.dump(avg_scores, f, ensure_ascii=False, indent=4)
# print(f"✅ 新评估结果已保存至: {eval_results_file}")
#
# # 📊 打印结果
# print("\n📊 最终平均指标：")
# print(json.dumps(avg_scores, indent=4, ensure_ascii=False))