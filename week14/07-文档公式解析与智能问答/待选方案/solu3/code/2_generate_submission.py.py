# src/2_generate_submission.py
# 阶段二：根据日志进行后处理，生成最终提交文件
# -*- coding: utf-8 -*-
import pandas as pd
import json
import re
from tqdm import tqdm

# --- 1. 配置  ---
# 输入文件路径 
RUN_LOG_JSONL_PATH = '../user_data/run_log_final.jsonl'
QUESTIONS_CSV_PATH = '../xfdata/question.csv'
# 输出文件路径 
FINAL_SUBMISSION_CSV_PATH = '../prediction_result/result.csv'
# --- 2. 核心后处理函数 ---
def intelligent_answer_cleaning(log_record: dict) -> float:
    """
    根据第一轮推理的日志记录，将复杂的LLM回答智能地映射为单一的数值。
    这是被验证过的、效果最好的后处理策略。
    """
    # 获取日志中的关键信息
    final_answer_from_log = log_record.get("final_answer")
    llm_raw_output = log_record.get("llm_raw_output", "{}")
    
    # 规则1: 如果第一轮的最终答案已经是数字，直接采纳
    if isinstance(final_answer_from_log, (int, float)):
        return float(final_answer_from_log)

    # 规则2: 如果不是数字，则解析LLM的思考过程，根据其意图进行映射
    intent = None
    llm_answer_text = ""
    try:
        # 尝试从LLM的原始输出中解析出JSON
        json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\})', llm_raw_output)
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            llm_json = json.loads(json_str)
            intent = llm_json.get("question_intent")
            # 将llm_answer转换为小写字符串以便关键词匹配
            llm_answer_text = str(llm_json.get("llm_answer", "")).lower()
        else:
             # 如果连JSON都找不到，无法判断意图，返回安全值
            return 0.0
    except (json.JSONDecodeError, Exception):
        # 如果JSON格式错误，同样返回安全值
        return 0.0

    # 规则3: 根据解析出的意图，应用不同的数值映射规则
    if intent == "formula_retrieval":
        # 对于“公式检索”意图，返回 1.0 代表“成功识别并提供了模型”
        # 除非LLM明确表示无法确定
        if "unable to determine" in str(final_answer_from_log).lower():
            return 0.0
        return 1.0
        
    elif intent == "conclusion_derivation":
        # 对于“结论推导”意图，根据关键词映射为 1.0, -1.0, 或 0.0
        positive_keywords = ["是", "会", "出现", "正面", "增长", "增加", "高于"]
        negative_keywords = ["否", "不会", "不会出现", "负面", "受限", "减少", "低于"]
        
        if any(keyword in llm_answer_text for keyword in positive_keywords):
            return 1.0
        elif any(keyword in llm_answer_text for keyword in negative_keywords):
            return -1.0
        else:
            # 如果没有明确的倾向性词汇，返回中性值
            return 0.0
    
    # 规则4: 对于所有其他情况 (如计算失败、意图不明等)，返回安全的兜底值
    return 0.0

# --- 3. 主流程 ---
def main():
    """
    读取第一轮的日志文件，应用后处理规则，并生成最终的、顺序正确的提交文件。
    """
    print(f"--- 正在从 '{RUN_LOG_JSONL_PATH}' 读取日志并生成最终提交文件 ---")
    
    try:
        # 1. 加载原始问题文件，以确保最终提交的顺序是正确的
        original_questions_df = pd.read_csv(QUESTIONS_CSV_PATH, encoding='gbk')
        
        # 2. 加载所有日志记录，并存入一个字典以便快速查找
        all_logs = {json.loads(line)['question']: json.loads(line) for line in open(RUN_LOG_JSONL_PATH, 'r', encoding='utf-8')}
    
    except FileNotFoundError as e:
        print(f"错误: 找不到必需的输入文件: {e}。请确保已成功运行 '1_run_inference.py'。")
        return
        
    # 3. 遍历原始问题，逐条应用后处理逻辑
    submission_data = []
    for question in tqdm(original_questions_df['question'], desc="后处理日志中"):
        log_record = all_logs.get(question)
        
        if log_record:
            # 如果找到了对应问题的日志，就进行智能清洗
            cleaned_answer = intelligent_answer_cleaning(log_record)
            submission_data.append({"question": question, "answer": cleaned_answer})
        else:
            # 如果某个问题因为某种原因在第一轮就没有日志，则兜底为0.0
            submission_data.append({"question": question, "answer": 0.0})

    # 4. 创建 DataFrame 并保存为最终的提交文件
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(FINAL_SUBMISSION_CSV_PATH, index=False)
    print(f"最终提交文件 '{FINAL_SUBMISSION_CSV_PATH}' 已成功生成！")

if __name__ == '__main__':
    main()