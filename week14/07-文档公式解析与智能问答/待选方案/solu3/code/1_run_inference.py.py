# src/1_run_inference.py
# 阶段一：执行LLM推理，生成详细日志
# -*- coding: utf-8 -*-
import os
import zipfile
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import json
import ollama
import re
import fitz
from sympy import sympify, SympifyError
from sympy.parsing.latex import parse_latex
from typing import Union
import traceback

# --- 1. 全局配置 ---
# 输入文件路径 
DOCUMENTS_ZIP_PATH = '../xfdata/documents.zip'
QUESTIONS_CSV_PATH = '../xfdata/question.csv'
# 句向量模型路径
LOCAL_EMBED_MODEL_PATH = '../user_data/models/shibing624-text2vec-base-chinese'
# 输出文件路径 
RUN_LOG_JSONL_PATH = '../user_data/run_log_final.jsonl'
# 模型配置 
LLM_MODEL_NAME = 'qwen3:8b'
TOP_K_RETRIEVAL = 3
# --- 2. 辅助函数 ---
def build_simple_knowledge_base():
    """从ZIP文件中解析所有PDF和Markdown文档，提取纯文本。"""
    print("--- 1. 正在构建纯文本知识库 ---")
    knowledge_base = []
    with zipfile.ZipFile(DOCUMENTS_ZIP_PATH, 'r') as zf:
        doc_files = [f for f in zf.namelist() if not f.startswith('__MACOSX') and f.endswith(('.pdf', '.md'))]
        for doc_path in tqdm(doc_files, desc="解析文档纯文本"):
            doc_id, full_text = os.path.basename(doc_path), ""
            try:
                with zf.open(doc_path) as doc_file:
                    content_bytes = doc_file.read()
                    if doc_path.endswith('.pdf'):
                        with fitz.open(stream=content_bytes, filetype="pdf") as doc:
                            for page in doc: 
                                full_text += page.get_text() + "\n"
                    elif doc_path.endswith('.md'):
                        full_text = content_bytes.decode('utf-8')
                knowledge_base.append({"id": doc_id, "full_text": full_text.strip()})
            except Exception:
                knowledge_base.append({"id": doc_id, "full_text": ""})
    return knowledge_base

def calculate_with_sympy(formula_string: str, params: dict) -> Union[float, None]:
    """使用SymPy对公式和参数进行精确计算。"""
    if not formula_string or not isinstance(params, dict): return None
    try:
        if '\\' in formula_string:
            if '=' in formula_string: formula_string = formula_string.split('=', 1)[1]
            formula_string = formula_string.replace('\\cdot', '*')
            expr = parse_latex(formula_string)
        else:
            if '=' in formula_string: formula_string = formula_string.split('=', 1)[1]
            elif '一' in formula_string: formula_string = formula_string.split('一', 1)[1]
            formula_string = formula_string.replace('（', '(').replace('）', ')').replace('×', '*').replace('÷', '/').replace('－', '-')
            expr = sympify(formula_string)
        
        safe_params = {}
        for k, v in params.items():
            try:
                safe_params[k] = float(v)
            except (ValueError, TypeError):
                continue
        
        free_symbols_str = {str(s) for s in expr.free_symbols}
        sympy_params = {k: v for k, v in safe_params.items() if k in free_symbols_str}

        if len(sympy_params) != len(expr.free_symbols):
             return None

        result = expr.subs(sympy_params).evalf()
        return float(result)
    except (SympifyError, SyntaxError, TypeError, Exception, ValueError):
        return None

# --- 3. 主流程 ---
def main():
    # 阶段 1: 构建知识库和检索索引
    knowledge_base = build_simple_knowledge_base()
    valid_knowledge_base = [doc for doc in knowledge_base if doc['full_text']]
    print(f"--- 知识库构建完毕，有效文档共 {len(valid_knowledge_base)} 篇 ---")
    
    print("--- 2. 加载句向量模型用于检索 ---")
    embed_model = SentenceTransformer(LOCAL_EMBED_MODEL_PATH)
    corpus_texts = [doc['full_text'] for doc in valid_knowledge_base]
    corpus_embeddings = torch.tensor(embed_model.encode(corpus_texts, show_progress_bar=True))

    questions_df = pd.read_csv(QUESTIONS_CSV_PATH, encoding='gbk')
    run_logs = []

    # 阶段 2: 定义Prompt模板
    prompt_template = """
    你是一位极其严谨且聪明的科学助理。你的核心任务是根据上下文信息，精准回答用户的问题，并以指定的JSON格式返回你的分析和结果。

    **第一步：选择最佳文档**
    仔细阅读用户问题和下面提供的所有“备选文档”。首先判断并选择出与用户问题**最相关**的那个文档。在你的思考过程中，明确指出你选择了哪个备选文档。

    **第二步：分析问题意图 (基于你选择的最佳文档)**
    判断问题意图属于以下三种之一：
    1.  **"numerical_calculation"**: 问题要求计算一个最终的数值。
    2.  **"formula_retrieval"**: 问题询问“如何”预测/评估，要求返回核心的数学模型。
    3.  **"conclusion_derivation"**: 问题询问“是否”会产生某种影响，要求基于模型进行逻辑推断。

    **第三步：根据意图，严格按照下面的JSON格式填充并返回结果**
    ```json
    {{
      "thought_process": "在这里描述你的完整思考步骤。例如：1. 对比三个备选文档，发现[备选文档 X]最相关，因为它讨论了...。2. 识别问题意图为'numerical_calculation'。3. 从最佳文档中定位到公式... 4. 从问题中提取参数... 5. (如果适用)进行参数猜测。6. 进行计算。",
      "question_intent": "这里填入你判断出的意图",
      "structured_data": {{
        "formula": "这里填入你从**最佳文档**中找到的核心数学公式。",
        "parameters": {{ "变量名1": "数值1" }},
        "guessed_parameters": {{ "缺失变量名A": "猜测值A", "说明": "陈述你进行猜测的依据。" }}
      }},
      "llm_answer": "这里填入你生成的最终答案。计算结果（浮点数）、公式（Markdown LaTeX）或结论文本。"
    }}
    ```
    
    **第四步：关于“数值计算”的特殊指令**
    - **检查参数完整性**: 基于**最佳文档**中的公式，检查问题是否提供了所有必需参数。
    - **“有根据的猜测”**: 仅在缺少少量参数且能做出合理猜测时，填充`"guessed_parameters"`字段。如果无法猜测，在`"llm_answer"`中返回 "Unable to determine"。

    **重要规则：**
    - 你的回答必须是且仅是一个**格式完全正确的JSON对象**。
    - 你的所有分析和回答都必须基于你**首先选择出的那个最佳文档**。

    ---
    [上下文 - 包含 {k} 个备选文档]
    {context}
    ---
    [用户问题]
    {question}
    ---
    [你的JSON回答]
    """

    # 阶段 3: 迭代处理问题
    print(f"--- 3. 开始使用 {LLM_MODEL_NAME} (Top-{TOP_K_RETRIEVAL}) 回答问题 ---")
    with open(RUN_LOG_JSONL_PATH, 'a', encoding='utf-8') as log_file:
        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc=f"使用 {LLM_MODEL_NAME} (Top-{TOP_K_RETRIEVAL}) 回答问题中"):
            query_text = row['question']
            
            query_embedding = torch.tensor(embed_model.encode(query_text))
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_k_scores, top_k_indices = torch.topk(cos_scores, k=TOP_K_RETRIEVAL)

            retrieved_contexts, retrieved_doc_ids = [], []
            for i in range(TOP_K_RETRIEVAL):
                doc_index = top_k_indices[i].item()
                retrieved_doc_ids.append(valid_knowledge_base[doc_index]['id'])
                doc_text = valid_knowledge_base[doc_index]['full_text']
                retrieved_contexts.append(f"--- [备选文档 {i+1}] ---\n{doc_text}")
            full_context = "\n\n".join(retrieved_contexts)
            
            prompt = prompt_template.format(k=TOP_K_RETRIEVAL, context=full_context, question=query_text)
            
            llm_raw_output, final_answer = "{}", "Unable to determine"

            try:
                response = ollama.chat(model=LLM_MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
                llm_raw_output = response['message']['content'].strip()
                
                llm_json = None
                try:
                    json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\})', llm_raw_output)
                    if json_match:
                        json_str = json_match.group(1) or json_match.group(2)
                        llm_json = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"\n警告: JSON解析失败，启动正则抢救模式。")
                    llm_json = {}
                    intent_match = re.search(r'"question_intent":\s*"([^"]+)"', llm_raw_output)
                    if intent_match: llm_json["question_intent"] = intent_match.group(1)
                    answer_match = re.search(r'"llm_answer":\s*("?)(.*?)\1\s*[,}]', llm_raw_output, re.DOTALL)
                    if answer_match: llm_json["llm_answer"] = answer_match.group(2).strip()
                    formula_match = re.search(r'"formula":\s*"([^"]+)"', llm_raw_output)
                    if formula_match:
                        llm_json.setdefault("structured_data", {})["formula"] = bytes(formula_match.group(1), "utf-8").decode("unicode_escape")

                if llm_json:
                    intent = llm_json.get("question_intent")
                    structured_data = llm_json.get("structured_data", {}) or {}
                    llm_answer = llm_json.get("llm_answer")

                    if intent == "numerical_calculation":
                        formula = structured_data.get("formula")
                        params = structured_data.get("parameters", {}) or {}
                        guessed_params = structured_data.get("guessed_parameters", {}) or {}
                        guessed_params.pop("说明", None)
                        final_params = {**guessed_params, **params}
                        
                        sympy_result = calculate_with_sympy(formula, final_params)
                        
                        if sympy_result is not None:
                            final_answer = sympy_result
                        else:
                            try: final_answer = float(llm_answer)
                            except (ValueError, TypeError, AttributeError): final_answer = 0.0
                    elif intent in ["formula_retrieval", "conclusion_derivation"]:
                        if llm_answer is not None: final_answer = llm_answer
                    else:
                        if isinstance(llm_answer, (int, float, str)) and llm_answer != "": final_answer = llm_answer
                
            except Exception as e:
                final_answer = "Error in processing"
                print(f"处理问题时发生严重错误: {e}\nLLM原始输出:\n---\n{llm_raw_output}\n---")
                traceback.print_exc()

            log_record = {"question": query_text, "retrieved_doc_ids": retrieved_doc_ids, "llm_raw_output": llm_raw_output, "final_answer": final_answer}
            log_file.write(json.dumps(log_record, ensure_ascii=False) + '\n')
            run_logs.append(log_record)

    print(f"\n--- 4. 推理完成，详细日志已保存到 '{RUN_LOG_JSONL_PATH}' ---")
    print("--- 现在可以运行 '2_generate_submission.py' 来生成最终提交文件。 ---")
    
if __name__ == '__main__':
    main()