# 赛题：文档公式解析与智能问答挑战赛 - 解决方案说明

## 解决方案概述

本方案采用了一种基于**检索增强生成 (Retrieval-Augmented Generation, RAG)** 的先进架构，并以本地部署的开源大语言模型 (LLM) 作为核心推理引擎。整体流程分为两个主要阶段：

1.  **文档解析与检索**:
    *   使用 `PyMuPDF` 从所有文档中快速、准确地提取纯文本内容，构建一个包含所有文档信息的知识库。
    *   利用 `sentence-transformers` 库加载一个强大的中文句向量模型 (`shibing624/text2vec-base-chinese`)，对知识库中的所有文档进行向量化，建立语义索引。
    *   对于每一个问题，通过计算向量余弦相似度，检索出与问题语义最相关的 **Top-3** 个候选文档。

2.  **LLM 推理与后处理**:
    *   使用 **Ollama** 在本地部署并运行业界领先的开源模型 **`Qwen3-8B`**。
    *   设计了一个精巧的 **Prompt (提示)**，引导 `Qwen3` 模型对问题和检索到的文档进行深度分析，并输出一个包含其**思考过程、意图判断和多种形式答案**的JSON对象。此过程的详细日志被记录在 `run_log_final.jsonl` 中。
    *   最后，通过一个**后处理脚本**，对第一阶段生成的复杂回答进行**智能映射**，将所有答案（包括数值、公式、文本结论等）转换为符合赛题评测要求的、统一的**数值格式**，生成最终的提交文件。

## 环境配置

本项目代码已在以下环境中测试通过：

*   操作系统: Windows 11
*   Python: 3.11 (通过 Conda 管理)
*   核心依赖: PyTorch, transformers, sentence-transformers, ollama, PyMuPDF, sympy

**步骤 1: 安装 Conda**
请从 Anaconda 官网下载并安装 Conda。

**步骤 2: 创建并激活 Conda 环境**
```bash
conda create --name docformula_qa python=3.11
conda activate docformula_qa
```

**步骤 3: 安装 Python 依赖**
请使用项目根目录下的 `requirements.txt` 文件安装所有必要的库：
```bash
pip install -r requirements.txt
```

**步骤 4: 安装并配置 Ollama**
1.  从 [https://ollama.com](https://ollama.com) 下载并安装 Ollama。
2.  在终端中运行以下命令，拉取本项目使用的 `Qwen3-8B` 模型（约4.7GB）：
    ```bash
    ollama pull qwen3:8b
    ```
    请确保 Ollama 服务在后台正在运行。

## 如何运行


本项目遵循官方代码打包规范，使用 `test.sh` 作为统一的执行入口。

**运行命令:**
在项目的根目录下，执行 `code` 文件夹中的 `test.sh` 脚本：
```bash
bash code/test.sh
```
**执行流程详解:**
`test.sh` 脚本会自动按顺序完成以下两个阶段：

1.  **阶段一 (`1_run_inference.py`)**:
    *   **输入**: 从 `xfdata/` 目录读取 `documents.zip` 和 `question.csv`。
    *   **模型**: 从 `user_data/models/` 加载句向量模型。
    *   **推理**: 调用 Ollama 的 `Qwen3-8B` 模型进行推理。
    *   **输出**: 在 `user_data/` 目录下生成详细的日志文件 `run_log_final.jsonl`。

2.  **阶段二 (`2_generate_submission.py`)**:
    *   **输入**: 读取 `user_data/run_log_final.jsonl` 和 `xfdata/question.csv`。
    *   **处理**: 应用后处理规则，将模型的复杂回答转换为纯数值格式。
    *   **输出**: 在 `prediction_result/` 目录下生成最终的提交文件，命名为 `result`。

## 最终提交文件结构 (Zip包解压后)

```
/
|-- code/
|   |-- 1_run_inference.py
|   |-- 2_generate_submission.py
|   |-- test.sh  <-- 预测执行脚本 (必选)
|
|-- user_data/
|   |-- models/
|       |-- shibing624-text2vec-base-chinese/
|           |-- (精简后的句向量模型文件)
|
|-- prediction_result/
|   |-- (此目录初始为空，由代码生成 result 文件)
|
|-- README.md
|-- requirements.txt```
```
