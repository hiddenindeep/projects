## 数据库设计

### 元信息存储（mysql、sqlite）

- 知识库（knowledge_database）

| 字段      | 字段类型 | 字段含义         |
| --------- | -------- | ---------------- |
| knowledge_id        | bigint   | 知识库主键       |
| title     | varchar  | 名称             |
| category  | varchar  | 类型             |
| author_id | bigint   | 用户主键（外键） |
| create_dt | datetime | 创建时间         |
| update_dt | datetime | 更新时间         |

- 知识文档（knowledge_document）

| 字段         | 字段类型 | 字段含义           |
| ------------ | -------- | ------------------ |
| document_id           | bigint   | 文档主键           |
| title        | varchar  | 文档名称           |
| category     | varchar  | 文档类型           |
| knowledge_id | bigint   | 知识库主键（外键） |
| file_path    | varchar  | 储存地址           |
| file_type    | varchar  | 数据类型           |
| create_dt    | datetime | 创建时间           |
| update_dt    | datetime | 更新时间           |


### 文档全文与向量存储（es）

- **document_meta** 存储知识文档的元信息，包括文件路径、名称、摘要和全部内容等。

| 字段名           | 数据类型     | 字段说明               |
|------------------|--------------|------------------------|
| `document_id`    | `int`       | 文档元信息的唯一标识符   |
| `document_name`      | `text`       | 文档的名称              |
| `knowledge_id`   | `text`       | chunk 所属知识库的标识符 |
| `file_path`      | `text`       | 文档的存储路径          |
| `abstract`       | `text`       | 文档的摘要信息          |

- **chunk_info** 存储分块的文档内容，包含 chunk 的文字内容、图片、表格地址等。

| 字段名           | 数据类型     | 字段说明               |
|------------------|--------------|------------------------|
| `chunk_id`       | `text`       | chunk 的唯一标识符      |
| `document_id`    | `int`       | 文档元信息的唯一标识符   |
| `knowledge_id`   | `text`       | chunk 所属知识库的标识符 |
| `page_number`    | `integer`    | chunk 所在文档的页码    |
| `chunk_content`  | `text`       | chunk 的文字内容        |
| `chunk_images`   | `text`       | chunk 相关图片的路径     |
| `chunk_tables`   | `text`       | chunk 相关表格的路径     |
| `embedding_vector` | `array<float>` | chunk 的语义编码结果   |

# Create an index
PUT document_meta
{
  "mappings": {
    "properties": {
      "document_id": { "type": "integer" },
      "document_name": { "type": "text" },
      "knowledge_id": { "type": "keyword" },
      "file_path": { "type": "text" },
      "abstract": { "type": "text" }
    }
  }
}

PUT chunk_info
{
  "mappings": {
    "properties": {
      "chunk_id": { "type": "keyword" },
      "document_id": { "type": "integer" },
      "knowledge_id": { "type": "keyword" },
      "page_number": { "type": "integer" },
      "chunk_content": { "type": "text" },
      "chunk_images": { "type": "text" },
      "chunk_tables": { "type": "text" },
      "embedding_vector": {
        "type": "dense_vector", 
        "dims": 512          //bge-small-zh-v1.5的向量维度
      }
    }
  }
}

# Add a document to my-index
POST document_meta/_doc
{
  "document_id": 1,
  "document_name": "机器学习基础.pdf",
  "knowledge_id": "ml001",
  "file_path": "/docs/ml/机器学习基础.pdf",
  "abstract": "这是一份关于机器学习基础知识的文档。"
}

POST chunk_info/_doc
{
  "chunk_id": "c001",
  "document_id": 1,
  "knowledge_id": "ml001",
  "page_number": 1,
  "chunk_content": "机器学习是一门研究如何让计算机从数据中学习的学科。",
  "chunk_images": "/images/ml001_page1.png",
  "chunk_tables": "/tables/ml001_page1.csv",
  "embedding_vector": []  // 这里是模型生成的向量
}

# Perform a search in my-index
GET /chunk_info/_search?q="机器学习"



### 知识节点与图片存储（neo4j）

TODO

## 开发资料

- https://elasticsearch-py.readthedocs.io/en/latest/


# 04-government-advanced-rag 项目结构与流程详解

## 项目概述
这是一个基于政府文档的高级检索增强生成（RAG）系统，结合了向量检索、重排序和大语言模型技术，用于提供准确的政府文档问答服务。

## 项目结构

### 1. 配置文件
- `config.yaml`：核心配置文件
  - 设备配置（CUDA）
  - Elasticsearch连接配置
  - 数据库配置（SQLite）
  - 模型配置（嵌入模型、重排序模型、LLM）
  - RAG系统参数（块大小、重叠度等）

### 2. 核心模块

#### 2.1 rag_api.py
RAG系统的核心实现，包含：
- `RAG`类：主要功能实现
  - 文本嵌入：`get_embedding()`
  - 重排序：`get_rank()`
  - 对话生成：`chat()`
- 提示词模板：`BASIC_QA_TEMPLATE`

#### 2.2 es_api.py
Elasticsearch接口实现：
- ES连接初始化：`init_es()`
- 向量检索相关功能
- 配置参数：端口、用户名等

#### 2.3 db_api.py
数据库操作接口：
- 数据库连接管理
- 会话管理
- 数据模型创建

#### 2.4 router_schemas.py
API接口定义：
- 请求/响应模型定义
- 包括嵌入、重排序、知识库、文档和RAG等接口

#### 2.5 main.py
FastAPI应用入口：
- 应用初始化
- 服务器启动配置

### 3. 测试文件（test/）
- `test_api.py`：API接口测试
- `test_db.py`：数据库功能测试
- `test_es.py`：Elasticsearch功能测试
- `test_rag.py`：RAG系统测试

### 4. 资源文件
- `assets/`：存储模型和其他资源
- `upload_files/`：文件上传目录

## 项目流程

### 1. 系统初始化
1. 加载配置文件（`config.yaml`）
2. 初始化Elasticsearch连接（`es_api.py`）
3. 初始化数据库连接（`db_api.py`）
4. 加载嵌入模型和重排序模型（`rag_api.py`）

### 2. 文档处理流程
1. 文档上传（通过`DocumentRequest`接口）
2. 文档分块处理（chunk_size=256, chunk_overlap=20）
3. 生成文档向量表示（使用bge-small-zh-v1.5模型）
4. 存储到Elasticsearch和数据库

### 3. 查询处理流程
1. 接收用户查询（通过`RAGRequest`接口）
2. 查询向量生成
3. 相似文档检索（从Elasticsearch）
4. 结果重排序（可选，使用bge-reranker-base）
5. 生成回答（使用GLM-4-Air模型）

### 4. 关键特性
1. 模块化设计：各功能模块独立，便于维护
2. 可配置性：通过配置文件灵活调整参数
3. 多模型支持：支持不同的嵌入模型、重排序模型和LLM
4. 完整的测试覆盖：包含各模块的单元测试

## 技术栈
1. 向量检索：Elasticsearch
2. 数据库：SQLite
3. 嵌入模型：BGE-small-zh-v1.5
4. 重排序模型：BGE-reranker-base
5. 大语言模型：GLM-4-Air
6. Web框架：FastAPI
7. 深度学习框架：PyTorch

这个项目实现了一个完整的政府文档问答系统，通过检索增强的方式，确保回答的准确性和可靠性。系统设计合理，模块化程度高，便于维护和扩展。