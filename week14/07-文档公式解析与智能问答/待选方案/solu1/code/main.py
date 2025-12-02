'''
pdf 公式计算，但是公式库有几百个
根据问题召回topN 相关的公式，然后让大模型来根据公式计算
'''
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import glob
import torch
import json
from tqdm import tqdm
import re

##这个模型是直接用的魔搭的api，最开始使用的是Qwen3-32B,后面使用了Qwen/Qwen3-235B-A22B-Thinking-2507，差别不大
api_key = '*****'
llm = ChatOpenAI(
    # model="Qwen/Qwen3-32B",
    model="Qwen/Qwen3-235B-A22B-Thinking-2507",
    openai_api_key=api_key,  # ModelScope Token
    openai_api_base='https://api-inference.modelscope.cn/v1/',
    streaming=True,
)


##文档数据读取
md_data = pd.read_excel("./user_data/tmp_data/md.xlsx")
md_data["len"] = md_data["text"].apply(lambda e: len(e))
print(md_data["len"].describe())
contents = []
for f in glob.glob("./xfdata/documents/*.md"): #markdown是解析的结果
    with open(f, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        contents.append(content)

contents.extend(md_data["text"].values)
print(len(contents))

question = pd.read_csv("./xfdata/question.csv")

model_name = './user_data/model/Qwen3-Embedding-0.6B' # qwen3 embedding， 和bge 相比更加强大，但速度更慢。

embedder = SentenceTransformer(model_name, trust_remote_code=True)
corpus = contents
corpus_embeddings = embedder.encode_document(corpus, convert_to_tensor=True)
top_k = min(8, len(corpus))

# 技术路线1，但没有生成代码，直接借助qwen3 thinking 生成结果
# 提问， 待选公式1、2、3...8  汇总得到提示词 -》 qwen3-thinking -》 结果
# 优点： 借助qwen3 thinking 模型进一步选择合适的公式， 因为RAG的检索和排序存在精度误差。
# 缺点： 非常消耗token，没有生成代码，依赖qwen3 thinking 直接生成答案。
template = '''
   你是一位理工科的博士，下面给出问题query和多个参考公式列表，需要你给出最终的计算结果。
   具体做法：
   1.参考公式存在多个，但是真的用来计算的最多只有一个。需要根据query来选择合适的公式，建模的背景和query是相符合的,保证计算公式的因变量都在query里面提供了，query里面要的目标值和公式是一致的.
   如果存在多个合适的公式，选择第一个。然后根据公式计算得到结果。计算结果需要数值,比如是小数或者整数，不要是各种无理数，分数。
   如果最合适的公式里面有缺失1-2个因变量，而这些因变量可以有个常识的估计值，也需要给出计算后的估计值。如果变量估计值无法知道，则不需要计算。
   2.如果问题可以不依赖参考公式，只要根据query就可以，可以直接计算。
   3.如果问题不明确或者无法根据提供的公式介绍给出具体结果，则返回无答案
    例子：在抗震设计中，如何利用单自由度系统评估Case1所描述结构在地震作用下的动态位移响应？ 该问题提到的case没有明确参数
   4.如果问题的结果不是数值，比如是否可能，可能回复1，不能回复0。其他情况下回复无答案
   5.需要给出具体的计算公式，计算过程和最终的计算结果.计算结果以json格式输出，例子"answer":'100'，不要包含单位，百分比例也不要百分号,只要数值。
   **参考公式：**
    {info}
   问题：{query}

   '''

prompt = PromptTemplate(
    template=template,
    input_variables=["query", "info"])


def qa(query):
    # 对于每个提问，qwen3 embeding 进行编码
    query_embedding = embedder.encode_query(query, convert_to_tensor=True)

    # 计算每个提问与 每个公式之间的相似度
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]

    # 保留最相关的8个公式
    scores, indices = torch.topk(similarity_scores, k=top_k)
    info = ""
    for score, idx in zip(scores, indices):
        # print(f"(Score: {score:.4f})", corpus[idx])
        info += corpus[idx] + "\n"

    message = prompt.format_prompt(query=query, info=info)

    result = llm.invoke(message)
    return result.content



# 结果写入jsonl，这个answer还包含思考过程，所以需要后处理下。当然为了简化也可以修改prompt让只返回结果即可
with  open("./user_data/tmp_data/qa_results.jsonl", encoding="utf-8") as w:
    # 遍历所有问题
    for question in tqdm(question["question"].values):
            response = qa(question)
            # 构建结果字典
            result = {
                "question": question,
                "answer": response}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')




results = []

# 更强的正则：匹配独立的 {"answer": "..."} 结构，支持转义引号
pattern = re.compile(r'\{\s*"answer"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', re.DOTALL)


with open("./user_data/tmp_data/qa_results.jsonl", encoding="utf-8") as w:
    lines = w.readlines()
    for line in lines:
        try:
            line = line.strip()
            #print(line)
            if not line:
               continue


            data = json.loads(line)
            question = data["question"]
            answer_str = data["answer"]

            #final_answer = re.findall(r"【(.*)】",str(answer_str)
            #final_answer = re.findall(r'\{"answer":\s*"([^"]*)"\}', data["answer"])
            final_answer =re.search(r'\{\s*"answer"\s*:\s*"([^"]*)"\s*\}', data["answer"])

            if final_answer:
                 final_answer=final_answer[0]
            elif "无答案" in answer_str:
                final_answer=10
            else:
                final_answer=answer_str
            final_answer=json.loads(final_answer)["answer"]
            if final_answer=="无答案":
                final_answer=10
            results.append({
                "question": question,
                "answer": final_answer
               })
        except Exception as e:
            results.append({
                "question": question,
                "answer": 10
            })


# 转为 DataFrame
df = pd.DataFrame(results)
df.fillna(10,inplace=True)

# 保存为 Excel 或 CSV
df.to_csv("./user_data/submit.csv", index=False)
