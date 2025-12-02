import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader

# 你的文件夹路径
folder_path = "/home/caotiezheng/Downloads/票据信息抽取和问答挑战赛/receipts_scanned"

records = []
from tqdm import tqdm
# 遍历文件夹里的所有文件
for filename in tqdm(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, filename)

    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        # 拼接所有页
        full_text = "\n".join(doc.page_content for doc in documents)
        records.append({"filename": filename, "content": full_text})

    elif filename.lower().endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        records.append({"filename": filename, "content": full_text})

# 保存成 CSV，每个文件一行
df = pd.DataFrame(records)
df.to_csv("output.csv", index=False, encoding="utf-8-sig")

print(f"已处理 {len(records)} 个文件，保存到 output.csv")
