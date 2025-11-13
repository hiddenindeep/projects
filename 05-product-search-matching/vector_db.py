from pymilvus import MilvusClient, DataType # milvus客户端
import traceback
from PIL import Image
from typing import Optional, List

import nlp_models

# 创建客户端
client = MilvusClient(
    uri="https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
    token="9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b" 
)  

def insert_product(image_path: str, title: str):
    # 向milvus 插入一条记录
    
    # 提取特征
    try:
        image = Image.open(image_path)
        image_clip_features = list(nlp_models.get_clip_image_features([image])[0])
    except Exception as e:
        traceback.print_exc()
        image_clip_features = [0]*512

    # 提取特征
    try:
        title_bge_embedding = list(nlp_models.get_text_bge_features([title])[0])
    except Exception as e:
        traceback.print_exc()
        title_bge_embedding = [0] *512

    # 提取特征
    try:
        title_clip_embedding = list(nlp_models.get_clip_text_features([title])[0])
    except Exception as e:
        traceback.print_exc()
        title_clip_embedding = [0]*512
    
    try:
        data = [
            {
                "image_clip_vector": image_clip_features,
                "text_bge_vector": title_bge_embedding,
                "text_clip_vector": title_clip_embedding,
                "image_path": image_path,
                "title": title
            }
        ]
        insert_result = client.insert(
            collection_name="product_new",
            data=data
        )
        milvus_primary_key = insert_result["ids"][0] # 主键
        return True, milvus_primary_key

    except Exception as e:
        traceback.print_exc()
        return False, None


def search_product(image: Optional[Image.Image] = None, title: Optional[str] = None, task: str = "text2text", top_k: int = 10):
    try:
        if image is None and title is None:
            return False, None
        
        if image is not None:
            image_clip_features = list(nlp_models.get_clip_image_features([image])[0])
        else:
            image_clip_features = None

        if title is not None:
            title_bge_embedding = list(nlp_models.get_clip_text_features([title])[0])
        else:
            title_bge_embedding = None
        
        if image_clip_features:
            data = [image_clip_features]
        else:
            data = [title_bge_embedding]
        
        if task in ["text2text"]:
            anns_field = "text_clip_vector"
        elif task in ["text2image"]:
            anns_field = "image_clip_vector"
        elif task in ["image2text"]:
            anns_field = "text_clip_vector"
        elif task in ["image2image"]:
            anns_field = "image_clip_vector"
        else:
            return False, None

        # milvus的搜索
        results = client.search(
            collection_name="product_new", # 对哪一个collection搜索
            anns_field=anns_field, # 对哪一个一段进行排序
            data=data, # 传入的查询 向量
            limit=top_k, # 返回多少个

            # 打分方法
            search_params={
            "metric_type": "COSINE"
            }
        )
        return True, results

    except Exception as e:
        traceback.print_exc()
        return False, None


def delete_product(ids: List[int]):
    try:
        client.delete(
            collection_name="product_new",
            ids=ids
        )
        return True
    except Exception as e:
        traceback.print_exc()
        return False