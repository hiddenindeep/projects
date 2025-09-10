import yaml  # type: ignore
from elasticsearch import Elasticsearch  # type: ignore
import traceback

# 读取配置文件
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 从配置文件中提取ES的连接参数
es_host = config["elasticsearch"]["host"]
es_port = config["elasticsearch"]["port"]
es_scheme = config["elasticsearch"]["scheme"]
es_username = config["elasticsearch"]["username"]
es_password = config["elasticsearch"]["password"]

if es_username != "" and es_password != "":
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
        basic_auth=(es_username, es_password)
    )
else:
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
    )

embedding_dims = config["models"]["embedding_model"][
    config["rag"]["embedding_model"]
]["dims"]


def init_es():
    """
    检查es环境配置
    :return: 环境是否配置成功
    """
    if not es.ping():
        print("Could not connect to Elasticsearch.")
        return False

    # document_meta_mapping = {
    #     "mappings":{
    #         'properties': {
    #             'file_name': {
    #                 'type': 'text',
    #                 'analyzer': 'ik_max_word',
    #                 'search_analyzer': 'ik_max_word'
    #             },
    #             'abstract': {
    #                 'type': 'text',
    #                 'analyzer': 'ik_max_word',
    #                 'search_analyzer': 'ik_max_word'
    #             },
    #             'full_content': {
    #                 'type': 'text',
    #                 'analyzer': 'ik_max_word',
    #                 'search_analyzer': 'ik_max_word'
    #             }
    #         }
    #     }
    # }

    document_meta_mapping = {
        "mappings": {
            "properties": {
                "document_name": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
                "abstract": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
                "full_content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
                "file_path": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "knowledge_id": {"type": "keyword"}
            }
        }
    }

    try:
        # es.indices.delete(index='document_meta')
        if not es.indices.exists(index="document_meta"):
            es.indices.create(index='document_meta', body=document_meta_mapping)
    except:
        print(traceback.format_exc())
        print("Could not create index of document_meta.")
        return False

    # chunk_info_mapping = {
    #     'mappings': {  # Add 'mappings' here
    #         'properties': {
    #             'chunk_content': {
    #                 'type': 'text',
    #                 'analyzer': 'ik_max_word',
    #                 'search_analyzer': 'ik_max_word'
    #             },
    #             "embedding_vector": {
    #                 "type": "dense_vector",
    #                 "element_type": "float",
    #                 "dims": embedding_dims,
    #                 "index": True,
    #                 "index_options": {
    #                     "type": "int8_hnsw"
    #                 }
    #             }
    #         }
    #     }
    # }

    chunk_info_mapping = {
        "mappings": {
            "properties": {
                "document_id": {
                    "type": "keyword"   # 文档ID，不需要分词
                },
                "knowledge_id": {
                    "type": "keyword"   # 知识库ID，不需要分词
                },
                "page_number": {
                    "type": "integer"   # 页码
                },
                "chunk_id": {
                    "type": "integer"   # 每页的分块编号
                },
                "chunk_content": {
                    "type": "text",     # 分块内容，支持中文分词
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
                "chunk_images": {
                    "type": "keyword"   # 存文件路径/URL，数组也可以存
                },
                "chunk_tables": {
                    "type": "keyword"   # 存表格的标识或路径
                },
                "embedding_vector": {
                    "type": "dense_vector",
                    "dims": embedding_dims,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }


    try:
        # es.indices.delete(index='chunk_info')
        if not es.indices.exists(index="chunk_info"):
            es.indices.create(index='chunk_info', body=chunk_info_mapping)
    except:
        print(traceback.format_exc())
        print("Could not create index of chunk_info.")
        return False

    print("Successfully connected to Elasticsearch!")
    return True

init_es()
