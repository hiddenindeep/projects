CATEGORY_NAME = [
    '差评', '好评'
]

BERT_MODEL_PKL_PATH = "assets/weights/best_model.pt"
BERT_MODEL_PERTRAINED_PATH = "../models/google-bert/bert-base-chinese/"

#本地
# LLM_OPENAI_SERVER_URL = f"http://127.0.0.1:11434/v1" # ollama
# LLM_OPENAI_API_KEY = "None"
# LLM_MODEL_NAME = "qwen2.5:0.5b"
#云端
LLM_OPENAI_API_KEY = "sk-b2fcae19cd1f4a7dbe605ce9fc8ef3be"
LLM_OPENAI_SERVER_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = "qwen-plus"
