from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from modelscope import ChineseCLIPProcessor, ChineseCLIPModel
from sklearn.preprocessing import normalize

device = "cuda"
bge_model = SentenceTransformer('/root/autodl-tmp/models/bge-small-zh-v1.5', device=device)

clip_model = ChineseCLIPModel.from_pretrained("/root/autodl-tmp/models/chinese-clip-vit-base-patch16")
clip_model.to(device)
clip_processor = ChineseCLIPProcessor.from_pretrained("/root/autodl-tmp/models/chinese-clip-vit-base-patch16")

# 相同模态文本检索文本用途的，bge编码
# bge模型计算时间: 0.2953503131866455 秒
def get_text_bge_features(texts):
    embeddings = bge_model.encode(texts, normalize_embeddings=True)
    return embeddings

# 跨模态的clip 对 图 的编码
# clip模型计算时间: 0.17340493202209473 秒
def get_clip_image_features(images):
    inputs = clip_processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    image_features = clip_model.get_image_features(**inputs)
    return normalize(image_features.data.cpu().numpy(), axis=1)

# 跨模态的clip 对 文 的编码
# clip模型计算时间: 0.01234126091003418 秒
def get_clip_text_features(texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_features = clip_model.get_text_features(**inputs)
    return normalize(text_features.data.cpu().numpy(), axis=1)

# 计算时间
import time

start_time = time.time()
get_text_bge_features(["你好"])
end_time = time.time()
print(f"bge模型计算时间: {end_time - start_time} 秒")

start_time = time.time()
get_clip_image_features([Image.open("pokemon.jpeg").resize((224, 224))])
end_time = time.time()
print(f"clip模型计算时间: {end_time - start_time} 秒")

start_time = time.time()
get_clip_text_features(["你好"])
end_time = time.time()
print(f"clip模型计算时间: {end_time - start_time} 秒")
