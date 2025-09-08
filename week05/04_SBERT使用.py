from sentence_transformers import SentenceTransformer

model = SentenceTransformer("../models/google-bert/bert-base-chinese/")

sentences = [
    "我今天很开心",
    "我今天很不开心",
    "我今天很幸福"
]

embeddings = model.encode(sentences)
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)

# modelscope download --model BAAI/bge-small-zh-v1.5  --local_dir BAAI/bge-small-zh-v1.5
model = SentenceTransformer("../models/BAAI/bge-small-zh-v1.5/")
embeddings = model.encode(sentences)
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)