from sentence_transformers import SentenceTransformer

sentences = [
    "我喜欢机器学习", # 768 512
    "我喜欢深度学习",
    "我今天心情很不错"
]

query = "我今天很开心"

model = SentenceTransformer("../../../../../models/BAAI/bge-small-zh-v1.5/") # sentence-bert 微调之后的
embeddings = model.encode(sentences)
query_embedding = model.encode([query])
print(embeddings.shape)

similarities = model.similarity(query_embedding, embeddings)
print(similarities)

best_idx = similarities.argmax().item()

print(f"你的输入：{query}")
print(f"匹配到的句子：{sentences[best_idx]}")
print(f"相似度：{similarities[0, best_idx].item():.4f}")

print("全部句子的相似度：")
for i, s in enumerate(sentences):
    print(f"  {s}  相似度={similarities[0, i].item():.4f}")