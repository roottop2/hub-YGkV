import json
import datetime
from itertools import product

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# 连接ElasticSearch
es = Elasticsearch("http://localhost:9200")

# 连接bge模型, 将句子转为512维度向量
model = SentenceTransformer("../../../../../models/BAAI/bge-small-zh-v1.5")

# 创建表
index_name = "products_index"
if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "product_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "description": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "category": {"type": "keyword"},
                    "brand": {"type": "keyword"},
                    "price": {"type": "float"},
                    "stock": {"type": "integer"},
                    "on_sale": {"type": "boolean"},
                    "created_at": {"type": "date"},
                    "text_vector": {
                        "type": "dense_vector",
                        "dims": 512,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
    )

# 定义商品表
try:
    with open("./products.json", "r", encoding="utf-8") as fi:
        PRODUCTS = json.load(fi)
except Exception as e:
    print("读取数据文件失败: ", e)
    exit()

for i, product in enumerate(PRODUCTS):
    embedding_text = f"{product["title"]}: {product["description"]}"
    vector = model.encode(embedding_text).tolist()

    es.index(
        index=index_name,
        id=product["product_id"],
        document={
            "product_id": product["product_id"],
            "title": product["title"],
            "description": product["description"],
            "category": product["category"],
            "brand": product["brand"],
            "price": product["price"],
            "stock": product["stock"],
            "on_sale": product["on_sale"],
            "created_at": product["created_at"],
            "text_vector": vector
        }
    )

    print(f"插入第{i}条商品:{product["product_id"]}成功")

print(f"插入成功, 总计{len(PRODUCTS)}条记录")

es.indices.refresh(index=index_name)

def print_results(response):

    if response is None:
        print("查询结果为空")
        return

    hits = response["hits"]["hits"]

    print(f"本次返回 {len(hits)} 条结果：")

    for position, hit in enumerate(hits, start=1):
        source = hit["_source"]

        print("-" * 60)
        print(f"排名：{position}")
        print(f"得分：{hit.get('_score')}")
        print(f"商品编号：{source['product_id']}")
        print(f"商品名称：{source['title']}")
        print(f"商品分类：{source['category']}")
        print(f"商品品牌：{source['brand']}")
        print(f"商品价格：{source['price']}")
        print(f"商品库存：{source['stock']}")

# 全文检索
def full_text_search(es, keyword):
    query = {
        "query": {
            "multi_match": {
                "query": keyword,
                "fields": ["title", "description"]
            }
        },
        "size": 5
    }

    return es.search(index=index_name, body=query)

# 条件检索
def filter_products(es, category:str=None, max_price:float=None, on_sale:bool=None, min_stock:int=None):

    filters = []

    if category is not None:
        filters.append({"term": {"category": category}})
    if max_price is not None:
        filters.append({"range": {"price": {"lte": max_price}}})
    if on_sale is not None:
        filters.append({"term": {"on_sale": on_sale}})
    if min_stock is not None:
        filters.append({"range": {"stock": {"gte": min_stock}}})

    if len(filters) == 0:
        print("查询条件为空!")
        return None

    query = {
        "query": {
            "bool": {
                "must": {"match_all": {}},
                "filter": filters
            }
        }
    }

    return es.search(index=index_name, body=query)

# 向量检索
def vector_search(es, text, k:int=3):

    query_vector = model.encode(text).tolist()
    num_candidates = k * 2
    query = {
        "knn": {
            "field": "text_vector",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": num_candidates
        },
        "_source" :{"excludes": ["text_vector"]}
    }

    return es.search(index=index_name, body=query)


if __name__ == "__main__":
    print("\n========== 全文检索 ==========")
    response = full_text_search(
        es,
        keyword="降噪耳机"
    )
    print_results(response)

    print("\n========== 条件筛选 ==========")
    response = filter_products(
        es,
        category="电子产品",
        max_price=2000,
        on_sale=True,
        min_stock=1
    )
    print_results(response)

    print("\n========== 向量检索 ==========")
    response = vector_search(
        es,
        text="我每天坐地铁，想买一个安静又方便携带的耳机",
        k=3
    )
    print_results(response)

