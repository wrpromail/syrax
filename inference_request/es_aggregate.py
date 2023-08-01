from elasticsearch import Elasticsearch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

es = Elasticsearch(
    hosts=["http://es-cn-x0r3ba8hc0002lx4v.public.elasticsearch.aliyuncs.com:9200"],
    basic_auth=("elastic", "devopsCS123")
)

query = {
  "size": 0,
  "aggs": {
    "group_by_request_id": {
      "terms": {
        "field": "metadata.id.keyword",
        "size": 1000
      },
      "aggs": {
        "group_by_accuracy": {
          "terms": {
            "field": "metadata.accuracy.keyword",
            "size": 3
          },
          "aggs": {
            "vectors": {
              "top_hits": {
                "size": 1,
                "_source": ["vector"]
              }
            }
          }
        }
      }
    }
  }
}
#
response = es.search(index="trition-test", body=query)
#
#
# # 提取推理结果
buckets = response["aggregations"]["group_by_request_id"]["buckets"]
similarity_scores = []
lowest_similarities = []
pairs = []


for bucket in buckets:
    request_id = bucket["key"]
    results = bucket["group_by_accuracy"]["buckets"]
    if len(results) == 2:
        request_id = bucket["key"]
        vector1 = results[0]["vectors"]["hits"]["hits"][0]["_source"]["vector"]
        vector2 = results[1]["vectors"]["hits"]["hits"][0]["_source"]["vector"]
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)

        # 计算余弦相似度
        similarity = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
        similarity_scores.append(similarity)
        pairs.append((request_id, similarity))

avg = np.mean(similarity_scores)
med = np.median(similarity_scores)
for pair in pairs:
    if pair[1] < med or pair[1] < avg:
        print(pair)

print(f"平均相似度: {avg}")
print(f"中位数相似度: {med}")

        # 如果相似度低于阈值，将其添加到 lowest_similarities 列表中
        #similarity_threshold = 0.5  # 根据需要调整阈值
        #if similarity < similarity_threshold:
        #    lowest_similarities.append((request_id, similarity))

# 计算平均相似度
# average_similarity = sum(similarity_scores) / len(similarity_scores)
# print(f"平均相似度: {average_similarity}")

# 获取相似度最低的记录
# lowest_similarities.sort(key=lambda x: x[1])
# print("相似度最低的记录：")
# for request_id, similarity in lowest_similarities:
#     print(f"ID: {request_id}, 相似度: {similarity}")
