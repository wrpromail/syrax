import json
import elasticsearch

es = elasticsearch.Elasticsearch(
    hosts=["http://es-cn-x0r3ba8hc0002lx4v.public.elasticsearch.aliyuncs.com:9200"],
    basic_auth=("elastic", "devopsCS123")
)


def process_and_upload_jsonl(file_path, model_config, index_name):
    with open(file_path, 'r') as file:
        for line in file:
            # 从 jsonl 文件中读取一行并解析为 JSON 对象
            data = json.loads(line)

            # 添加 model_version 字段
            data['model_config'] = model_config

            # 将数据上传到 Elasticsearch
            es.index(index=index_name, document=data)
            print("upload one")


if __name__ == "__main__":
    #print(es.ping())
    process_and_upload_jsonl("sample8i.jsonl", "int8", "llama_test")
