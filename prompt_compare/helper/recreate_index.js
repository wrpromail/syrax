import { Client } from "@elastic/elasticsearch";

const client = new Client({
  node: "http://es-cn-x0r3ba8hc0002lx4v.public.elasticsearch.aliyuncs.com:9200",
  auth: {
    username: "elastic",
    password: "devopsCS123",
  },
});

const indexName = "llama_test";

async function recreateIndex() {
  try {
    // 删除现有索引（注意：这将删除所有数据，请确保您已备份数据）
    await client.indices.delete({ index: indexName });

    // 使用更新后的映射创建新索引
    await client.indices.create({
      index: indexName,
      body: {
        mappings: {
          properties: {
            id: {
              type: "keyword",
            },
            prompt_str: {
              type: "text",
            },
            model_config: {
              type: "keyword",
            },
            result: {
              type: "text",
            },
          },
        },
      },
    });

    console.log("Index recreated successfully.");
  } catch (error) {
    console.error("Error recreating index:", error);
  }
}

recreateIndex();
