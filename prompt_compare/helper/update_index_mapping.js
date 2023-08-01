import { Client } from "@elastic/elasticsearch";

const client = new Client({
  node: "http://es-cn-x0r3ba8hc0002lx4v.public.elasticsearch.aliyuncs.com:9200",
  auth: {
    username: "elastic",
    password: "devopsCS123",
  },
});

const indexName = "llama_test";

async function updateMapping() {
  try {
    await client.indices.putMapping({
      index: indexName,
      body: {
        properties: {
          id: {
            type: "keyword",
          },
          // ...其他字段映射（如果需要）
        },
      },
    });
    console.log("Mapping updated successfully.");
  } catch (error) {
    console.error("Error updating mapping:", error);
  }
}

updateMapping();
