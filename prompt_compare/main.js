import { Client } from "@elastic/elasticsearch";
import express from "express";
import cors from "cors";
import yaml from "js-yaml";
import fs from "fs";

// 加载配置
const config = yaml.load(fs.readFileSync("config.yaml", "utf8"));
console.log("load config");
console.log(config);


const app = express();
const port = 3001;

app.use(cors());

const client = new Client({
  node: config.data_source.endpoint,
  auth: {
    username: config.data_source.auth.username,
    password: config.data_source.auth.password,
  },
});

// 指定要比较的模型配置
const modelConfigA = "fp16";
const modelConfigB = "int8";
const indexName = config.data_source.index_name;

async function fetchData() {
  try {
    const tableData = [];
    // 获取所有唯一的 prompt_id
    const promptIdsResult = await client.search({
      index: indexName,
      body: {
        size: 0,
        aggs: {
          unique_ids: {
            terms: {
              field: "id",
              size: 10000, // 增加此值以获取更多唯一的 prompt_id
            },
          },
        },
      },
    });
    const promptIds = promptIdsResult.aggregations.unique_ids.buckets.map(
      (bucket) => bucket.key,
    );

    for (const promptId of promptIds) {
      const results = await client.search({
        index: indexName,
        body: {
          query: {
            bool: {
              must: [
                { term: { id: promptId } },
                {
                  terms: {
                    model_config: [modelConfigA, modelConfigB],
                  },
                },
              ],
            },
          },
        },
      });
      const resultA = results.hits.hits.find(
        (hit) => hit._source.model_config === modelConfigA,
      );
      const resultB = results.hits.hits.find(
        (hit) => hit._source.model_config === modelConfigB,
      );
      if (resultA && resultB) {
        tableData.push({
          prompt_id: resultA._source.id,
          prompt: resultA._source.prompt_str,
          first_label: resultA._source.model_config,
          first_result: resultA._source.result[0],
          second_label: resultB._source.model_config,
          second_result: resultB._source.result[0],
        });
      }
    }
    return tableData;
  } catch (error) {
    console.error("Error fetching data from Elasticsearch:", error);
  }
}

// 目前从 elasticsearch 查询数据是比较耗时的，暂时添加一个最简陋的缓存
let cachedData = null;
let lastFetchTime = null;

app.get("/data", async (req, res) => {
  try {
    const currentTime = new Date();
    if (!cachedData || !lastFetchTime || currentTime - lastFetchTime > 60000) {
      // 如果没有缓存数据或缓存数据已过期（超过1分钟），则获取新数据
      cachedData = await fetchData();
      lastFetchTime = currentTime;
    }
    res.json(cachedData);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
