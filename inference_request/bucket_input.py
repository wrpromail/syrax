from tokenizer_impl import LocalTokenizer
from prompt_source_impl import ParquetFilePromptSource
from collections import defaultdict
import json
import uuid
import random


class LineContent:
    def __init__(self, token_range=None, token_count=None, token_step=None, prompt_str=None):
        self.token_range = token_range
        self.token_count = token_count
        self.token_step = token_step
        self.prompt_str = prompt_str

        self.data = {"id": uuid.uuid4().__str__(), "token_range": self.token_range, "token_count": self.token_count,
                     "token_step": self.token_step, "prompt_str": self.prompt_str}

    def get_json_str(self):
        return json.dumps(self.data)

    def unmarshal(self, json_str):
        self.data = json.loads(json_str)


#
if __name__ == "__main__":
    length_distribution = defaultdict(list)
    range_size = 100
    num_samples = 10

    tokenizer = LocalTokenizer("./tokenizer")

    dps = ParquetFilePromptSource(
        parquet_file_path="source/leetcode_free_questions_text.parquet",
        json_key="question", truncate_count=0)
    dps.load_source()

    for prompt in dps.get_all_prompts():
        prompt_token = tokenizer.tokenize_single(prompt)
        length_range = len(prompt_token) // range_size
        length_distribution[length_range].append(prompt)

    with open("divided_prompt.jsonl", "wb") as f:
        for length_range, prompts in length_distribution.items():
            random.shuffle(prompts)
            if len(prompts) >= 10:
                for p in prompts[:10]:
                    lc = LineContent(length_range, len(tokenizer.tokenize_single(p)), 100, p)
                    f.write(lc.get_json_str().encode("utf-8"))
                    f.write("\n".encode("utf-8"))
