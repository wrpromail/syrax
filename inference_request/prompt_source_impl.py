import os.path
from prompt_source import PromptSource
import datasets
import pandas as pd


class DatasetsPromptSource(PromptSource):
    def __init__(self, local_path: str, online_path: str, json_key: str):
        super().__init__()
        self.local_path = local_path
        self.online_path = online_path
        self.ds = None
        self.json_key = json_key

    def _load_local(self):
        self.ds = datasets.load_from_disk(self.local_path)

    def _load_online(self):
        pass

    def load_source(self):
        if self.local_path != "":
            self._load_local()
        else:
            self._load_online()

    def get_prompt_iterator(self) -> iter:
        pass

    def get_single_prompt(self) -> str:
        pass

    def get_all_prompts(self) -> list:
        result = []
        if not self.ds:
            return result
        for data in self.ds:
            result.append(data[self.json_key])
        return result


class MeleysDatasetPromptSource(PromptSource):
    def __init__(self, meleys_server_endpoint: str):
        super().__init__()
        self.meleys_server_endpoint = meleys_server_endpoint

    def load_source(self):
        pass

    def get_prompt_iterator(self) -> iter:
        pass

    def get_single_prompt(self) -> str:
        pass

    def get_all_prompts(self) -> list:
        result = []
        return result


class ParquetFilePromptSource(PromptSource):
    def __init__(self, parquet_file_path: str, json_key: str, truncate_count=0):
        super().__init__(truncate_count)
        self.parquet_file_path = parquet_file_path
        self.json_key = json_key
        self.ds = None

    def load_source(self):
        if os.path.splitext(self.parquet_file_path)[1] != ".parquet":
            raise Exception("File type not supported")
        self.ds = pd.read_parquet(self.parquet_file_path)

    def get_prompt_iterator(self) -> iter:
        for index, row in self.ds.iterrows():
            if self.json_key in row:
                yield row[self.json_key]

    def get_single_prompt(self) -> str:
        pass

    def get_all_prompts(self) -> list:
        result = []
        succ_count = 0
        error_count = 0
        for index, row in self.ds.iterrows():
            if self.truncate_count and 0 < self.truncate_count <= succ_count:
                break
            if self.json_key in row:
                succ_count += 1
                result.append(row[self.json_key])
            else:
                error_count += 1
            if index < 5 and error_count > 0:
                raise Exception("too much error")
        return result


if __name__ == "__main__":
    dps = ParquetFilePromptSource(
        parquet_file_path="/Users/wangrui/leetcode_free_questions_text/leetcode_free_questions_text.parquet",
        json_key="question", truncate_count=0)
    dps.load_source()
    for prompt in dps.get_all_prompts():
        print(prompt)

