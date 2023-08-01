import os
from prompt_source_impl import ParquetFilePromptSource
from prompt_source import TritonInputGenerator
from tokenizer_impl import LocalTokenizer
from perf_analyzer import LegacyThreadingBurner
from utils import dump_result, load_result
from utils import get_inference_result

if __name__ == "__main__":
    job_seq = os.getenv("JOB_SEQ", "default")
    prompt_prefix = os.getenv("PROMPT_PREFIX",
                              "Now you are a python programmer and join a programming exam. Complete the following "
                              "function in python. Please give out the most possible result. Make sure the program "
                              "can run properly. You just give me the code and comments among the code,do not give me "
                              "any other explanation. The Problem is as follows: \n")
    model_name = os.getenv("MODEL_NAME", "fastertransformer")
    protocol = os.getenv("REQUEST_PROTOCOL", "http")
    if protocol is None or protocol == "":
        protocol = "http"
    ip = os.getenv("TARGET_IP", "42.192.252.180")
    endpoint = os.getenv("TARGET_ENDPOINT")
    mode = os.getenv("MODE", "legacy")
    url = ip + ":8000" if protocol == "http" else ip + ":8001"
    if endpoint is not None:
        url = endpoint

    # 初始化 tokenizer
    tokenizer = LocalTokenizer("./tokenizer")
    # 初始化测试 prompt 数据源
    dps = ParquetFilePromptSource(
        parquet_file_path="./source/leetcode_free_questions_text.parquet",
        json_key="question", truncate_count=200)
    dps.load_source()
    tig = TritonInputGenerator(prompt_source=dps, tokenizer=tokenizer, prompt_prefix=prompt_prefix)
    all_input = tig.generate_input_list()

    burner = LegacyThreadingBurner(protocol, url, model_name, 1, all_input, job_seq, ".")
    resp = burner.run_single_test_request()

    # dump_result(resp, "test.pkl")
    # cache = load_result("test.pkl")
    # print(cache)
    print(get_inference_result(resp, tokenizer.get_tokenizer()))

