import datasets
import json
import re
import argparse
import numpy as np
import json
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import time
from transformers import AutoTokenizer


def convert_original_to_json():
    # ds_dir = '/Users/xjx/huggingface/datasets/BoyuanJackchen/leetcode_free_questions_text/train'
    # ds_dir = '/Users/xjx/huggingface/datasets/Saauan/leetcode-performance/train'
    ds_dir = '/Users/xjx/huggingface/datasets/mhhmm/leetcode-solutions-python/train'
    ds = datasets.load_from_disk(ds_dir)

    prompt_list = []
    for data in ds:
        pattern = r"(.*?)```python"
        match = re.search(pattern, data['code_with_problem'], re.DOTALL)

        if match:
            question_description = match.group(1).strip()
            prompt_list.append({"prompt": question_description, "length": len(question_description)})
            # print(question_description)
        else:
            print("未找到匹配项")

    with open("./input_mhhmm_leetcode-solutions-python.json", 'w', encoding='utf-8') as f:
        json.dump(prompt_list, f, ensure_ascii=False)
        f.close()


def combine_json():
    all_files = ["input_BoyuanJackchen_leetcode_free_questions_text.json", "input_mhhmm_leetcode-solutions-python.json",
                 "input_Saauan_leetcode-performance.json"]
    all_input = []
    for file in all_files:
        with open(file, 'r', encoding='utf-8') as f:
            input = json.load(f)
            all_input += input
    print(len(all_input))
    with open("./input_raw.json", 'w', encoding='utf-8') as f:
        json.dump(all_input, f, ensure_ascii=False)
        f.close()


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                 concurrency=concurrency,
                                                 verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                 verbose=verbose)


def convert(prompt):
    tasks = []
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    base_prompt = "Now you are a python programmer and join a programming exam. Complete the following function in python. " \
                  "Please give out the most possible result. Make sure the program can run properly. You just give me the " \
                  "code and comments among the code, do not give me any other explanation. The Problem is as follows: \n"
    for i in range(1):
        task = {'tokens': tokenizer.encode(base_prompt + prompt)}
        tasks.append(task)
    # print("Triton tokens: ", task['tokens'])
    batch_size = 1

    input_lens = [len(task['tokens']) for task in tasks[:batch_size]]
    max_len = max(input_lens)
    input_tokens = []
    for task in tasks[:batch_size]:
        input_tokens.append(task['tokens'] + [0] * (max_len - len(task['tokens'])))
    request_lens = [512] * batch_size
    print("Triton input_tokens: ", input_tokens)

    input_ids = np.atleast_1d(np.squeeze(np.array(input_tokens, dtype=np.uint32)))
    input_lens = np.atleast_1d(np.squeeze(np.array(input_lens, dtype=np.uint32).reshape(-1, 1)))
    request_lens = np.atleast_1d(np.squeeze(np.array(request_lens, dtype=np.uint32).reshape(-1, 1)))
    input_json = {"input_ids": {"content": input_ids.flatten().tolist(), "shape": input_ids.shape},
                  "input_lengths": {"content": input_lens.flatten().tolist(), "shape": input_lens.shape},
                  "request_output_len": {"content": request_lens.flatten().tolist(), "shape": request_lens.shape}}
    return input_json


def convert_json_to_tokens():
    all_input = []
    with open('./input_raw.json', 'r', encoding='utf-8') as f:
        inputs = json.load(f)[:10]
        cnt = 0
        for input in inputs:
            all_input.append(convert(input['prompt']))
            cnt += 1
    output = {"data": all_input}
    with open("input_final.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)
        f.close()


if __name__ == "__main__":
    # convert_original_to_json()
    # combine_json()
    convert_json_to_tokens()
