import os
import numpy as np
import json
import re
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import time
from transformers import AutoTokenizer
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio

from prompt_source_impl import ParquetFilePromptSource
from prompt_source import TritonInputGenerator
from tokenizer_impl import LocalTokenizer
from burner import Burner


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                 concurrency=concurrency,
                                                 verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                 verbose=verbose)


def save_result(result_list, save_dir):
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    problem_solution_list = []
    for result in result_list:
        output0 = result.as_numpy("output_ids")
        result_len = (output0[0][0] != 0).sum()
        result_tokens = output0[0][0][:result_len]
        text = tokenizer.decode(result_tokens.tolist()).replace("<s>", "").replace("</s>", "")
        parts = re.split(r'(?=class Solution)', text, maxsplit=1)
        try:
            problem_solution_list.append({"problem": parts[0].strip(), "solution": parts[1].strip()})
        except Exception as e:
            print(e)
    # print(problem_solution_list)
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(problem_solution_list, f, ensure_ascii=False)
        f.close()


lock = threading.Lock()
count = 0


def send_request(client, input, concurrency, model_name):
    global count
    result = None
    try:
        result = client.infer(model_name, input)
        with lock:
            count += 1
            print("concurrency: {}, count: {}".format(concurrency, count))
    except Exception as e:
        print(e)
    return result


def run_single_thread(all_input, holder):
    # 不并发
    elapsed_times = []
    all_result = []
    num_requests = len(all_input)
    with create_inference_server_client(protocol, url, concurrency=1, verbose=False) as client:
        for i in tqdm(range(len(all_input))):
            try:
                start_time = time.time()
                result = client.infer(model_name, all_input[i])
                all_result.append(result)
                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)
            except Exception as e:
                print(e)
    # 计算吞吐量和平均计算延迟
    elapsed_times = np.array(elapsed_times)
    avg_latency = np.mean(elapsed_times) * 1000  # 转换为毫秒
    throughput = num_requests / np.sum(elapsed_times)
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} requests/s")
    return all_result


def run_multi_thread(protocol, url, all_input, model_name, concurrency=50):
    global count
    count = 0
    # 并发
    all_result = []
    with create_inference_server_client(protocol, url, concurrency=concurrency, verbose=False) as client:
        # 使用 ThreadPoolExecutor 创建一个包含 10 个线程的线程池
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            start_time = time.time()
            # 将 10 个 GET 请求任务提交给线程池
            futures = [executor.submit(send_request, client, input, concurrency) for input in all_input]
            # 等待所有任务完成
            for future in futures:
                r = future.result()
                if r is not None:
                    all_result.append(r)
            end_time = time.time()
    total_time = end_time - start_time
    avg_latency = total_time * 1000 / len(all_input)
    throughput = len(all_input) / total_time
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} requests/s")

    return all_result, avg_latency, throughput


# async def async_send_request(client, prompt, concurrency):
#     start_time = time.time()
#     response = await client.infer(model_name, prompt)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"single request sended, concurrency: {concurrency}, elapsed_time: {elapsed_time:.2f} ms")
#     return response

async def async_send_request(client, model_name, prompt, concurrency):
    print("single request invoked")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, client.infer, model_name, prompt)
    return response


async def async_run_multi_thread(all_input, concurrency=50):
    global count
    count = 0
    all_result = []
    client = create_inference_server_client(protocol, url, concurrency=concurrency, verbose=False)
    start_time = time.time()
    # 使用 asyncio.gather() 方法并发执行所有请求任务
    coroutines = [async_send_request(client, model_name, single_input, concurrency) for single_input in all_input]
    results = await asyncio.gather(*coroutines)
    print("all requests sended")
    print("results count: " + str(len(results)))
    for r in results:
        if r is not None:
            all_result.append(r)
    end_time = time.time()
    total_time = end_time - start_time
    avg_latency = total_time * 1000 / len(all_input)
    throughput = len(all_input) / total_time
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} requests/s")
    return all_result, avg_latency, throughput


async def run(is_legacy: bool = False):
    tokenizer = LocalTokenizer("./tokenizer")
    # 初始化测试 prompt 数据源
    dps = ParquetFilePromptSource(
        parquet_file_path="./source/leetcode_free_questions_text.parquet",
        json_key="question", truncate_count=20)
    dps.load_source()
    tig = TritonInputGenerator(prompt_source=dps, tokenizer=tokenizer, prompt_prefix=prompt_prefix)
    all_input = tig.generate_input_list()
    print("perform infer...")
    points = []
    for i in np.arange(1, 5, 5):
        # all_result = run_single_thread(all_input)
        print("\nmulti thread request, concurrency is {}".format(i))
        start_time = time.time()
        all_result, avg_latency, throughput = await async_run_multi_thread(all_input, i)
        end_time = time.time()
        print(f"Time spent: {end_time - start_time:.2f} s")
        # 记录坐标点
        points.append({"concurrency": int(i), "avg_latency": avg_latency, "throughput": throughput})
        # 保存到json文件中
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        # 保存结果
        save_result(all_result, "./concurrency_{}_{}_result.json".format(i, formatted_time))
    # 画图
    print(points)
    save_dir = "points_8x2_v2.json"
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(points, f, ensure_ascii=False)
        f.close()


class LegacyThreadingBurner(Burner):
    def __init__(self, protocol, target_url, model_name, request_concurrency, input_list, job_seq: str,
                 folder: str = "."):
        super().__init__()
        self.protocol = protocol
        self.target_url = target_url
        self.model_name = model_name
        self.elapsed = None
        self.request_concurrency = request_concurrency
        self.input_list = input_list
        self.all_result = None
        self.avg_latency = None
        self.throughput = None
        self.result = None
        self.job_seq = job_seq
        self.folder = folder

    def run_single_test_request(self):
        if self.input_list is None or len(self.input_list) == 0:
            print("input list is empty")
            return
        with create_inference_server_client(self.protocol, self.target_url, concurrency=1, verbose=True) as client:
            start_time = time.time()
            response = client.infer(self.model_name, self.input_list[0])
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"single request sended, concurrency: {1}, elapsed_time: {elapsed_time:.2f} ms")
            return response

    def run(self):
        start_time = time.time()
        self.run_multi_thread()
        end_time = time.time()
        self.elapsed = end_time - start_time
        print(f"Time spent: {self.elapsed:.2f} s")

    def get_result(self):
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        filename = "concurrency_{}_{}_{}_result.json".format(self.job_seq, self.request_concurrency, formatted_time)
        filepath = os.path.join(self.folder, filename)
        save_result(self.all_result, filepath)

    def run_multi_thread(self):
        global count
        count = 0
        # 并发
        self.all_result = []
        with create_inference_server_client(protocol, url, concurrency=self.request_concurrency,
                                            verbose=False) as client:
            # 使用 ThreadPoolExecutor 创建一个包含 10 个线程的线程池
            with ThreadPoolExecutor(max_workers=self.request_concurrency) as executor:
                start_time = time.time()
                # 将 10 个 GET 请求任务提交给线程池
                futures = [executor.submit(send_request, client, input, self.request_concurrency) for input in
                           self.input_list]
                # 等待所有任务完成
                for future in futures:
                    r = future.result()
                    if r is not None:
                        self.all_result.append(r)
                end_time = time.time()
        total_time = end_time - start_time
        self.avg_latency = total_time * 1000 / len(self.input_list)
        self.throughput = len(self.input_list) / total_time
        print(f"Average Latency: {self.avg_latency:.2f} ms")
        print(f"Throughput: {self.throughput:.2f} requests/s")
        self.result = {"concurrency": request_concurrency, "avg_latency": self.avg_latency,
                       "throughput": self.throughput}
        print(self.result)


def legacy_run():
    # 初始化 tokenizer
    tokenizer = LocalTokenizer("./tokenizer")
    # 初始化测试 prompt 数据源
    dps = ParquetFilePromptSource(
        parquet_file_path="./source/leetcode_free_questions_text.parquet",
        json_key="question", truncate_count=prompt_truncate)
    dps.load_source()
    tig = TritonInputGenerator(prompt_source=dps, tokenizer=tokenizer, prompt_prefix=prompt_prefix)
    all_input = tig.generate_input_list()

    print("perform infer...")

    burner = LegacyThreadingBurner(protocol, url, model_name, request_concurrency, all_input, job_seq, output_folder)
    burner.run()
    burner.get_result()

    # start_time = time.time()
    # all_result, avg_latency, throughput = run_multi_thread(all_input, request_concurrency)
    # end_time = time.time()
    # print(f"Time spent: {end_time - start_time:.2f} s")
    # # 记录坐标点
    # result = {"concurrency": request_concurrency, "avg_latency": avg_latency, "throughput": throughput}
    # print(result)
    # # 保存到json文件中
    # now = datetime.now()
    # formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # # 保存结果
    # save_result(all_result, "./concurrency_{}_{}_{}_result.json".format(job_seq, request_concurrency, formatted_time))


if __name__ == "__main__":
    job_seq = os.getenv("JOB_SEQ", "default")
    prompt_prefix = os.getenv("PROMPT_PREFIX",
                              "Now you are a python programmer and join a programming exam. Complete the following "
                              "function in python. Please give out the most possible result. Make sure the program "
                              "can run properly. You just give me the code and comments among the code,do not give me "
                              "any other explanation. The Problem is as follows: \n")
    prompt_suffix = os.getenv("PROMPT_SUFFIX")
    model_name = os.getenv("MODEL_NAME", "fastertransformer")
    protocol = os.getenv("REQUEST_PROTOCOL", "http")
    ip = os.getenv("TARGET_IP", "1.117.135.251")
    endpoint = os.getenv("TARGET_ENDPOINT")
    mode = os.getenv("MODE", "legacy")
    prompt_truncate = os.getenv("PROMPT_COUNT")
    if prompt_truncate:
        prompt_truncate = int(prompt_truncate)
    request_concurrency = os.getenv("REQUEST_CONCURRENCY", "5")
    if request_concurrency:
        request_concurrency = int(request_concurrency)
    output_folder = os.getenv("OUTPUT_FOLDER", ".")

    url = ip + ":8000" if protocol == "http" else ip + ":8001"
    if endpoint is not None:
        url = endpoint

    if mode == "legacy":
        legacy_run()
    else:
        asyncio.run(run(False))
