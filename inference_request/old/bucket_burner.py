import datetime
import os
import time

from tokenizer_impl import LocalTokenizer
from tokenizer import Tokenizer
from bucket_input import LineContent
from utils import generate_model_single_input, create_inference_server_client
from utils import get_inference_result


class BucketBurner:
    def __init__(self, ep, tz: Tokenizer, input_jsonl, repeat=1):
        self.tz = tz
        self.input_jsonl = input_jsonl
        self.repeat = repeat
        self.ep = ep

    def direct_run_once(self, tp=0.0):
        result_list = []
        with open(self.input_jsonl, "rb") as f:
            with create_inference_server_client("http", self.ep, 1, False) as client:
                for line in f.readlines():
                    obj = LineContent()
                    obj.unmarshal(line.decode("utf-8"))
                    obj_input = generate_model_single_input("http", self.tz.tokenize_single(obj.data.get("prompt_str")),
                                                            tp)
                    start = time.time()
                    result = client.infer(model_name, obj_input)
                    spent = time.time() - start
                    obj.data["spent"] = int(spent * 1000)
                    if tokenizer is not None:
                        obj.data['result'] = get_inference_result(result, self.tz.get_tokenizer())
                    result_list.append(obj)
        return result_list

    @staticmethod
    def result_output(output_list, o_folder, tag):
        fn = os.path.join(o_folder, tag + ".{}.jsonl".format(str(datetime.datetime.now().timestamp())))
        with open(fn, "wb") as f:
            for obj in output_list:
                content = obj.get_json_str().encode("utf-8")
                f.write(content)
                f.write("\n".encode("utf-8"))
        print("Output to file: " + fn)


if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME", "fastertransformer")
    ip = os.getenv("TARGET_IP", "42.192.252.180")
    protocol = os.getenv("REQUEST_PROTOCOL", "http")
    if protocol is None or protocol == "":
        protocol = "http"
    endpoint = os.getenv("TARGET_ENDPOINT")
    url = ip + ":8000" if protocol == "http" else ip + ":8001"
    if endpoint is not None:
        url = endpoint

    output_folder = os.getenv("OUTPUT_FOLDER", ".")
    output_tag = os.getenv("OUTPUT_TAG", "default")
    repeat = int(os.getenv("REPEAT", "1"))
    temperature = float(os.getenv("TEMPERATURE", "0.0"))

    tokenizer = LocalTokenizer("./tokenizer")
    bb = BucketBurner(url, tokenizer, "source/divided_prompt.jsonl")
    for i in range(repeat):
        rst = bb.direct_run_once(temperature)
        bb.result_output(rst, output_folder, ip.replace(output_folder, "", -1) + "." + output_tag)
