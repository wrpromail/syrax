import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from tokenizer import Tokenizer


# 读取压测数据来源
class PromptSource:
    def __init__(self, truncate_count: int = 0):
        self.truncate_count = truncate_count

    def load_source(self):
        pass

    def get_prompt_iterator(self) -> iter:
        pass

    def get_single_prompt(self) -> str:
        pass

    def get_all_prompts(self) -> list:
        pass


class InputGenerator:
    def __init__(self, prompt_source: PromptSource, tokenizer: Tokenizer):
        self.prompt_source = prompt_source
        self.tokenizer = tokenizer


class TritonInputGenerator(InputGenerator):
    def __init__(self, prompt_source: PromptSource, tokenizer: Tokenizer, prompt_prefix=None, prompt_suffix=None):
        super().__init__(prompt_source, tokenizer)
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.protocol = "http"

    @staticmethod
    def prepare_tensor(name, input, protocol):
        client_util = httpclient if protocol == "http" else grpcclient
        t = client_util.InferInput(
            name, input.shape, np_to_triton_dtype(input.dtype))
        t.set_data_from_numpy(input)
        return t

    def get_prompt(self, raw):
        output = raw
        if self.prompt_prefix:
            output = self.prompt_prefix + output
        if self.prompt_suffix:
            output = output + self.prompt_suffix
        return output

    def generate_input_list(self):
        protocol = self.protocol

        raw = self.prompt_source.get_all_prompts()
        if not raw:
            raise Exception

        all_inputs = []
        start_id = 1
        end_id = 2
        batch_size = 1
        topk = 1
        topp = 0.0
        return_log_probs = False
        beam_width = 1

        for prompt in raw:
            tasks = []
            for i in range(1):
                task = {'tokens': self.tokenizer.tokenize_single(self.get_prompt(prompt))}
                # task = {'tokens': tokenizer.encode(base_prompt + prompt)}
                tasks.append(task)

            input_lens = [len(task['tokens']) for task in tasks[:batch_size]]
            max_len = max(input_lens)
            input_tokens = []
            for task in tasks[:batch_size]:
                input_tokens.append(task['tokens'] + [0] * (max_len - len(task['tokens'])))
            request_lens = [512] * batch_size

            input_ids = np.array(input_tokens, dtype=np.uint32)
            input_lens = np.array(input_lens, dtype=np.uint32).reshape(-1, 1)
            request_lens = np.array(request_lens, dtype=np.uint32).reshape(-1, 1)
            runtime_top_k = (topk * np.ones([batch_size, 1])).astype(np.uint32)
            runtime_top_p = topp * np.ones([batch_size, 1]).astype(np.float32)
            beam_search_diversity_rate = 0.0 * np.ones([batch_size, 1]).astype(np.float32)
            temperature = 1.0 * np.ones([batch_size, 1]).astype(np.float32)
            len_penalty = 1.0 * np.ones([batch_size, 1]).astype(np.float32)
            repetition_penalty = 1.0 * np.ones([batch_size, 1]).astype(np.float32)
            random_seed = 0 * np.ones([batch_size, 1]).astype(np.uint64)
            is_return_log_probs = return_log_probs * np.ones([batch_size, 1]).astype(np.bool_)
            beam_width = (beam_width * np.ones([batch_size, 1])).astype(np.uint32)
            start_ids = start_id * np.ones([batch_size, 1]).astype(np.uint32)
            end_ids = end_id * np.ones([batch_size, 1]).astype(np.uint32)
            bad_words_ids = np.array([[[0], [-1]]] * batch_size, dtype=np.int32)
            stop_words_ids = np.array([[[0], [-1]]] * batch_size, dtype=np.int32)

            inputs = [
                self.prepare_tensor("input_ids", input_ids, protocol),
                self.prepare_tensor("input_lengths", input_lens, protocol),
                self.prepare_tensor("request_output_len", request_lens, protocol),
                self.prepare_tensor("runtime_top_k", runtime_top_k, protocol),
                self.prepare_tensor("runtime_top_p", runtime_top_p, protocol),
                self.prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, protocol),
                self.prepare_tensor("temperature", temperature, protocol),
                self.prepare_tensor("len_penalty", len_penalty, protocol),
                self.prepare_tensor("repetition_penalty", repetition_penalty, protocol),
                self.prepare_tensor("random_seed", random_seed, protocol),
                self.prepare_tensor("is_return_log_probs", is_return_log_probs, protocol),
                self.prepare_tensor("beam_width", beam_width, protocol),
                self.prepare_tensor("start_id", start_ids, protocol),
                self.prepare_tensor("end_id", end_ids, protocol),
                self.prepare_tensor("bad_words_list", bad_words_ids, protocol),
                self.prepare_tensor("stop_words_list", stop_words_ids, protocol),
            ]

            all_inputs.append(inputs)
        return all_inputs


class Burner:
    def __init__(self):
        pass


class Output:
    def __init__(self, output_type: str):
        pass
