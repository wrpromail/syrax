import numpy as np
import pickle
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def generate_model_single_input(protocol, prompt_token, temp=1.0):
    start_id = 1
    end_id = 2
    batch_size = 1
    topk = 1
    topp = 0.0
    return_log_probs = False
    beam_width = 1
    tasks = []
    for i in range(1):
        task = {'tokens': prompt_token}
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
    temperature = temp * np.ones([batch_size, 1]).astype(np.float32)
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
        prepare_tensor("input_ids", input_ids, protocol),
        prepare_tensor("input_lengths", input_lens, protocol),
        prepare_tensor("request_output_len", request_lens, protocol),
        prepare_tensor("runtime_top_k", runtime_top_k, protocol),
        prepare_tensor("runtime_top_p", runtime_top_p, protocol),
        prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, protocol),
        prepare_tensor("temperature", temperature, protocol),
        prepare_tensor("len_penalty", len_penalty, protocol),
        prepare_tensor("repetition_penalty", repetition_penalty, protocol),
        prepare_tensor("random_seed", random_seed, protocol),
        prepare_tensor("is_return_log_probs", is_return_log_probs, protocol),
        prepare_tensor("beam_width", beam_width, protocol),
        prepare_tensor("start_id", start_ids, protocol),
        prepare_tensor("end_id", end_ids, protocol),
        prepare_tensor("bad_words_list", bad_words_ids, protocol),
        prepare_tensor("stop_words_list", stop_words_ids, protocol),
    ]
    return inputs


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                 concurrency=concurrency,
                                                 verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                 verbose=verbose)


# beam_width = 1
def get_inference_result(result, tokenizer, batch_size=1, beam_width=1):
    output_ids = result.as_numpy("output_ids")
    inference_results = []
    for i in range(batch_size):
        for j in range(beam_width):
            result_len = (output_ids[i][j] != 0).sum()
            result_tokens = output_ids[i][j][:result_len]
            text = tokenizer.decode(result_tokens.tolist())
            inference_results.append(text)
    return inference_results


def dump_result(triton_result, save_path):
    result_str = pickle.dumps(triton_result)
    with open(save_path, 'wb') as f:
        f.write(result_str)


def load_result(load_path):
    with open(load_path, "rb") as f:
        result_str = f.read()
        result = pickle.loads(result_str)
        return result
