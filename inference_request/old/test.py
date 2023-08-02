import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import time
from transformers import AutoTokenizer


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')
    parser.add_argument('-beam',
                        '--beam_width',
                        type=int,
                        default=1,
                        help='beam width.')
    parser.add_argument('-topk',
                        '--topk',
                        type=int,
                        default=1,
                        required=False,
                        help='topk for sampling')
    parser.add_argument('-topp',
                        '--topp',
                        type=float,
                        default=0.0,
                        required=False,
                        help='topp for sampling')
    parser.add_argument(
                        '-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                             'communicate with inference service. Default is "http".')
    parser.add_argument('--return_log_probs',
                        action="store_true",
                        default=False,
                        required=False,
                        help='return the cumulative log probs and output log probs or not')
    parser.add_argument('-prompt', "--prompt", type=str, required=False,
                        default="Now you are a python programmer and join a programming exam. Complete the following "
                                "function in python. Please give out the most possible result. Make sure the program "
                                "can run properly. You just give me the code and comments among the code, do not give "
                                "me any other explanation.\n# quick sort\n")

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)
    if FLAGS.url is None:
        ip = "1.117.137.243"
        FLAGS.url = ip + ":8000" if FLAGS.protocol == "http" else ip + ":8001"
    tasks = []
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    print("Triton tokenizer: ", tokenizer)
    prompt = FLAGS.prompt
    for i in range(1):
        task = {}
        task['tokens'] = tokenizer.encode(prompt)
        tasks.append(task)
    print("Triton tokens: ", task['tokens'])
    batch_size = 1
    start_id = 1
    end_id = 2
    input_lens = [len(task['tokens']) for task in tasks[:batch_size]]
    max_len = max(input_lens)
    input_tokens = []
    for task in tasks[:batch_size]:
        input_tokens.append(task['tokens'] + [0] * (max_len - len(task['tokens'])))
    request_lens = [512] * batch_size
    print("Triton input_tokens: ", input_tokens)
    model_name = "fastertransformer"
    batch_size = 1
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=1,
                                        verbose=FLAGS.verbose) as client:
        input_ids = np.array(input_tokens, dtype=np.uint32)
        input_lens = np.array(input_lens, dtype=np.uint32).reshape(-1, 1)
        request_lens = np.array(request_lens, dtype=np.uint32).reshape(-1, 1)
        runtime_top_k = (FLAGS.topk * np.ones([batch_size, 1])).astype(np.uint32)
        runtime_top_p = FLAGS.topp * np.ones([batch_size, 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([batch_size, 1]).astype(np.float32)
        temperature = 1.0 * np.ones([batch_size, 1]).astype(np.float32)
        len_penalty = 1.0 * np.ones([batch_size, 1]).astype(np.float32)
        repetition_penalty = 1.0 * np.ones([batch_size, 1]).astype(np.float32)
        random_seed = 0 * np.ones([batch_size, 1]).astype(np.uint64)
        is_return_log_probs = FLAGS.return_log_probs * np.ones([batch_size, 1]).astype(np.bool)
        beam_width = (FLAGS.beam_width * np.ones([batch_size, 1])).astype(np.uint32)
        start_ids = start_id * np.ones([batch_size, 1]).astype(np.uint32)
        end_ids = end_id * np.ones([batch_size, 1]).astype(np.uint32)
        bad_words_ids = np.array([[[0], [-1]]] * batch_size, dtype=np.int32)
        stop_words_ids = np.array([[[0], [-1]]] * batch_size, dtype=np.int32)

        inputs = [
            prepare_tensor("input_ids", input_ids, FLAGS.protocol),
            prepare_tensor("input_lengths", input_lens, FLAGS.protocol),
            prepare_tensor("request_output_len", request_lens, FLAGS.protocol),
            prepare_tensor("runtime_top_k", runtime_top_k, FLAGS.protocol),
            prepare_tensor("runtime_top_p", runtime_top_p, FLAGS.protocol),
            prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, FLAGS.protocol),
            prepare_tensor("temperature", temperature, FLAGS.protocol),
            prepare_tensor("len_penalty", len_penalty, FLAGS.protocol),
            prepare_tensor("repetition_penalty", repetition_penalty, FLAGS.protocol),
            prepare_tensor("random_seed", random_seed, FLAGS.protocol),
            prepare_tensor("is_return_log_probs", is_return_log_probs, FLAGS.protocol),
            prepare_tensor("beam_width", beam_width, FLAGS.protocol),
            prepare_tensor("start_id", start_ids, FLAGS.protocol),
            prepare_tensor("end_id", end_ids, FLAGS.protocol),
            prepare_tensor("bad_words_list", bad_words_ids, FLAGS.protocol),
            prepare_tensor("stop_words_list", stop_words_ids, FLAGS.protocol),
        ]

        print("Triton start_ids: ", start_ids)
        print("Triton end_ids: ", end_ids)
        print("Triton bad_words_ids: ", bad_words_ids)
        print("Triton stop_words_ids: ", stop_words_ids)
        print("Triton prepare_tensor start_ids: ", prepare_tensor("start_id", start_ids, FLAGS.protocol))

        try:
            start = time.time()
            result = client.infer(model_name, inputs)
            print(f"done in {time.time() - start} seconds")
            output0 = result.as_numpy("output_ids")
            output1 = result.as_numpy("sequence_length")
            print("============After fastertransformer============")
            # print(output0)
            # print(output1)
            if FLAGS.return_log_probs:
                output2 = result.as_numpy("cum_log_probs")
                output3 = result.as_numpy("output_log_probs")
                print(f"output2: {output2}")
                print(f"output3: {output3}")
            print("===============================================\n")
        except Exception as e:
            print(e)

    for i, task in enumerate(tasks[:batch_size]):
        result_text = []
        for j in range(FLAGS.beam_width):
            result_len = (output0[i][j] != 0).sum()
            result_tokens = output0[i][j][:result_len]
            text = tokenizer.decode(result_tokens.tolist())
            print("=================== result ====================\n\n\n")
            print(text)
            result_text.append(text)
        task['result'] = result_text
    print('Finished!\n')
