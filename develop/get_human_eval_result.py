from human_eval.data import write_jsonl, read_problems
import requests

problems = read_problems()


def generate_one_completion(prompt):
    url = 'http://10.0.0.2:8000/infer'
    headers = {'Content-Type': 'application/json'}
    data = {'prompt': '# language: Python\n#{}\n'.format(prompt)}
    response = requests.post(url, headers=headers, json=data)
    return response.text


num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)
