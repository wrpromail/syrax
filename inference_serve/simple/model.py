import os
from transformers import AutoTokenizer, AutoModel

tokenizer_path = os.environ.get("TOKENIZER_PATH", "THUDM/chatglm2-6b")
model_path = os.environ.get("MODEL_PATH", "THUDM/chatglm2-6b")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()


def model_infer(request: str) -> str:
    response, history = model.chat(tokenizer, request)
    return response


def model_eval() -> str:
    return model.eval()
