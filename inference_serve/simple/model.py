import os
from transformers import AutoTokenizer, AutoModel

model_path = os.environ.get("MODEL_PATH", "THUDM/chatglm2-6b")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

def infer(request: str) -> str:
    response, history = model.chat(tokenizer, request)
    return response

def eval() -> str:
    return model.eval()