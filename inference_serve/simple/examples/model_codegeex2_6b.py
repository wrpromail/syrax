from transformers import AutoTokenizer, AutoModel
import os

tokenizer_path = os.environ.get("TOKENIZER_PATH", "THUDM/codegeex2-6b")
model_path = os.environ.get("MODEL_PATH", "THUDM/codegeex2-6b")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cuda')

# remember adding a language tag for better performance
sample_prompt = "# language: Python\n# write a bubble sort function\n"
sample_prompt1 = "# language: Python\n# write some code about python fastapi library, such like how to provider post http interface\n"


def model_infer(request:str) -> str:
    inputs = tokenizer.encode(request, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=256, top_k=1)
    return tokenizer.decode(outputs[0])

def model_eval() -> str:
    return model.eval()