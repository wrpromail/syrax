import torch, transformers
import os
from transformers import AutoTokenizer

model_path = os.environ.get("MODEL_PATH", "mosaicml/mpt-7b")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "EleutherAI/gpt-neox-20b")


tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096
config.init_device = 'cuda:0'
model = transformers.AutoModelForCausalLM.from_pretrained(model_path,config=config,trust_remote_code=True).half().cuda()


def model_infer(request: str) -> str:
    raise NotImplementedError("TODO: Implement me!")

def model_eval() -> str:
    return model.eval()