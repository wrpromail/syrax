from diffusers import StableDiffusionPipeline
import torch
import os

model_path = os.environ.get("MODEL_PATH", "runwayml/stable-diffusion-v1-5")

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def model_infer(request: str) -> str:
    image = pipe(request).images[0]
    # 将图片转换为 base64 编码的字符串
    return image.to_base64()

def model_eval() -> str:
    try:
        return pipe.eval()
    except:
        return "not transformer model"