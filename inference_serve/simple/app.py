from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import model_infer, model_eval
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()


class InferRequest(BaseModel):
    prompt: str


@app.get("/eval")
def model_eval_get():
    return model_eval()


@app.get("/infer")
def model_infer_get(request: str):
    logger.info('Received GET request: %s', request)
    try:
        result = model_infer(request)
        logger.info('Inference result: %s', result)
        return result
    except Exception as e:
        logger.error('Error occurred: %s', str(e))
        return str(e)


@app.post("/infer", status_code=200)
def model_infer_post(data: InferRequest):
    logger.info('Received POST request: %s', data)
    try:
        result = model_infer(data.prompt)
        logger.info('Inference result: %s', result)
        return model_infer(data.prompt)
    except Exception as e:
        logger.error('Error occurred: %s', str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ping")
def system_ping():
    return "pong"
