from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import model_infer, model_eval

app = FastAPI()


class InferRequest(BaseModel):
    prompt: str


@app.get("/eval")
def model_eval_get():
    return model_eval()


@app.get("/infer")
def model_infer_get(request: str):
    try:
        return model_infer(request)
    except Exception as e:
        return str(e)


@app.post("/infer", status_code=200)
def model_infer_post(data: InferRequest):
    try:
        return model_infer(data.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ping")
def system_ping():
    return "pong"
