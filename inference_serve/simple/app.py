from fastapi import FastAPI


app = FastAPI()


from model import infer_func, eval_func
print("finish model loading")

@app.get("/eval")
def model_eval():
    return eval_func()

@app.get("/infer")
def model_infer(request: str):
    try:
        return infer_func(request)
    except Exception as e:
        return str(e)


@app.get("/ping")
def system_ping():
    return "pong"


