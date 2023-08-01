from fastapi import FastAPI


app = FastAPI()

def import_model():
    try:
        from model import infer, eval
    except ImportError:
        print("ImportError: model.py not found")
    return infer, eval

infer_func, eval_func = import_model()

@app.get("/eval")
def model_eval():
    return eval_func()

@app.get("/infer")
def model_infer(request: str):
    try:
        return infer_func(request)
    except Exception as e:
        return str(e)


