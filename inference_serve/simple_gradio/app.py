import gradio as gr
from model import model_infer, model_eval
from impl import InferRequesstMetadata, InferResponseMetadata


def infer(prompt, metadata):
    metadata = InferRequesstMetadata(from_str=metadata)
    response = model_infer(prompt, metadata)
    return response, {"input_token":0}


def geval():
    return model_eval()


if __name__ == "__main__":
    with gr.Blocks() as app:
        with gr.Tab("infer"):
            infer_gradio_inputs = [
                gr.Textbox(label="prompt", ),
                gr.Code(label="metadata", value='{"temperature":0,"top_k":1.0,"top_p":0.0}'),
            ]
            infer_gradio_outputs = [
                gr.Textbox(label="response"),
                gr.JSON(label="metadata"),
            ]
            infer_button = gr.Button("infer")
        with gr.Tab("eval"):
            eval_gradio_outputs = [
                gr.Textbox("eval"),
            ]
            eval_button = gr.Button("eval")

        infer_button.click(fn=infer, inputs=infer_gradio_inputs, outputs=infer_gradio_outputs)
        eval_button.click(fn=geval, outputs=eval_gradio_outputs)
    app.launch(server_name="0.0.0.0")
