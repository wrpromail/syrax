import os
from impl import InferRequesstMetadata, InferResponseMetadata


def model_infer(request: str, request_meta: InferRequesstMetadata):
    return request, None


def model_eval() -> str:
    return "mock eval"
