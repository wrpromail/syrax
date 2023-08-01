import os


class Config:
    def __init__(self):
        self.job_seq = os.getenv("JOB_SEQ", "default")
        self.prompt_prefix = os.getenv("PROMPT_PREFIX", "")
        self.model_name = os.getenv("MODEL_NAME", "fastertransformer")
        self.prompt_suffix = os.getenv("PROMPT_SUFFIX", "")
        self.protocol = os.getenv("REQUEST_PROTOCOL", "http")
        self.endpoint = os.getenv("TARGET_ENDPOINT")
        self.mode = os.getenv("MODE", "legacy")
