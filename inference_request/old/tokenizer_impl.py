from tokenizer import Tokenizer
from transformers import AutoTokenizer


class LocalTokenizer(Tokenizer):
    def __init__(self, tokenizer_path: str):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self._load()

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def tokenize_single(self, raw_input):
        return self.tokenizer.encode(raw_input)

    def get_tokenizer(self):
        return self.tokenizer

