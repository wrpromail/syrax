#!/usr/bin/env bash

python3 -m pip install fastapi uvicorn idna click padantic

wget -O app.py https://raw.githubusercontent.com/wrpromail/syrax/develop/inference_serve/simple/app.py

wget -O model.py https://raw.githubusercontent.com/wrpromail/syrax/develop/inference_serve/simple/examples/model_codegeex2_6b.py

uvicorn app:app --host 0.0.0.0 --port 8000
