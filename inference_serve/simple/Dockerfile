FROM python:3.10-slim 

WORKDIR /app 
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
COPY app.py .

