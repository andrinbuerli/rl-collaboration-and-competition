
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

RUN pip install unityagents==0.4.0 --no-deps
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

RUN ipython kernel install --name python3 --user

COPY . /app