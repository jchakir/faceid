FROM tensorflow/tensorflow:latest

ARG MODEL_NAME

WORKDIR /home/$MODEL_NAME

COPY . .

RUN python3 -m pip install -r requirements.txt

CMD [ "python3", "./faceid.py"]
