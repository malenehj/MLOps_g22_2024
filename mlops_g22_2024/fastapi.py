from fastapi import FastAPI
from http import HTTPStatus
from mlops_g22_2024.predict_model import predict 

app = FastAPI()


@app.get("/text_model/{text}")
def is_emotion(text: str):
    
    response = {
        "input": text,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "emotion": predict(text)
    }
    return response

