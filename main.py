import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-2.0-flash")

app = FastAPI()

class CandleReq(BaseModel):
    candles: list  # 60x5 list
    pair: str
    interval: str

@app.post("/predict")
def predict(req: CandleReq):
    prompt = f"""
    You are crypto direction model.

    Input pair: {req.pair}
    interval: {req.interval}
    candle normalized values (60 rows): {req.candles}

    Return JSON: {{"direction":"UP/SIDE/DOWN","confidence":0.xx}}
    """
    r = model.generate_content(prompt)
    return r.text
