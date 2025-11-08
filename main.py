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

    Return JSON EXACTLY like:
    {{"direction":"UP","confidence":0.73}}
    """
    r = model.generate_content(prompt)
    text = r.text.strip()

    # ensure valid JSON
    import json
    try:
        data = json.loads(text)
        return data
    except:
        # if model returns text string not valid json
        if "UP" in text: return {"direction":"UP","confidence":0.0}
        if "DOWN" in text: return {"direction":"DOWN","confidence":0.0}
        return {"direction":"SIDE","confidence":0.0}

