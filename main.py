import os
import json
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

# 1) configure key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# 2) choose correct model id
model = genai.GenerativeModel("models/gemini-2.5-flash")   # <= this one exists!

app = FastAPI()

class CandleReq(BaseModel):
    candles: list  # 60x5 list
    pair: str
    interval: str

@app.post("/predict")
def predict(req: CandleReq):
    prompt = f"""
    You are a crypto direction predictor.

    Input pair: {req.pair}
    interval: {req.interval}

    Here are normalized candles (60 rows: open,high,low,close,volume):
    {req.candles}

    You must reply ONLY valid compact JSON:
    {{
       "direction": "<UP|SIDE|DOWN>",
       "confidence": <number 0 to 1>
    }}

    RULES:
    - direction must be exactly UP, SIDE, or DOWN
    - confidence must be a float between 0 and 1
    - NO explanation text
    - NO markdown
    - ONLY the pure JSON object
    """

    gem = model.generate_content(prompt)
    text = gem.text.strip()

    # attempts to parse json
    try:
        data = json.loads(text)
        return data
    except:
        # fallback if bad format
        if "UP" in text:
            return {"direction": "UP", "confidence": 0.0}
        if "DOWN" in text:
            return {"direction": "DOWN", "confidence": 0.0}
        return {"direction": "SIDE", "confidence": 0.0}
