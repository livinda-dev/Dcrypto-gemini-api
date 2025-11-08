import os, json, math
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ---- Gemini setup ----
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-2.5-pro")


app = FastAPI()

class CandleReq(BaseModel):
    candles: List[List[float]]  # 60 x [open, high, low, close, volume]
    pair: str
    interval: str

class RecommendReq(CandleReq):
    market_type: str  # "spot" or "futures"

@app.post("/predict")
def predict(req: CandleReq):
    prompt = f"""
You are a crypto direction classifier.
Return ONLY JSON:
{{"direction":"UP|SIDE|DOWN","confidence":0.0-1.0}}

PAIR: {req.pair}
TF: {req.interval}
CANDLES (60 rows of [O,H,L,C,V]):
{req.candles}
"""
    gem = model.generate_content(prompt)
    text = gem.text.strip()
    try:
        return json.loads(text)
    except:
        if "UP" in text: return {"direction":"UP","confidence":0.0}
        if "DOWN" in text: return {"direction":"DOWN","confidence":0.0}
        return {"direction":"SIDE","confidence":0.0}

def compute_atr14(candles: List[List[float]]) -> float:
    # candles are oldest->newest
    if len(candles) < 15: return 0.0
    trs = []
    for i in range(1, len(candles)):
        o, h, l, c, v = candles[i]
        _, _, _, prev_c, _ = candles[i-1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    # simple average of last 14 TR values
    return sum(trs[-14:]) / 14.0 if len(trs) >= 14 else (sum(trs)/len(trs) if trs else 0.0)

@app.post("/recommend")
def recommend(req: RecommendReq):
    candles = req.candles[:]  # oldest->newest
    last_close = candles[-1][3]
    atr = compute_atr14(candles)

    prompt = f"""
You are a crypto trade recommender AI. Markets: {req.market_type.upper()}.
Given 60 sequential candles [O,H,L,C,V], choose a single action:
- spot: BUY or SELL or SIDE
- futures: LONG or SHORT or SIDE

TP and SL returned as FRACTION of entry price.

RULES:
- confidence MUST be float between 0.1 and 0.9 (never 0.0, never 1.0)
- tp_pct must be 0.005..0.03 (0.5%..3%)
- sl_pct must be 0.003..0.02 (0.3%..2%)

RETURN STRICT JSON ONLY:

{{
  "direction":"UP|SIDE|DOWN",
  "action":"BUY|SELL|LONG|SHORT|SIDE",
  "confidence":0.55,
  "tp_pct":0.012,
  "sl_pct":0.009
}}

PAIR: {req.pair}
TF: {req.interval}
CANDLES:{candles}
"""
    gem = model.generate_content(prompt)
    raw = gem.text.strip()

    # defaults
    direction, action, confidence = "SIDE", "SIDE", 0.0
    tp_pct, sl_pct = 0.01, 0.008

    try:
        data = json.loads(raw)

        direction = data.get("direction", direction)
        action = data.get("action", action)

        # robust floats
        def fkey(k, default):
            v = data.get(k, default)
            return float(v) if v is not None and v != "" else default

        confidence = fkey("confidence", confidence)
        tp_pct = fkey("tp_pct", tp_pct)
        sl_pct = fkey("sl_pct", sl_pct)

    except:
        # fallback ATR approx
        if atr > 0:
            tp_pct = min(max(atr / max(last_close,1e-9),0.004),0.03)
            sl_pct = min(max(tp_pct*0.8,0.003),0.02)

    # normalize action to market type
    if req.market_type.lower()=="spot":
        if direction=="UP": action="BUY"
        if direction=="DOWN": action="SELL"
    else:
        if direction=="UP": action="LONG"
        if direction=="DOWN": action="SHORT"

    entry = last_close
    if action in ["BUY","LONG"]:
        tp = entry*(1+tp_pct)
        sl = entry*(1-sl_pct)
    elif action in ["SELL","SHORT"]:
        tp = entry*(1-tp_pct)
        sl = entry*(1+sl_pct)
    else:
        tp = entry
        sl = entry

    return {
        "direction": direction,
        "confidence": confidence,
        "action": action,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
    }

