import os, json, math
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ---- Gemini setup ----
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-2.5-flash")

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
    """
    Returns:
    {
      "direction": "UP|SIDE|DOWN",
      "confidence": 0..1,
      "action": "BUY|SELL|LONG|SHORT|SIDE",
      "entry": 0.0,
      "tp": 0.0,
      "sl": 0.0,
      "tp_pct": 0.0,
      "sl_pct": 0.0
    }
    """
    candles = req.candles[:]  # oldest->newest
    last_close = candles[-1][3]
    atr = compute_atr14(candles)

    # Ask Gemini for side + risk percentage bands
    prompt = f"""
You are a crypto trade recommender. Markets: {req.market_type.upper()}.
Given 60 sequential candles [O,H,L,C,V], choose a single action:
- If market_type==spot: BUY or SELL or SIDE
- If market_type==futures: LONG or SHORT or SIDE

Also output TP and SL as *fractions* of entry price (e.g., 0.01 = 1%).
Pick reasonable, conservative values (tp_pct 0.5%..3%, sl_pct 0.3%..2%) based on trend and volatility.
OUTPUT STRICT JSON ONLY:
{{
  "direction":"UP|SIDE|DOWN",
  "action":"BUY|SELL|LONG|SHORT|SIDE",
  "confidence":0.0-1.0,
  "tp_pct":0.005,
  "sl_pct":0.004
}}

PAIR: {req.pair}
TF: {req.interval}
CANDLES(60): {candles}
"""
    gem = model.generate_content(prompt)
    raw = gem.text.strip()

    # Defaults (fallback)
    direction, action, confidence = "SIDE", "SIDE", 0.0
    tp_pct, sl_pct = 0.01, 0.008  # 1% TP, 0.8% SL defaults

    # Try parse model JSON
    try:
        data = json.loads(raw)
        direction = data.get("direction", direction)
        action = data.get("action", action)
        confidence = float(data.get("confidence", confidence))
        tp_pct = float(data.get("tp_pct", tp_pct))
        sl_pct = float(data.get("sl_pct", sl_pct))
    except:
        # fallback: derive crude % from ATR if possible
        if atr > 0:
            tp_pct = min(max(atr / max(last_close, 1e-9), 0.004), 0.03)
            sl_pct = min(max(tp_pct * 0.8, 0.003), 0.02)

    # Normalize action for market type
    if req.market_type.lower() == "spot":
        if direction == "UP" and action not in ["BUY", "SIDE"]:
            action = "BUY"
        if direction == "DOWN" and action not in ["SELL", "SIDE"]:
            action = "SELL"
    else:
        if direction == "UP" and action not in ["LONG", "SIDE"]:
            action = "LONG"
        if direction == "DOWN" and action not in ["SHORT", "SIDE"]:
            action = "SHORT"

    entry = last_close
    if action in ["BUY", "LONG"]:
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
    elif action in ["SELL", "SHORT"]:
        tp = entry * (1 - tp_pct)
        sl = entry * (1 + sl_pct)
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
