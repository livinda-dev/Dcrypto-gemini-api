import os, json, math
from typing import List, Literal, Dict, Any

import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Gemini setup
# -----------------------------
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# You can switch to flash if you want cheaper calls:
# model = genai.GenerativeModel("models/gemini-2.5-flash")
model = genai.GenerativeModel("models/gemini-2.5-pro")

app = FastAPI()


# -----------------------------
# Schemas
# -----------------------------
class CandleReq(BaseModel):
    candles: List[List[float]]  # 60 x [open, high, low, close, volume] oldest->newest
    pair: str
    interval: str

class RecommendReq(CandleReq):
    market_type: Literal["spot", "futures"]


# -----------------------------
# Tech utils (Option C: trend + vol)
# -----------------------------
def ema(series: List[float], n: int) -> List[float]:
    if not series:
        return []
    a = 2.0 / (n + 1.0)
    out = [series[0]]
    for x in series[1:]:
        out.append(a * x + (1 - a) * out[-1])
    return out

def compute_atr14(candles: List[List[float]]) -> float:
    # candles: [O,H,L,C,V] oldest->newest
    if len(candles) < 2:
        return 0.0
    trs = []
    prev_c = candles[0][3]
    for i in range(1, len(candles)):
        _, h, l, c, _ = candles[i]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
        prev_c = c
    if not trs:
        return 0.0
    if len(trs) >= 14:
        return sum(trs[-14:]) / 14.0
    return sum(trs) / len(trs)

def macd_hist(close: List[float]) -> List[float]:
    # STANDARD: 12/26/9
    if len(close) < 26:
        return [0.0] * len(close)
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal = ema(macd_line, 9)
    # Align lengths
    m = min(len(macd_line), len(signal))
    hist = [macd_line[i] - signal[i] for i in range(m)]
    # pad if needed
    if m < len(close):
        hist = [0.0] * (len(close) - m) + hist
    return hist

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def engineered_confidence(candles: List[List[float]]) -> Dict[str, Any]:
    """
    Returns:
    {
      "direction_from_engineering": "UP|DOWN|SIDE",
      "conf_engineered": float in [0.1, 0.9],
      "tp_pct_hint": float,
      "sl_pct_hint": float
    }
    """
    closes = [r[3] for r in candles]
    last_close = closes[-1]
    atr = compute_atr14(candles)
    atr_pct = atr / max(last_close, 1e-9)  # volatility ratio

    # Trend (MACD histogram z-score-ish scaling)
    hist = macd_hist(closes)
    h = hist[-1]
    # Normalize hist by price to make it scale-free; scale factor chosen empirically
    hist_norm = h / max(last_close * 0.002, 1e-9)  # 0.2% of price as scale
    # squash to [-1,1]
    trend_strength = math.tanh(hist_norm)  # sign = direction, magnitude = strength

    if trend_strength > 0.08:
        dir_eng = "UP"
    elif trend_strength < -0.08:
        dir_eng = "DOWN"
    else:
        dir_eng = "SIDE"

    # Volatility score: 0 at 0% ATR, 1 near ~2% ATR
    vol_score = _clamp(atr_pct / 0.02, 0.0, 1.0)

    # Confidence blend (ensure never 0.0/1.0)
    # weight trend magnitude more than volatility presence
    conf = 0.15 + 0.55 * abs(trend_strength) + 0.30 * vol_score
    conf = _clamp(conf, 0.10, 0.90)

    # TP/SL hints based on ATR
    # keep within the app's bounds: tp 0.5%..3%, sl 0.3%..2%
    tp_pct = _clamp(max(atr_pct * 1.2, 0.005), 0.005, 0.03)
    sl_pct = _clamp(max(atr_pct * 0.9, 0.003), 0.003, 0.02)

    return {
        "direction_from_engineering": dir_eng,
        "conf_engineered": conf,
        "tp_pct_hint": tp_pct,
        "sl_pct_hint": sl_pct,
    }


# -----------------------------
# Gemini helpers
# -----------------------------
def ask_gemini(prompt: str) -> Dict[str, Any]:
    try:
        r = model.generate_content(prompt)
        t = (r.text or "").strip()
        return json.loads(t)
    except Exception:
        # If it’s not JSON or Gemini hiccuped
        return {}

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/predict")
def predict(req: CandleReq):
    eng = engineered_confidence(req.candles)
    # Ask Gemini only for direction (we’ll ignore its confidence if it’s junk)
    g = ask_gemini(
        f"""
Return ONLY JSON:
{{"direction":"UP|SIDE|DOWN","confidence":0.0-1.0}}

PAIR:{req.pair}
TF:{req.interval}
CANDLES(60 [O,H,L,C,V]):{req.candles}
"""
    )

    # Direction: prefer Gemini if valid, else engineered
    direction = g.get("direction") if g.get("direction") in {"UP","DOWN","SIDE"} \
        else eng["direction_from_engineering"]

    # Confidence: safe blend (if Gemini has a float, use it; else 0.5)
    g_conf_raw = g.get("confidence", 0.5)
    try:
        g_conf = float(g_conf_raw)
    except Exception:
        g_conf = 0.5
    g_conf = _clamp(g_conf, 0.10, 0.90)

    # Final blend (50/50)
    confidence = _clamp(0.5 * g_conf + 0.5 * eng["conf_engineered"], 0.10, 0.90)

    return {
        "direction": direction,
        "confidence": confidence,
    }


@app.post("/recommend")
def recommend(req: RecommendReq):
    candles = req.candles[:]  # oldest->newest
    last_close = candles[-1][3]

    # Engineered metrics first
    eng = engineered_confidence(candles)

    # Ask Gemini for action + (optional) tp/sl fractions
    g = ask_gemini(
        f"""
You are a crypto trade recommender. Markets: {req.market_type.upper()}.
Given 60 candles [O,H,L,C,V], choose ONE action:
- spot: BUY or SELL or SIDE
- futures: LONG or SHORT or SIDE

Return TP/SL as FRACTIONS of entry price.

Return STRICT JSON ONLY:
{{
  "direction":"UP|SIDE|DOWN",
  "action":"BUY|SELL|LONG|SHORT|SIDE",
  "confidence":0.55,
  "tp_pct":0.012,
  "sl_pct":0.009
}}

PAIR:{req.pair}
TF:{req.interval}
CANDLES:{candles}
"""
    )

    # Direction
    direction = g.get("direction") if g.get("direction") in {"UP","DOWN","SIDE"} \
        else eng["direction_from_engineering"]

    # Confidence
    try:
        g_conf = float(g.get("confidence", 0.5))
    except Exception:
        g_conf = 0.5
    g_conf = _clamp(g_conf, 0.10, 0.90)
    confidence = _clamp(0.5 * g_conf + 0.5 * eng["conf_engineered"], 0.10, 0.90)

    # Action normalization by market type
    action = (g.get("action") or "SIDE").upper()
    if req.market_type == "spot":
        if direction == "UP" and action not in {"BUY", "SIDE"}:   action = "BUY"
        if direction == "DOWN" and action not in {"SELL", "SIDE"}: action = "SELL"
    else:  # futures
        if direction == "UP" and action not in {"LONG", "SIDE"}:  action = "LONG"
        if direction == "DOWN" and action not in {"SHORT", "SIDE"}: action = "SHORT"

    # TP/SL pct either from Gemini or engineered hints
    def _f(d: Dict, k: str, default: float) -> float:
        try:
            v = d.get(k, default)
            return float(v)
        except Exception:
            return default

    tp_pct = _clamp(_f(g, "tp_pct", eng["tp_pct_hint"]), 0.005, 0.03)
    sl_pct = _clamp(_f(g, "sl_pct", eng["sl_pct_hint"]), 0.003, 0.02)

    entry = last_close
    if action in {"BUY", "LONG"}:
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
    elif action in {"SELL", "SHORT"}:
        tp = entry * (1 - tp_pct)
        sl = entry * (1 + sl_pct)
    else:
        tp = entry
        sl = entry

    return {
        "direction": direction,
        "confidence": float(confidence),
        "action": action,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        # For debugging (optional):
        # "debug": { "eng": eng, "gemini_raw": g }
    }
