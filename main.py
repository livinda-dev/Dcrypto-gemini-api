# app.py
import os, json, math, re
from typing import List, Literal, Optional, Dict, Any

import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

# ---------- Gemini setup ----------
# Requires: pip install google-generativeai fastapi uvicorn
# Make sure: setx GEMINI_API_KEY "your-key"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL_ID = "models/gemini-2.5-flash"   # fast & cheap; change to pro if you prefer
model = genai.GenerativeModel(MODEL_ID)

# ---------- FastAPI ----------
app = FastAPI()

# ---------- Schemas ----------
class CandleReq(BaseModel):
    candles: List[List[float]]   # 60 x [open, high, low, close, volume] (oldest -> newest)
    pair: str
    interval: str                # e.g. "15m"

class RecommendReq(CandleReq):
    market_type: Literal["spot", "futures"]

# ---------- Helpers ----------
def ema(series: List[float], n: int) -> List[float]:
    if not series: return []
    a = 2.0 / (n + 1.0)
    out = [series[0]]
    for i in range(1, len(series)):
        out.append(a * series[i] + (1.0 - a) * out[-1])
    return out

def rsi14(closes: List[float]) -> List[float]:
    if len(closes) < 2: return [50.0] * len(closes)
    gains, losses = [0.0], [0.0]
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    rsi = []
    for i in range(len(closes)):
        if i < 14:
            rsi.append(50.0)
        else:
            avg_gain = sum(gains[i-13:i+1]) / 14.0
            avg_loss = sum(losses[i-13:i+1]) / 14.0
            rs = avg_gain / (avg_loss + 1e-12)
            rsi.append(100.0 - (100.0 / (1.0 + rs)))
    return rsi

def bb20(closes: List[float]) -> Dict[str, List[float]]:
    out_mid, out_up, out_low = [], [], []
    for i in range(len(closes)):
        if i < 19:
            out_mid.append(closes[i])
            out_up.append(closes[i])
            out_low.append(closes[i])
        else:
            win = closes[i-19:i+1]
            m = sum(win) / 20.0
            var = sum((x - m) ** 2 for x in win) / 20.0
            sd = math.sqrt(var)
            out_mid.append(m)
            out_up.append(m + 2.0 * sd)
            out_low.append(m - 2.0 * sd)
    return {"mid": out_mid, "up": out_up, "low": out_low}

def atr14(ohlc: List[List[float]]) -> List[float]:
    # ohlc: [open, high, low, close, volume]
    trs = [0.0]
    for i in range(1, len(ohlc)):
        _, h, l, c, _ = ohlc[i]
        _, _, _, pc, _ = ohlc[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    out = []
    for i in range(len(trs)):
        if i < 14:
            out.append(sum(trs[:i+1]) / (i + 1))
        else:
            out.append(sum(trs[i-13:i+1]) / 14.0)
    return out

def indicators_last(candles: List[List[float]]) -> Dict[str, float]:
    closes = [c[3] for c in candles]
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
    signal = ema(macd, 9)
    hist = [m - s for m, s in zip(macd, signal)]
    rsi = rsi14(closes)
    bb = bb20(closes)
    atr = atr14(candles)

    i = len(closes) - 1
    return {
        "close": closes[i],
        "ema12": ema12[i],
        "ema26": ema26[i],
        "macd": macd[i],
        "signal": signal[i],
        "hist": hist[i],
        "rsi14": rsi[i],
        "bb_mid": bb["mid"][i],
        "bb_up": bb["up"][i],
        "bb_low": bb["low"][i],
        "atr14": atr[i],
    }

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to pull the first {...} JSON object from LLM text.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def normalize_action_for_market(direction: str, action: str, market: str) -> str:
    market = market.lower()
    direction = direction.upper()
    action = action.upper()
    if market == "spot":
        valid = {"BUY", "SELL", "SIDE"}
        if direction == "UP" and action not in valid:  action = "BUY"
        if direction == "DOWN" and action not in valid: action = "SELL"
        if direction == "SIDE": action = "SIDE"
        if action not in valid: action = "SIDE"
    else:
        valid = {"LONG", "SHORT", "SIDE"}
        if direction == "UP" and action not in valid:  action = "LONG"
        if direction == "DOWN" and action not in valid: action = "SHORT"
        if direction == "SIDE": action = "SIDE"
        if action not in valid: action = "SIDE"
    return action

def tp_sl_prices(entry: float, action: str, tp_pct: float, sl_pct: float):
    action = action.upper()
    if action in ("BUY", "LONG"):
        return entry * (1 + tp_pct), entry * (1 - sl_pct)
    if action in ("SELL", "SHORT"):
        return entry * (1 - tp_pct), entry * (1 + sl_pct)
    return entry, entry

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID}

@app.post("/recommend")
def recommend(req: RecommendReq):
    """
    Returns JSON:
    {
      "direction": "UP|SIDE|DOWN",
      "confidence": 0.10..0.90,
      "primary":  { "action": "...", "tp_pct": ..., "sl_pct": ..., "entry": ..., "tp": ..., "sl": ... },
      "alternate":{ "action": "...", "tp_pct": ..., "sl_pct": ..., "entry": ..., "tp": ..., "sl": ... },
      "indicators": { last indicator snapshot ... }
    }
    """
    candles = req.candles[:]  # oldest -> newest
    if len(candles) < 20:
        return {
            "direction": "SIDE",
            "confidence": 0.10,
            "primary": {"action": "SIDE", "tp_pct": 0.01, "sl_pct": 0.008, "entry": 0, "tp": 0, "sl": 0},
            "alternate": {"action": "SIDE", "tp_pct": 0.008, "sl_pct": 0.006, "entry": 0, "tp": 0, "sl": 0},
            "indicators": {}
        }

    info = indicators_last(candles)
    entry = info["close"]
    # ATR-based bands as fallback
    base_tp = clamp(info["atr14"] / max(entry, 1e-9), 0.005, 0.03)   # 0.5%..3%
    base_sl = clamp(base_tp * 0.8, 0.003, 0.02)                      # 0.3%..2%

    # Build compact prompt (MACD 12/26/9 as requested)
    prompt = f"""
You are a crypto recommender.
Decide the trend direction and propose TWO actions (primary + alternate).

Market: {req.market_type.upper()}
Pair: {req.pair}
TF: {req.interval}

Recent indicators (latest value only):
- Price (close): {entry:.8f}
- EMA12: {info['ema12']:.8f}
- EMA26: {info['ema26']:.8f}
- MACD(12,26): {info['macd']:.8f}
- Signal(9): {info['signal']:.8f}
- Histogram: {info['hist']:.8f}
- RSI14: {info['rsi14']:.2f}
- BB20 mid: {info['bb_mid']:.8f}, up: {info['bb_up']:.8f}, low: {info['bb_low']:.8f}
- ATR14: {info['atr14']:.8f}

Constraints:
- direction ∈ ["UP","SIDE","DOWN"]
- confidence is float in [0.10, 0.90] (never 0.0, never 1.0).
- For SPOT: actions ∈ ["BUY","SELL","SIDE"]
- For FUTURES: actions ∈ ["LONG","SHORT","SIDE"]
- tp_pct ∈ [0.005, 0.03], sl_pct ∈ [0.003, 0.02] (fractions, not percents).
- Keep alternate different from primary if reasonable.

Return STRICT JSON ONLY, no markdown, exactly like:
{{
  "direction": "UP",
  "confidence": 0.42,
  "primary":   {{"action":"LONG","tp_pct":0.012,"sl_pct":0.009}},
  "alternate": {{"action":"SIDE","tp_pct":0.008,"sl_pct":0.006}}
}}
"""

    # Call Gemini
    try:
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
    except Exception:
        raw = ""

    data = extract_json(raw) or {}

    # Defaults
    direction = str(data.get("direction", "SIDE")).upper()
    confidence = float(data.get("confidence", 0.10))
    confidence = clamp(confidence, 0.10, 0.90)

    # Pull primary/alternate with fallbacks
    def get_block(key: str) -> Dict[str, float | str]:
        blk = data.get(key, {}) if isinstance(data.get(key, {}), dict) else {}
        action = str(blk.get("action", "SIDE")).upper()
        tp_pct = blk.get("tp_pct", base_tp)
        sl_pct = blk.get("sl_pct", base_sl)
        try: tp_pct = float(tp_pct)
        except: tp_pct = base_tp
        try: sl_pct = float(sl_pct)
        except: sl_pct = base_sl
        tp_pct = clamp(tp_pct, 0.005, 0.03)
        sl_pct = clamp(sl_pct, 0.003, 0.02)
        return {"action": action, "tp_pct": tp_pct, "sl_pct": sl_pct}

    primary = get_block("primary")
    alternate = get_block("alternate")

    # Normalize actions to market & direction
    primary["action"]   = normalize_action_for_market(direction, primary["action"], req.market_type)
    alternate["action"] = normalize_action_for_market(direction, alternate["action"], req.market_type)

    # Compute TP/SL prices
    p_tp, p_sl = tp_sl_prices(entry, primary["action"], primary["tp_pct"], primary["sl_pct"])
    a_tp, a_sl = tp_sl_prices(entry, alternate["action"], alternate["tp_pct"], alternate["sl_pct"])

    primary_out = {
        "action": primary["action"],
        "tp_pct": primary["tp_pct"],
        "sl_pct": primary["sl_pct"],
        "entry": float(entry),
        "tp": float(p_tp),
        "sl": float(p_sl),
    }
    alternate_out = {
        "action": alternate["action"],
        "tp_pct": alternate["tp_pct"],
        "sl_pct": alternate["sl_pct"],
        "entry": float(entry),
        "tp": float(a_tp),
        "sl": float(a_sl),
    }

    return {
        "direction": direction if direction in ("UP","SIDE","DOWN") else "SIDE",
        "confidence": float(confidence),
        "primary": primary_out,
        "alternate": alternate_out,
        "indicators": info,
    }
