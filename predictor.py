"""
Forex Predictor Engine — H1 Timeframe
- Fetches real H1 candles via yfinance
- Embeds data directly into dashboard.html (no server needed, works on phone)
- Run every hour via GitHub Actions or cron

pip install yfinance pandas numpy
"""

import json, math, re, os
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np


# ── CONFIG ────────────────────────────────────────────────────────────────────

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
}

# H1 = 1h candles. yfinance allows up to 730 days for 1h but typically 60 days works reliably.
LOOKBACK_DAYS  = 30     # 30 days of H1 candles (~720 candles) — plenty for analysis
INTERVAL       = "1h"

ATR_PERIOD     = 14
RSI_PERIOD     = 14
BB_PERIOD      = 20
MARKOV_SEQ_LEN = 3
TP_MULTIPLIER  = 1.5
SL_MULTIPLIER  = 1.0


# ── DATA FETCHING ─────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=INTERVAL,
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df[["Open", "High", "Low", "Close"]].dropna()

    if len(df) < 40:
        raise ValueError(f"Not enough H1 candles ({len(df)}) for {ticker}")

    return df


# ── INDICATORS ────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr)


def compute_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> float:
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_zscore(df: pd.DataFrame, period: int = BB_PERIOD) -> float:
    close = df["Close"]
    mean  = close.rolling(period).mean()
    std   = close.rolling(period).std()
    z     = (close - mean) / std.replace(0, np.nan)
    return float(z.iloc[-1])


def compute_bollinger(df: pd.DataFrame, period: int = BB_PERIOD):
    close = df["Close"]
    mean  = close.rolling(period).mean().iloc[-1]
    std   = close.rolling(period).std().iloc[-1]
    upper = float(mean + 2 * std)
    lower = float(mean - 2 * std)
    price = float(close.iloc[-1])
    return upper, lower, float(mean), price


def compute_macd(df: pd.DataFrame):
    close  = df["Close"]
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = (macd - signal).iloc[-1]
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist)


# ── MARKOV CHAIN ──────────────────────────────────────────────────────────────

def build_markov_chain(returns: pd.Series, seq_len: int = MARKOV_SEQ_LEN) -> dict:
    directions  = ["U" if r > 0 else "D" for r in returns]
    transitions = defaultdict(lambda: {"U": 0, "D": 0})
    for i in range(seq_len, len(directions)):
        state = "".join(directions[i - seq_len:i])
        transitions[state][directions[i]] += 1
    return dict(transitions)


def markov_predict(transitions: dict, recent_returns: pd.Series, seq_len: int = MARKOV_SEQ_LEN):
    directions    = ["U" if r > 0 else "D" for r in recent_returns]
    current_state = "".join(directions[-seq_len:])
    counts        = transitions.get(current_state, {"U": 0, "D": 0})
    total         = counts["U"] + counts["D"]
    if total == 0:
        return 0.5, 0.5, current_state, 0
    return counts["U"] / total, counts["D"] / total, current_state, total


# ── INDICATOR SCORE ───────────────────────────────────────────────────────────

def indicator_score(rsi, zscore, macd_hist, upper_bb, lower_bb, price) -> float:
    score, weights = 0.0, 0.0

    if not math.isnan(rsi):
        if rsi < 30:   score += 1.0 * ((30 - rsi) / 30)
        elif rsi > 70: score -= 1.0 * ((rsi - 70) / 30)
        weights += 1.0

    if not math.isnan(zscore):
        score   -= 0.5 * max(-2, min(2, zscore)) / 2
        weights += 0.5

    if not math.isnan(macd_hist):
        score   += 0.5 * (1 if macd_hist > 0 else -1)
        weights += 0.5

    if not (math.isnan(upper_bb) or math.isnan(lower_bb)):
        band_range = upper_bb - lower_bb
        if band_range > 0:
            pct_b = (price - lower_bb) / band_range
            if pct_b < 0.2:   score   += 0.4
            elif pct_b > 0.8: score   -= 0.4
            weights += 0.4

    return score / weights if weights > 0 else 0.0


# ── KELLY ─────────────────────────────────────────────────────────────────────

def kelly_fraction(p_win: float, tp: float, sl: float) -> float:
    if sl <= 0: return 0.0
    R = tp / sl
    f = p_win - (1 - p_win) / R
    return round(max(0.0, min(f, 0.25)) * 100, 1)


# ── ANALYZE ONE PAIR ──────────────────────────────────────────────────────────

def analyze_pair(name: str, ticker: str) -> dict:
    try:
        df      = fetch_ohlcv(ticker)
        price   = float(df["Close"].iloc[-1])
        returns = df["Close"].pct_change().dropna()
        atr     = compute_atr(df)

        rsi                        = compute_rsi(df)
        zscore                     = compute_zscore(df)
        upper_bb, lower_bb, mb, _  = compute_bollinger(df)
        _, _, macd_hist            = compute_macd(df)

        transitions                        = build_markov_chain(returns)
        p_up, p_down, mstate, mobs         = markov_predict(transitions, returns)

        ind_s       = indicator_score(rsi, zscore, macd_hist, upper_bb, lower_bb, price)
        combined    = 0.45 * p_up + 0.55 * ((ind_s + 1) / 2)
        direction   = "UP" if combined >= 0.5 else "DOWN"
        confidence  = round(max(combined, 1 - combined) * 100, 1)

        tp_dist = atr * TP_MULTIPLIER
        sl_dist = atr * SL_MULTIPLIER
        tp = price + tp_dist if direction == "UP" else price - tp_dist
        sl = price - sl_dist if direction == "UP" else price + sl_dist

        pip_f   = 100 if "JPY" in name else 10000
        tp_pips = round(abs(tp - price) * pip_f, 1)
        sl_pips = round(abs(sl - price) * pip_f, 1)

        recent_candles = ["U" if r > 0 else "D" for r in returns.tail(5).tolist()]

        # Last candle time (human readable, UTC)
        last_time = df.index[-1]
        if hasattr(last_time, 'strftime'):
            last_candle = last_time.strftime("%Y-%m-%d %H:%M UTC")
        else:
            last_candle = str(last_time)

        return {
            "pair":           name,
            "price":          round(price, 5),
            "direction":      direction,
            "confidence":     confidence,
            "tp":             round(tp, 5),
            "sl":             round(sl, 5),
            "tp_pips":        tp_pips,
            "sl_pips":        sl_pips,
            "kelly_pct":      kelly_fraction(confidence / 100, tp_dist, sl_dist),
            "atr":            round(atr, 5),
            "rsi":            round(rsi, 1),
            "zscore":         round(zscore, 2),
            "macd_bull":      macd_hist > 0,
            "markov_state":   mstate,
            "markov_p_up":    round(p_up * 100, 1),
            "markov_obs":     mobs,
            "ind_score":      round(ind_s, 3),
            "recent_candles": recent_candles,
            "last_candle":    last_candle,
            "candles_used":   len(df),
            "error":          None,
        }

    except Exception as e:
        return {"pair": name, "error": str(e), "direction": None}


# ── EMBED INTO HTML ───────────────────────────────────────────────────────────

def embed_data_into_html(results: list, html_path: str = "dashboard.html"):
    """
    Reads the dashboard template and replaces the FOREX_DATA placeholder
    with the freshly computed JSON. This makes the HTML fully self-contained —
    no server, no fetch() call needed. Works when opened directly on a phone.
    """
    template_path = Path(html_path)
    if not template_path.exists():
        print(f"⚠  {html_path} not found — skipping HTML update.")
        return

    html = template_path.read_text(encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload   = json.dumps(results, indent=2)

    # Replace the data block between the sentinel comments
    pattern     = r"(// <<<FOREX_DATA_START>>>).*?(// <<<FOREX_DATA_END>>>)"
    replacement = f"// <<<FOREX_DATA_START>>>\nconst FOREX_DATA={payload};\nconst GENERATED_AT='{timestamp}';\n// <<<FOREX_DATA_END>>>"
    new_html, n = re.subn(pattern, replacement, html, flags=re.DOTALL)

    if n == 0:
        print("⚠  Sentinel comments not found in dashboard.html — HTML not updated.")
        return

    template_path.write_text(new_html, encoding="utf-8")
    print(f"✓  dashboard.html updated with live data ({timestamp})")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*58}")
    print(f"  FX PREDICTOR  |  H1  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*58}")

    results = []
    for name, ticker in PAIRS.items():
        print(f"  Fetching {name} ({ticker}) ...", end="", flush=True)
        r = analyze_pair(name, ticker)
        results.append(r)
        if r["error"]:
            print(f"  ✗  {r['error']}")
        else:
            print(f"  ✓  {r['direction']}  {r['confidence']}%  |  last candle: {r['last_candle']}")

    print(f"{'='*58}\n")

    # Always save JSON alongside the HTML
    with open("forex_data.json", "w") as f:
        json.dump(results, f, indent=2)

    # Embed directly into the HTML so it works without a server
    embed_data_into_html(results, html_path="dashboard.html")

    print("Done. Open dashboard.html on any device.")


if __name__ == "__main__":
    main()
