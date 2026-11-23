"""Dados reais com fonte e timestamp. Nada é inventado: se o download falhar, o erro sobe."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from . import config as C

API = "https://data-api.binance.vision/api/v3/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time"]


def _get(params: dict) -> list:
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:  # rede instável: tenta de novo com espera
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"falha ao baixar {url}: {last}")


def fetch_klines(symbol: str, start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
    rows, cur = [], start_ms
    while True:
        params = {"symbol": symbol, "interval": C.INTERVAL, "startTime": cur, "limit": 1000}
        if end_ms:
            params["endTime"] = end_ms
        batch = _get(params)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + C.BAR_MS
        if len(batch) < 1000 or (end_ms and nxt > end_ms):
            break
        cur = nxt
    df = pd.DataFrame([r[:7] for r in rows], columns=COLS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.drop_duplicates("open_time").reset_index(drop=True)


def _paths(symbol: str):
    return C.DATA_DIR / f"{symbol}_{C.INTERVAL}.csv", C.DATA_DIR / f"{symbol}_{C.INTERVAL}.meta.json"


def update(symbol: str, now_ms: int | None = None) -> pd.DataFrame:
    """Baixa/atualiza o cache e mantém só candles FECHADOS."""
    now_ms = now_ms or int(time.time() * 1000)
    csv, meta = _paths(symbol)
    C.DATA_DIR.mkdir(exist_ok=True)
    start = now_ms - int(C.YEARS_OF_DATA * 365.25 * 24 * 3600 * 1000)
    old = pd.read_csv(csv) if csv.exists() else None
    if old is not None and len(old):
        new = fetch_klines(symbol, int(old.open_time.iloc[-1]))
        df = pd.concat([old, new]).drop_duplicates("open_time", keep="last")
    else:
        df = fetch_klines(symbol, start)
    df = df[df.close_time < now_ms]  # descarta o candle em formação
    df = df[df.open_time >= start].sort_values("open_time").reset_index(drop=True)
    df.to_csv(csv, index=False)
    meta.write_text(json.dumps({
        "symbol": symbol, "interval": C.INTERVAL, "source": C.DATA_SOURCE,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_bar_utc": _iso(df.open_time.iloc[0]), "last_bar_utc": _iso(df.open_time.iloc[-1]),
        "bars": len(df),
    }, indent=2))
    return df


def load(symbol: str) -> pd.DataFrame:
    csv, _ = _paths(symbol)
    return pd.read_csv(csv)


def meta(symbol: str) -> dict:
    _, m = _paths(symbol)
    return json.loads(m.read_text())


def _iso(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat(timespec="minutes")
