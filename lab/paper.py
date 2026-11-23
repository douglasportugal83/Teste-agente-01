"""Paper trading incremental: processa cada candle fechado novo, com as mesmas regras do backtest.

Fills são SIMULADOS (abertura do candle seguinte, stop no preço do stop ou no gap) e custos são estimativas.
Nenhuma ordem real é enviada: não existe conexão com corretora neste código.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

from . import config as C
from . import data
from .research import Market
from .sentinel import kill_reason, review
from .strategies import FAMILY_BY_KEY

STATE = C.STATE_DIR / "paper.json"
CHAMPS = C.STATE_DIR / "champions.json"
WINDOW = 1500  # barras para aquecer EMA 200 / ADX


def iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat(timespec="minutes")


def load_json(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=_np))


def _np(x):
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def new_state(now_ms):
    books = {}
    for m in C.MARKETS:
        df = data.load(m)
        books[m] = {"last_bar": int(df.open_time.iloc[-1]), "position": None, "pending": None}
    return {"started_at_utc": iso(now_ms), "account": {
        "equity": C.CAPITAL, "realized": 0.0, "peak_equity": C.CAPITAL, "loss_streak": 0},
        "halted": None, "books": books, "trades": [], "reviews": [], "equity_curve": [], "alerts": []}


def _close(st, m, book, price, t, why):
    p = book["position"]
    gross = p["side"] * (price - p["entry"]) * p["qty"]
    cost = C.COST_PER_SIDE * price * p["qty"]
    pnl = gross - cost - p["costs"]
    st["account"]["realized"] += gross - cost - p["funding"]
    st["account"]["loss_streak"] = st["account"]["loss_streak"] + 1 if pnl - p["funding"] <= 0 else 0
    st["trades"].append({**p, "market": m, "exit": price, "exit_time": iso(t), "exit_reason": why,
                         "pnl_usd": round(pnl - p["funding"], 2),
                         "ret": round((pnl - p["funding"]) / (p["entry"] * p["qty"]), 5)})
    book["position"] = None


def _unrealized(st, last_close):
    u = 0.0
    for m, b in st["books"].items():
        p = b["position"]
        if p:
            u += p["side"] * (last_close[m] - p["entry"]) * p["qty"] - p["funding"]
    return u


def step(now_ms: int | None = None, fetch: bool = True) -> dict:
    now_ms = now_ms or int(time.time() * 1000)
    champs = load_json(CHAMPS)
    stale = False
    if fetch:
        for m in C.MARKETS:
            try:
                data.update(m, now_ms)
            except Exception as e:  # sem dados novos: o Sentinel decide
                print(f"[feed] {m}: {e}")
    st = load_json(STATE) or new_state(now_ms)
    mks, events = {}, []
    for m in C.MARKETS:
        df = data.load(m).iloc[-WINDOW:].reset_index(drop=True)
        mks[m] = Market.from_df(df)
        last_close_time = int(df.open_time.iloc[-1]) + C.BAR_MS
        if now_ms - last_close_time > C.STALE_FEED_BARS * C.BAR_MS:
            stale = True
        for i in np.nonzero(mks[m].t > st["books"][m]["last_bar"])[0]:
            events.append((int(mks[m].t[i]), m, int(i)))
    events.sort()
    acct = st["account"]
    last_close = {m: float(mks[m].c[-1]) for m in C.MARKETS}
    closes = {m: None for m in C.MARKETS}
    for t, m, i in events:
        mk, book = mks[m], st["books"][m]
        o, h, l, c = mk.o[i], mk.h[i], mk.l[i], mk.c[i]
        pend = book["pending"]
        # a) ordem pendente na abertura
        if pend and pend["action"] == "exit" and book["position"]:
            _close(st, m, book, o, t, pend["reason"])
        elif pend and pend["action"] == "enter" and not book["position"] and st["halted"] is None:
            book["position"] = {**pend["order"], "entry": float(o), "entry_time": iso(t), "held": 0,
                                "stop": float(o - pend["order"]["side"] * pend["order"]["stop_distance"]),
                                "costs": C.COST_PER_SIDE * o * pend["order"]["qty"], "funding": 0.0}
        book["pending"] = None
        # b) stop intrabarra
        p = book["position"]
        if p:
            hit = (p["side"] == 1 and l <= p["stop"]) or (p["side"] == -1 and h >= p["stop"])
            if hit:
                gap = (p["side"] == 1 and o < p["stop"]) or (p["side"] == -1 and o > p["stop"])
                _close(st, m, book, float(o if gap else p["stop"]), t, "stop")
            else:
                p["held"] += 1
                if p["side"] == -1:
                    p["funding"] += C.SHORT_FUNDING_PER_BAR * c * p["qty"]
        # c) contabilidade e travas
        closes[m] = float(c)
        mark = {k: (closes[k] if closes[k] is not None else last_close[k]) for k in C.MARKETS}
        acct["equity"] = round(C.CAPITAL + acct["realized"] + _unrealized(st, mark), 2)
        acct["peak_equity"] = max(acct["peak_equity"], acct["equity"])
        if st["halted"] is None:
            why = kill_reason(acct, False)
            if why:
                _halt(st, why, t)
        # d) decisões no fechamento
        reg = C.REGIMES[int(mk.reg[i])]
        p = book["position"]
        if p:
            fam = FAMILY_BY_KEY[p["family"]]
            _, lx, _, sx = fam.signals(mk.o[:i + 1], mk.h[:i + 1], mk.l[:i + 1], mk.c[:i + 1], p["params"]["p"])
            why = ("trava do Sentinel" if st["halted"] else
                   "troca de regime" if reg != C.REGIMES[fam.regime] else
                   "holding máximo" if p["held"] >= C.MAX_HOLD_BARS else
                   "sinal de saída" if (p["side"] == 1 and lx[-1]) or (p["side"] == -1 and sx[-1]) else None)
            if why:
                book["pending"] = {"action": "exit", "reason": why}
        elif champs and reg in champs.get("routing", {}):
            ch = (champs["routing"][reg].get(m) or {}).get("champion")
            if ch:
                _maybe_enter(st, m, book, mk, i, t, reg, ch, stale)
        book["last_bar"] = t
    if stale and st["halted"] is None:
        _halt(st, "feed de dados parado", now_ms)
    acct["equity"] = round(C.CAPITAL + acct["realized"] + _unrealized(st, last_close), 2)
    acct["peak_equity"] = max(acct["peak_equity"], acct["equity"])
    st["equity_curve"].append([iso(now_ms), acct["equity"]])
    st["equity_curve"] = st["equity_curve"][-2000:]
    st["reviews"] = st["reviews"][-500:]
    st["last_run_utc"] = iso(now_ms)
    save_json(STATE, st)
    return st


def _maybe_enter(st, m, book, mk, i, t, reg, ch, stale):
    fam = FAMILY_BY_KEY[ch["family"]]
    le, _, se, _ = fam.signals(mk.o[:i + 1], mk.h[:i + 1], mk.l[:i + 1], mk.c[:i + 1], ch["params"]["p"])
    side = 1 if le[-1] else -1 if se[-1] else 0
    if not side or np.isnan(mk.at[i]):
        return
    eq = st["account"]["equity"]
    dist = float(ch["params"]["stop"] * mk.at[i])
    qty = C.RISK_PER_TRADE * eq / dist
    qty = min(qty, C.MAX_NOTIONAL_PER_MARKET * eq / mk.c[i])
    order = {"family": fam.key, "params": ch["params"], "side": side, "qty": float(qty),
             "stop_distance": dist, "notional": float(qty * mk.c[i]), "risk_usd": float(qty * dist),
             "strategy_regime": C.REGIMES[fam.regime], "signal": True,
             "reason": f"{fam.key} ({fam.param_name}={ch['params']['p']}): {fam.entry}"}
    rv = review(order, {"regime": reg, "equity": eq, "stale": stale, "halted": st["halted"]})
    st["reviews"].append({"time": iso(t), "market": m, "order": order, **rv})
    if rv["approved"]:
        book["pending"] = {"action": "enter", "order": order}


def _halt(st, why, t):
    st["halted"] = {"reason": why, "at_utc": iso(t)}
    st["alerts"].append({"time": iso(t), "type": "halt", "message": f"Sentinel parou o bot: {why}",
                         "notified": False})
    for b in st["books"].values():  # zera posições na próxima abertura
        if b["position"]:
            b["pending"] = {"action": "exit", "reason": "trava do Sentinel"}


def resume():
    st = load_json(STATE)
    if st and st["halted"]:
        st["alerts"].append({"time": iso(time.time() * 1000), "type": "resume",
                             "message": f"Retomado manualmente (trava anterior: {st['halted']['reason']})",
                             "notified": True})
        st["halted"] = None
        st["account"]["loss_streak"] = 0
        st["account"]["peak_equity"] = st["account"]["equity"]
        save_json(STATE, st)
    return st
