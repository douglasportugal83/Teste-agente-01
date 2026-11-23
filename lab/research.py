"""Núcleo de pesquisa: otimização, walk-forward, permutações, métricas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .engine import regime_bh, run
from .indicators import atr
from .regime import regimes


@dataclass
class Market:
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    t: np.ndarray  # open_time em ms
    at: np.ndarray
    reg: np.ndarray

    @classmethod
    def from_arrays(cls, o, h, l, c, t):
        _, reg = regimes(h, l, c)
        return cls(o, h, l, c, t, atr(h, l, c, 14), reg)

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        return cls.from_arrays(*(df[k].to_numpy(float) for k in ["open", "high", "low", "close"]),
                               df.open_time.to_numpy(np.int64))


class Runner:
    """Roda uma família num mercado, guardando sinais por parâmetro."""

    def __init__(self, fam, mk: Market):
        self.fam, self.mk, self._sig = fam, mk, {}

    def sig(self, p):
        if p not in self._sig:
            m = self.mk
            self._sig[p] = self.fam.signals(m.o, m.h, m.l, m.c, p)
        return self._sig[p]

    def run(self, params, start, end):
        m = self.mk
        le, lx, se, sx = self.sig(params["p"])
        return run(m.o, m.h, m.l, m.c, m.at, m.reg, self.fam.regime, le, lx, se, sx,
                   float(params["stop"]), C.MAX_HOLD_BARS, C.COST_PER_SIDE,
                   C.SHORT_FUNDING_PER_BAR, start, end)

    def objective(self, params, start, end):
        r = self.run(params, start, end)[0][start:end]
        return float(np.log1p(r).sum())

    def optimize(self, start, end):
        best, best_v, vals = None, -np.inf, []
        for p in self.fam.grid():
            v = self.objective(p, start, end)
            vals.append(v)
            if v > best_v:
                best, best_v = p, v
        return best, best_v, vals

    def walk_forward(self, start, end):
        rets = np.zeros(self.mk.c.size)
        folds, entries, exits, sides, trets = [], [], [], [], []
        s = start
        while s + C.WF_TRAIN_BARS + C.WF_TEST_BARS // 3 <= end:
            ts, te = s + C.WF_TRAIN_BARS, min(s + C.WF_TRAIN_BARS + C.WF_TEST_BARS, end)
            p, _, _ = self.optimize(s, ts)
            r, e, x, sd, tr = self.run(p, ts, te)
            rets[ts:te] = r[ts:te]
            folds.append({"train": (s, ts), "test": (ts, te), "params": p,
                          "ret": float(np.expm1(np.log1p(r[ts:te]).sum()))})
            entries += list(e); exits += list(x); sides += list(sd); trets += list(tr)
            s += C.WF_TEST_BARS
        wf_start = start + C.WF_TRAIN_BARS
        wf_end = folds[-1]["test"][1] if folds else wf_start
        return {"rets": rets, "start": wf_start, "end": wf_end, "folds": folds,
                "trades": np.array(trets), "entries": np.array(entries, dtype=np.int64)}


def metrics(r: np.ndarray, trades: np.ndarray, t: np.ndarray, start: int, end: int) -> dict:
    seg = r[start:end]
    eq = np.cumprod(1 + seg)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = float((1 - eq / peak).max()) if eq.size else 0.0
    streak = cur = 0
    for x in trades:
        cur = cur + 1 if x <= 0 else 0
        streak = max(streak, cur)
    wins, losses = trades[trades > 0].sum(), -trades[trades <= 0].sum()
    months = pd.Series(seg, index=pd.to_datetime(t[start:end], unit="ms")).groupby(pd.Grouper(freq="ME"))
    monthly = months.apply(lambda x: float(np.prod(1 + x) - 1))
    worst = (str(monthly.idxmin().strftime("%Y-%m")), float(monthly.min())) if len(monthly) else ("-", 0.0)
    return {
        "net_return": float(eq[-1] - 1) if eq.size else 0.0,
        "max_dd": dd,
        "trades": int(trades.size),
        "win_rate": float((trades > 0).mean()) if trades.size else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0),
        "longest_losing_streak": int(streak),
        "worst_month": worst,
        "avg_trade": float(trades.mean()) if trades.size else 0.0,
    }


def benchmark(mk: Market, regime: int, start: int, end: int) -> dict:
    full = float(mk.c[end - 1] / mk.c[start] - 1)
    in_regime = float(np.expm1(regime_bh(mk.c, mk.reg, regime, start, end)))
    # Para estratégias de baixa (vendidas), segurar é perder; a régua mínima vira o caixa (0).
    hurdle = max(0.0, in_regime)
    return {"buy_hold_full": full, "buy_hold_in_regime": in_regime, "hurdle": hurdle}


def lookahead_check(fam, mk: Market, params, samples=6, seed=0) -> bool:
    """Recalcula sinais com dados truncados em pontos aleatórios: se mudar, há vazamento do futuro."""
    rng = np.random.default_rng(seed)
    full = fam.signals(mk.o, mk.h, mk.l, mk.c, params["p"])
    n = mk.c.size
    for cut in rng.integers(n // 2, n - 1, samples):
        part = fam.signals(mk.o[:cut + 1], mk.h[:cut + 1], mk.l[:cut + 1], mk.c[:cut + 1], params["p"])
        if any(bool(a[cut]) != bool(b[cut]) for a, b in zip(full, part)):
            return False
    _, reg_part = regimes(mk.h[:n - 50], mk.l[:n - 50], mk.c[:n - 50])
    return bool((reg_part == mk.reg[:n - 50]).all())
