import numpy as np
import pandas as pd
import pytest

from lab import config as C
from lab import data, paper, sentinel
from lab.engine import run
from lab.research import Market
from lab.strategies import FAMILY_BY_KEY


def frame(n=2600, seed=11):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    o = np.concatenate([[c[0]], c[:-1]]) * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    t = np.arange(n, dtype=np.int64) * C.BAR_MS + 1_600_000_000_000
    return pd.DataFrame({"open_time": t, "open": o, "high": h, "low": l, "close": c, "volume": 1.0,
                         "close_time": t + C.BAR_MS - 1})


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    df = frame()
    monkeypatch.setattr(C, "MARKETS", ["XUSDT"])
    monkeypatch.setattr(data, "load", lambda m: df)
    monkeypatch.setattr(paper, "STATE", tmp_path / "paper.json")
    monkeypatch.setattr(paper, "CHAMPS", tmp_path / "champions.json")
    monkeypatch.setattr(C, "KILL_LOSS_STREAK", 10**6)
    monkeypatch.setattr(C, "KILL_DRAWDOWN", 2.0)
    return df


def test_paper_matches_backtest_timing(sandbox, monkeypatch):
    df = sandbox
    fam = FAMILY_BY_KEY["donchian_breakout"]
    params = {"p": 20, "stop": 2.0}
    win = df.iloc[-paper.WINDOW:].reset_index(drop=True)
    mk = Market.from_df(win)
    reg = np.full(mk.c.size, C.BULL, np.int8)
    mk.reg = reg
    # força regime de alta para isolar a lógica de execução
    monkeypatch.setattr(paper.Market, "from_df", classmethod(lambda cls, d: mk))
    champs = {"routing": {"alta": {"XUSDT": {"champion": {"family": fam.key, "params": params}}}}}
    paper.save_json(paper.CHAMPS, champs)
    start = 600
    st = paper.new_state(0)
    st["books"]["XUSDT"]["last_bar"] = int(mk.t[start])
    paper.save_json(paper.STATE, st)
    now = int(mk.t[-1]) + C.BAR_MS + 1
    st = paper.step(now_ms=now, fetch=False)
    le, lx, se, sx = fam.signals(mk.o, mk.h, mk.l, mk.c, params["p"])
    _, e, x, s, tr = run(mk.o, mk.h, mk.l, mk.c, mk.at, reg, C.BULL, le, lx, se, sx, 2.0,
                         C.MAX_HOLD_BARS, C.COST_PER_SIDE, 0.0, start, mk.c.size)
    closed = len(st["trades"])
    assert closed >= 5
    t_iso = [paper.iso(v) for v in mk.t]
    assert [t["entry_time"] for t in st["trades"]] == [t_iso[i] for i in e[:closed]]
    assert [t["exit_time"] for t in st["trades"]] == [t_iso[i] for i in x[:closed]]
    # retornos iguais ao backtest (o paper cobra o custo de saída sobre o preço de saída)
    assert np.allclose([t["ret"] for t in st["trades"]], tr[:closed], atol=1e-3)
    # Sentinel aprovou todas as ordens dentro do risco
    assert all(r["approved"] for r in st["reviews"])
    assert all(r["order"]["risk_usd"] <= C.RISK_PER_TRADE * C.CAPITAL * 3 for r in st["reviews"])


def test_sentinel_blocks_bad_orders():
    ctx = {"regime": "alta", "equity": 10_000, "stale": False, "halted": None}
    good = {"strategy_regime": "alta", "signal": True, "qty": 1, "risk_usd": 50, "notional": 3000,
            "stop_distance": 50, "reason": "x"}
    assert sentinel.review(good, ctx)["approved"]
    for bad in [{"risk_usd": 51}, {"notional": 4000}, {"reason": ""}, {"stop_distance": 0},
                {"strategy_regime": "baixa"}]:
        assert not sentinel.review({**good, **bad}, ctx)["approved"]
    assert not sentinel.review(good, {**ctx, "stale": True})["approved"]
    assert not sentinel.review(good, {**ctx, "halted": {"reason": "x"}})["approved"]


def test_kill_switches():
    a = {"equity": 8_400, "peak_equity": 10_000, "loss_streak": 0}
    assert "drawdown" in sentinel.kill_reason(a, False)
    assert "perdas" in sentinel.kill_reason({**a, "equity": 10_000, "loss_streak": 3}, False)
    assert "feed" in sentinel.kill_reason({**a, "equity": 10_000}, True)
    assert sentinel.kill_reason({**a, "equity": 9_000}, False) is None
