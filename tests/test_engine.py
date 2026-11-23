import numpy as np
import pytest

from lab import config as C
from lab.engine import permute_bars, regime_bh, run
from lab.research import Market, lookahead_check
from lab.strategies import FAMILIES


def synth(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    o = np.concatenate([[c[0]], c[:-1]]) * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    t = np.arange(n, dtype=np.int64) * C.BAR_MS + 1_600_000_000_000
    return Market.from_arrays(o, h, l, c, t)


def test_always_long_matches_regime_hold_minus_costs():
    mk = synth()
    n = mk.c.size
    reg = np.ones(n, np.int8)  # regime de alta o tempo todo
    yes, no = np.ones(n, np.bool_), np.zeros(n, np.bool_)
    r, e, x, s, tr = run(mk.o, mk.h, mk.l, mk.c, mk.at, reg, 1, yes, no, no, no,
                         1e9, 10**9, 0.001, 0.0, 300, n)
    assert len(tr) == 1
    got = np.prod(1 + r[300:n])
    # entra na abertura de 301 e sai no fechamento final
    expect = (mk.c[-1] / mk.o[301]) * (1 - 0.001) ** 2
    assert got == pytest.approx(expect, rel=1e-3)
    assert regime_bh(mk.c, reg, 1, 300, n) == pytest.approx(np.log(mk.c[-1] / mk.c[300]))


def test_stop_is_respected_and_no_trades_outside_regime():
    mk = synth(seed=3)
    n = mk.c.size
    reg = np.zeros(n, np.int8)
    reg[1000:1500] = 1
    yes, no = np.ones(n, np.bool_), np.zeros(n, np.bool_)
    r, e, x, s, tr = run(mk.o, mk.h, mk.l, mk.c, mk.at, reg, 1, yes, no, no, no,
                         1.0, 10**9, 0.0, 0.0, 300, n)
    assert len(tr) > 0
    assert (e >= 1001).all() and (e <= 1500).all()
    assert (x <= 1501).all()
    assert (r[:1001] == 0).all() and (r[1502:] == 0).all()
    # perda por trade limitada ao stop (exceto gaps)
    assert tr.min() > -0.5


def test_permutation_preserves_return_distribution():
    mk = synth(seed=5)
    o, h, l, c = permute_bars(mk.o, mk.h, mk.l, mk.c, 500, 42)
    assert np.allclose(c[:501], mk.c[:501])
    a = np.sort(np.log(mk.c[501:] / mk.o[501:]))
    b = np.sort(np.log(c[501:] / o[501:]))
    assert np.allclose(a, b)
    assert np.isclose(np.log(c[-1] / c[500]), np.log(mk.c[-1] / mk.c[500]))
    assert not np.allclose(c, mk.c)
    assert (h >= np.maximum(o, c) - 1e-9).all() and (l <= np.minimum(o, c) + 1e-9).all()


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.key)
def test_no_lookahead(fam):
    mk = synth(seed=7)
    for p in fam.grid():
        assert lookahead_check(fam, mk, p, samples=10)
