"""Detector de regime: alta / lateral / baixa, com confirmação de N barras."""
import numpy as np
from numba import njit

from . import config as C
from .indicators import adx, ema


@njit(cache=True)
def _raw(c, e, a, slope_lb, adx_flat):
    out = np.zeros(c.size, np.int8)
    for i in range(slope_lb, c.size):
        if np.isnan(e[i - slope_lb]) or np.isnan(a[i]):
            continue
        if a[i] < adx_flat:
            out[i] = 2  # lateral: sem tendência
        elif c[i] > e[i] and e[i] > e[i - slope_lb]:
            out[i] = 1  # alta: preço acima de EMA 200 subindo
        elif c[i] < e[i] and e[i] < e[i - slope_lb]:
            out[i] = 3  # baixa: preço abaixo de EMA 200 caindo
        # senão neutro (0): sinais conflitantes -> caixa
    return out


@njit(cache=True)
def _confirm(raw, k):
    out = np.zeros(raw.size, np.int8)
    cur = 0
    run = 0
    for i in range(raw.size):
        run = run + 1 if i > 0 and raw[i] == raw[i - 1] else 1
        if run >= k:
            cur = raw[i]
        out[i] = cur
    return out


def regimes(h, l, c):
    """Retorna (bruto, confirmado). O confirmado só troca após o bruto se manter N barras."""
    e = ema(c, C.EMA_LEN)
    a = adx(h, l, c, C.ADX_LEN)
    raw = _raw(c, e, a, C.EMA_SLOPE_LOOKBACK, C.ADX_FLAT)
    return raw, _confirm(raw, C.REGIME_CONFIRM_BARS)
