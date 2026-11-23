"""Indicadores causais (cada valor em t usa só dados até t). Compilados com numba."""
import numpy as np
from numba import njit


@njit(cache=True)
def ema(x, n):
    out = np.full(x.size, np.nan)
    if x.size < n:
        return out
    a = 2.0 / (n + 1.0)
    out[n - 1] = x[:n].mean()
    for i in range(n, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


@njit(cache=True)
def _wilder(x, n, start):
    out = np.full(x.size, np.nan)
    if x.size < start + n:
        return out
    s = 0.0
    for i in range(start, start + n):
        s += x[i]
    out[start + n - 1] = s / n
    for i in range(start + n, x.size):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


@njit(cache=True)
def true_range(h, l, c):
    tr = np.empty(c.size)
    tr[0] = h[0] - l[0]
    for i in range(1, c.size):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return tr


@njit(cache=True)
def atr(h, l, c, n):
    return _wilder(true_range(h, l, c), n, 1)


@njit(cache=True)
def adx(h, l, c, n):
    size = c.size
    pdm = np.zeros(size)
    ndm = np.zeros(size)
    for i in range(1, size):
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        if up > dn and up > 0:
            pdm[i] = up
        if dn > up and dn > 0:
            ndm[i] = dn
    tr = _wilder(true_range(h, l, c), n, 1)
    pdi = _wilder(pdm, n, 1)
    ndi = _wilder(ndm, n, 1)
    dx = np.full(size, np.nan)
    for i in range(size):
        if tr[i] > 0 and not np.isnan(tr[i]):
            p = 100 * pdi[i] / tr[i]
            m = 100 * ndi[i] / tr[i]
            if p + m > 0:
                dx[i] = 100 * abs(p - m) / (p + m)
    first = n  # primeiro dx válido
    return _wilder(np.where(np.isnan(dx), 0.0, dx), n, first)


@njit(cache=True)
def rsi(c, n):
    size = c.size
    up = np.zeros(size)
    dn = np.zeros(size)
    for i in range(1, size):
        d = c[i] - c[i - 1]
        if d > 0:
            up[i] = d
        else:
            dn[i] = -d
    au = _wilder(up, n, 1)
    ad = _wilder(dn, n, 1)
    out = np.full(size, np.nan)
    for i in range(size):
        if not np.isnan(au[i]):
            out[i] = 100.0 if ad[i] == 0 else 100 - 100 / (1 + au[i] / ad[i])
    return out


@njit(cache=True)
def rolling_max(x, n):
    out = np.full(x.size, np.nan)
    for i in range(n - 1, x.size):
        out[i] = x[i - n + 1:i + 1].max()
    return out


@njit(cache=True)
def rolling_min(x, n):
    out = np.full(x.size, np.nan)
    for i in range(n - 1, x.size):
        out[i] = x[i - n + 1:i + 1].min()
    return out


@njit(cache=True)
def sma(x, n):
    out = np.full(x.size, np.nan)
    s = 0.0
    for i in range(x.size):
        s += x[i]
        if i >= n:
            s -= x[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


@njit(cache=True)
def rolling_std(x, n):
    out = np.full(x.size, np.nan)
    for i in range(n - 1, x.size):
        out[i] = x[i - n + 1:i + 1].std()
    return out


def shift1(x):
    """Valor da barra anterior (para comparar com máximas/mínimas que excluem a barra atual)."""
    out = np.empty_like(x)
    out[0] = np.nan
    out[1:] = x[:-1]
    return out
