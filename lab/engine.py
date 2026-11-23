"""Motor de backtest com exposição unitária (100% do capital do teste por trade).

Regras idênticas às do paper trading:
- sinal no fechamento de t, execução na abertura de t+1;
- stop fixo = entrada -/+ stop_mult x ATR(14) no fechamento do sinal, checado na máxima/mínima;
- se a barra abre além do stop, sai na abertura (gap);
- sai na abertura seguinte se: sinal de saída, regime mudou ou atingiu o holding máximo;
- custos por lado em toda entrada e saída; funding estimado em posições vendidas.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def run(o, h, l, c, at, reg, target, le, lx, se, sx, stop_mult, max_hold,
        cost, funding, start, end):
    n = c.size
    r = np.zeros(n)
    t_entry = np.empty(n, np.int64)
    t_exit = np.empty(n, np.int64)
    t_side = np.empty(n, np.int64)
    t_ret = np.empty(n)
    nt = 0
    pos = 0
    entry = 0.0
    stop = 0.0
    held = 0
    ei = 0
    pending = 0  # +1/-1 entrada, 2 = sair
    for i in range(start, end):
        # 1) ordem pendente na abertura
        if pending == 2 and pos != 0:
            r[i] += pos * (o[i] / c[i - 1] - 1) - cost
            t_entry[nt] = ei; t_exit[nt] = i; t_side[nt] = pos
            t_ret[nt] = pos * (o[i] / entry - 1) - 2 * cost - (funding * held if pos < 0 else 0.0)
            nt += 1
            pos = 0
        elif (pending == 1 or pending == -1) and pos == 0:
            pos = pending
            entry = o[i]
            stop = entry - pos * stop_mult * at[i - 1]
            held = 0
            ei = i
            r[i] -= cost
        pending = 0
        if pos == 0:
            pass
        else:
            ref = entry if ei == i else c[i - 1]
            hit = (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop)
            if hit:
                fill = stop
                if (pos == 1 and o[i] < stop) or (pos == -1 and o[i] > stop):
                    fill = o[i]
                r[i] += pos * (fill / ref - 1) - cost
                t_entry[nt] = ei; t_exit[nt] = i; t_side[nt] = pos
                t_ret[nt] = pos * (fill / entry - 1) - 2 * cost - (funding * held if pos < 0 else 0.0)
                nt += 1
                pos = 0
            else:
                r[i] += pos * (c[i] / ref - 1)
                held += 1
                if pos < 0:
                    r[i] -= funding
        # 2) decisões no fechamento
        if i >= end - 1:
            break
        if pos != 0:
            if (reg[i] != target or held >= max_hold
                    or (pos == 1 and lx[i]) or (pos == -1 and sx[i])):
                pending = 2
        elif reg[i] == target and not np.isnan(at[i]):
            if le[i]:
                pending = 1
            elif se[i]:
                pending = -1
    # encerra no fim da janela (fechamento da última barra)
    if pos != 0:
        last = end - 1
        r[last] -= cost
        t_entry[nt] = ei; t_exit[nt] = last; t_side[nt] = pos
        t_ret[nt] = pos * (c[last] / entry - 1) - 2 * cost - (funding * held if pos < 0 else 0.0)
        nt += 1
    return r, t_entry[:nt], t_exit[:nt], t_side[:nt], t_ret[:nt]


@njit(cache=True)
def regime_bh(c, reg, target, start, end):
    """Retorno log de 'segurar o ativo' apenas nas barras em que o regime-alvo estava confirmado."""
    s = 0.0
    for i in range(start + 1, end):
        if reg[i - 1] == target:
            s += np.log(c[i] / c[i - 1])
    return s


@njit(cache=True)
def permute_bars(o, h, l, c, start, seed):
    """Permutação de barras (estilo Masters): embaralha gaps e corpos intrabarra a partir de `start`.
    Preserva a distribuição dos retornos e destrói a ordem temporal (onde mora qualquer vantagem real)."""
    np.random.seed(seed)
    n = c.size
    lo, lh, ll, lc = np.log(o), np.log(h), np.log(l), np.log(c)
    m = n - start - 1
    gap = np.empty(m); rh = np.empty(m); rl = np.empty(m); rc = np.empty(m)
    for k in range(m):
        i = start + 1 + k
        gap[k] = lo[i] - lc[i - 1]
        rh[k] = lh[i] - lo[i]; rl[k] = ll[i] - lo[i]; rc[k] = lc[i] - lo[i]
    p1 = np.random.permutation(m)
    p2 = np.random.permutation(m)
    no, nh, nl, nc = o.copy(), h.copy(), l.copy(), c.copy()
    prev = lc[start]
    for k in range(m):
        i = start + 1 + k
        op = prev + gap[p2[k]]
        j = p1[k]
        no[i] = np.exp(op); nh[i] = np.exp(op + rh[j]); nl[i] = np.exp(op + rl[j])
        prev = op + rc[j]
        nc[i] = np.exp(prev)
    return no, nh, nl, nc
