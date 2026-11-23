"""Judge: backtesta cada ideia, aplica os 4 portões e mata o resto."""
from __future__ import annotations

import json
import math
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np

from . import config as C
from . import data
from .engine import permute_bars
from .research import Market, Runner, benchmark, lookahead_check, metrics
from .strategies import FAMILY_BY_KEY, ideas

WARMUP = C.EMA_LEN + C.EMA_SLOPE_LOOKBACK + 50


def splits(n):
    dev_end = int(n * (1 - C.HOLDOUT_FRACTION))
    return WARMUP, dev_end, n


def _perm_test(fam, mk: Market, start, end, real, n_perm, p_max, mode, seed):
    """p-valor = (#permutações >= real + 1) / (N + 1). Para cedo quando já reprovou."""
    limit = math.floor(p_max * (n_perm + 1))  # contagem a partir da qual p >= p_max
    count = done = 0
    perm_from = start if mode == "is" else start + C.WF_TRAIN_BARS
    for k in range(n_perm):
        o, h, l, c = permute_bars(mk.o[:end], mk.h[:end], mk.l[:end], mk.c[:end], perm_from, seed + k)
        pm = Market.from_arrays(o, h, l, c, mk.t[:end])
        rn = Runner(fam, pm)
        if mode == "is":
            v = rn.optimize(start, end)[1]
        else:
            wf = rn.walk_forward(start, end)
            v = float(np.log1p(wf["rets"][wf["start"]:wf["end"]]).sum())
        done += 1
        if v >= real:
            count += 1
            if count >= limit:
                break
    p = (count + 1) / (done + 1)
    return {"p_value": p, "permutations": done, "beaten_by": count,
            "pass": count < limit and done == n_perm, "early_stop": done < n_perm}


def _score(m, bench_hurdle, folds):
    excess = m["net_return"] - bench_hurdle
    s_profit = 1 + sum(excess > x for x in (0.0, 0.10, 0.25, 0.50))
    s_dd = 1 + sum(m["max_dd"] < x for x in (0.40, 0.30, 0.20, 0.10))
    s_n = 1 + sum(m["trades"] >= x for x in (15, 30, 60, 100))
    s_streak = 1 + sum(m["longest_losing_streak"] <= x for x in (10, 7, 5, 3))
    pos = np.mean([f["ret"] > 0 for f in folds]) if folds else 0
    s_reg = 1 + sum(pos >= x for x in (0.35, 0.5, 0.65, 0.8))
    parts = {"lucro_wf": s_profit, "drawdown": s_dd, "n_trades": s_n,
             "seq_perdas": s_streak, "estabilidade_regime": s_reg}
    return round(float(np.mean(list(parts.values()))), 2), parts


def evaluate(args):
    fam_key, symbol, n_is, n_wf = args
    fam = FAMILY_BY_KEY[fam_key]
    df = data.load(symbol)
    mk = Market.from_df(df)
    start, dev_end, n = splits(mk.c.size)
    rn = Runner(fam, mk)
    out = {"family": fam_key, "market": symbol, "regime": C.REGIMES[fam.regime],
           "data": data.meta(symbol), "split": {
               "dev": [_iso(mk.t[start]), _iso(mk.t[dev_end - 1])],
               "holdout": [_iso(mk.t[dev_end]), _iso(mk.t[n - 1])]},
           "gates": {}, "status": "morta"}
    seed = C.RANDOM_SEED + zlib.crc32(f"{fam_key}|{symbol}".encode()) % 100_000

    # Portão 1: dentro da amostra
    best, best_v, vals = rn.optimize(start, dev_end)
    r, _, _, _, tr = rn.run(best, start, dev_end)
    m_is = metrics(r, tr, mk.t, start, dev_end)
    b_is = benchmark(mk, fam.regime, start, dev_end)
    robust = float(np.mean([v > 0 for v in vals]))
    g1 = (best_v > 0 and m_is["trades"] >= C.MIN_TRADES_IS and m_is["profit_factor"] > 1
          and m_is["net_return"] > b_is["hurdle"])
    out["params_is"] = best
    out["gates"]["1_in_sample"] = {"pass": bool(g1), **m_is, **b_is, "robust_share": robust}
    if not g1:
        out["killed_at"] = "1_in_sample"
        return out

    # Portão 2: 1.000 permutações dentro da amostra (re-otimizando em cada uma)
    g2 = _perm_test(fam, mk, start, dev_end, best_v, n_is, C.IS_P_MAX, "is", seed)
    out["gates"]["2_is_permutation"] = g2
    if not g2["pass"]:
        out["killed_at"] = "2_is_permutation"
        return out

    # Portão 3: walk-forward (re-otimiza a cada janela; só o trecho de teste conta)
    wf = rn.walk_forward(start, dev_end)
    m_wf = metrics(wf["rets"], wf["trades"], mk.t, wf["start"], wf["end"])
    b_wf = benchmark(mk, fam.regime, wf["start"], wf["end"])
    g3 = m_wf["net_return"] > b_wf["hurdle"] and m_wf["net_return"] > 0 and m_wf["trades"] > 0
    out["gates"]["3_walk_forward"] = {"pass": bool(g3), **m_wf, **b_wf, "folds": [
        {"test": [_iso(mk.t[f["test"][0]]), _iso(mk.t[f["test"][1] - 1])], "params": f["params"],
         "ret": f["ret"]} for f in wf["folds"]]}
    if not g3:
        out["killed_at"] = "3_walk_forward"
        return out

    # Portão 4: permutações do walk-forward
    wf_obj = float(np.log1p(wf["rets"][wf["start"]:wf["end"]]).sum())
    g4 = _perm_test(fam, mk, start, dev_end, wf_obj, n_wf, C.WF_P_MAX, "wf", seed + 7)
    out["gates"]["4_wf_permutation"] = g4
    if not g4["pass"]:
        out["killed_at"] = "4_wf_permutation"
        return out

    # Checagem final (veto apenas; nunca usada para ranquear)
    r_h, _, _, _, tr_h = rn.run(best, dev_end, n)
    m_h = metrics(r_h, tr_h, mk.t, dev_end, n)
    b_h = benchmark(mk, fam.regime, dev_end, n)
    no_leak = lookahead_check(fam, mk, best)
    checks = {
        "ganha_em_dados_nunca_vistos": m_h["net_return"] > 0,
        "bate_segurar_apos_custos": m_h["net_return"] > b_h["hurdle"],
        "robusto_entre_parametros": robust >= C.MIN_ROBUST_SHARE,
        "mercado_embaralhado_nao_bateu": True,  # portões 2 e 4 já aprovados
        "sem_vazamento_do_futuro": no_leak,
    }
    out["final_check"] = {"checks": checks, "holdout": {**m_h, **b_h}}
    if not all(checks.values()):
        out["killed_at"] = "final_check"
        return out
    out["score"], out["score_parts"] = _score(m_wf, b_wf["hurdle"], wf["folds"])
    out["status"] = "sobrevivente"
    return out


def judge(n_is=C.IS_PERMUTATIONS, n_wf=C.WF_PERMUTATIONS, workers=4):
    jobs = [(f.key, m, n_is, n_wf) for f, m in ideas()]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(evaluate, jobs))
    survivors = sorted([r for r in results if r["status"] == "sobrevivente"],
                       key=lambda r: (-r["score"], r["gates"]["3_walk_forward"]["max_dd"]))
    board = {}
    for reg in ("alta", "lateral", "baixa"):
        ranked = [r for r in survivors if r["regime"] == reg]
        routing = {}
        for m in C.MARKETS:
            mine = [r for r in ranked if r["market"] == m]
            routing[m] = {"champion": _pick(mine[0]) if mine else None,
                          "challengers": [_pick(x) for x in mine[1:3]]}
        board[reg] = {"champion": _pick(ranked[0]) if ranked else None,
                      "challengers": [_pick(x) for x in ranked[1:3]], "routing": routing}
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "settings": {"is_permutations": n_is, "wf_permutations": n_wf,
                         "costs_per_side": C.COST_PER_SIDE, "interval": C.INTERVAL},
            "results": results, "board": board}


def _pick(r):
    return {"family": r["family"], "market": r["market"], "params": r["params_is"], "score": r["score"]}


def _iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def save(res, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=_json))


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))
