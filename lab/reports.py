"""Scout (regime do dia), cartões de estratégia, resumo das 07:30 e relatório semanal."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import config as C
from . import data
from .engine import run as engine_run
from .indicators import adx, ema
from .paper import CHAMPS, STATE, load_json, save_json
from .research import Market
from .strategies import FAMILY_BY_KEY, ideas

TZ = ZoneInfo(C.TIMEZONE)
PROPOSED = C.STATE_DIR / "proposed_champions.json"


def today():
    return datetime.now(TZ).strftime("%Y-%m-%d")


def pct(x):
    return f"{x:+.1%}"


# ---------------- Scout ----------------
def scout() -> dict:
    out = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "markets": {}}
    for m in C.MARKETS:
        df = data.load(m)
        mk = Market.from_df(df.iloc[-1500:].reset_index(drop=True))
        e, a = ema(mk.c, C.EMA_LEN), adx(mk.h, mk.l, mk.c, C.ADX_LEN)
        out["markets"][m] = {
            "regime": C.REGIMES[int(mk.reg[-1])], "last_bar_utc": data.meta(m)["last_bar_utc"],
            "close": float(mk.c[-1]), "ema200": round(float(e[-1]), 2),
            "ema200_rising": bool(e[-1] > e[-1 - C.EMA_SLOPE_LOOKBACK]), "adx": round(float(a[-1]), 1),
            "source": data.meta(m)["source"]}
    out["ideas"] = [{"family": f.key, "market": m, "regime": C.REGIMES[f.regime], "idea": f.idea,
                     "why": f.why} for f, m in ideas()]
    save_json(C.REPORTS_DIR / "scout" / f"{today()}.json", out)
    return out


def regime_lines(sc) -> list[str]:
    lines = []
    for m, x in sc["markets"].items():
        pos = "acima" if x["close"] > x["ema200"] else "abaixo"
        slope = "subindo" if x["ema200_rising"] else "caindo"
        lines.append(f"{m}: **{x['regime']}**. Preço {x['close']:,.2f} {pos} da EMA 200 ({x['ema200']:,.2f}, "
                     f"{slope}), ADX {x['adx']} (candle {x['last_bar_utc']} UTC)")
    return lines


# ---------------- Cartões ----------------
GATE_NAMES = {"1_in_sample": "1. Dentro da amostra", "2_is_permutation": "2. 1.000 permutações (p<1%)",
              "3_walk_forward": "3. Walk-forward", "4_wf_permutation": "4. Permutações do walk-forward (p<5%)",
              "final_check": "Checagem final (dados nunca vistos)"}


def card(r) -> str:
    f = FAMILY_BY_KEY[r["family"]]
    g = r["gates"]
    p = r.get("params_is", {})
    side = "venda a descoberto" if f.regime == C.BEAR else "compra" if f.regime == C.BULL else "compra e venda"
    L = [f"### {f.key} · {r['market']} · regime de {r['regime']} · {r['status'].upper()}",
         f"- **Ideia:** {f.idea}",
         f"- **Por que a vantagem existe e quem paga:** {f.why}",
         f"- **Entrada:** {f.entry} Parâmetro escolhido no treino: {f.param_name}={p.get('p')}.",
         f"- **Saída:** {f.exit}",
         f"- **Stop:** {p.get('stop')} x ATR(14) do candle do sinal, fixo.",
         f"- **Tamanho:** risco de {C.RISK_PER_TRADE:.1%} do patrimônio até o stop, no máximo 1/3 do patrimônio "
         f"por mercado ({side}).",
         f"- **Dados:** {r['data']['source']}, baixados em {r['data']['fetched_at_utc']}. "
         f"Desenvolvimento {r['split']['dev'][0]} a {r['split']['dev'][1]}; reservado {r['split']['holdout'][0]} a "
         f"{r['split']['holdout'][1]} (UTC)."]
    w = g.get("3_walk_forward")
    if w:
        L.append(f"- **Walk-forward após custos:** {pct(w['net_return'])} vs segurar no mesmo regime "
                 f"{pct(w['buy_hold_in_regime'])} (segurar o tempo todo {pct(w['buy_hold_full'])}); "
                 f"drawdown máx. {w['max_dd']:.1%}; {w['trades']} trades; pior mês {w['worst_month'][0]} "
                 f"({pct(w['worst_month'][1])}); maior sequência de perdas {w['longest_losing_streak']}.")
    L.append("- **Portões:**")
    for k, name in GATE_NAMES.items():
        if k == "final_check":
            fc = r.get("final_check")
            if fc:
                h = fc["holdout"]
                bad = [c for c, ok in fc["checks"].items() if not ok]
                L.append(f"  - {name}: {'passou' if not bad else 'REPROVOU (' + ', '.join(bad) + ')'}; "
                         f"no período reservado {pct(h['net_return'])} vs régua {pct(h['hurdle'])}, "
                         f"{h['trades']} trades, drawdown {h['max_dd']:.1%}")
            continue
        x = g.get(k)
        if not x:
            L.append(f"  - {name}: não executado (morreu antes)")
        elif "p_value" in x:
            L.append(f"  - {name}: {'passou' if x['pass'] else 'REPROVOU'} · p={x['p_value']:.4f} "
                     f"({x['beaten_by']} de {x['permutations']} mercados embaralhados igualaram ou bateram)")
        else:
            L.append(f"  - {name}: {'passou' if x['pass'] else 'REPROVOU'} · {pct(x['net_return'])} após custos vs "
                     f"régua {pct(x['hurdle'])}, {x['trades']} trades, PF {x['profit_factor']:.2f}")
    if r.get("score"):
        L.append(f"- **Nota:** {r['score']} / 5 ({', '.join(f'{k} {v}' for k, v in r['score_parts'].items())})")
    L.append(f"- **Primeiro sinal de que está morrendo:** {f.warning}")
    L.append(f"- **POR QUE ISSO PODE FALHAR:** {f.fail}")
    return "\n".join(L)


def research_report(res, sc) -> str:
    rs = res["results"]
    L = [f"# Pesquisa noturna · {today()}", "",
         f"Gerado em {res['generated_at_utc']} UTC. Custos estimados de {C.COST_PER_SIDE:.2%} por lado "
         f"(taxa + slippage). Todos os números foram calculados pelo código em `lab/` sobre dados reais.", "",
         "## Regime de hoje", *[f"- {x}" for x in regime_lines(sc)], "", "## Placar"]
    for reg, b in res["board"].items():
        ch = b["champion"]
        L.append(f"- **{reg}:** campeão {_fmt(ch)}; desafiantes: "
                 f"{', '.join(_fmt(x) for x in b['challengers']) or 'nenhum'}")
    if not any(b["champion"] for b in res["board"].values()):
        L += ["", "> Nenhuma ideia sobreviveu aos 4 portões e à checagem final. O bot fica em caixa até "
                  "alguma passar. Isso é o sistema funcionando, não um defeito."]
    L += ["", "## Todas as ideias", "", "| Família | Mercado | Regime | Resultado | Morreu em |",
          "|---|---|---|---|---|"]
    for r in rs:
        L.append(f"| {r['family']} | {r['market']} | {r['regime']} | {r['status']} | "
                 f"{GATE_NAMES.get(r.get('killed_at'), '-')} |")
    deep = [r for r in rs if len(r["gates"]) >= 2 or r["status"] == "sobrevivente"]
    L += ["", "## Cartões (ideias que passaram do portão 1)", ""]
    L += [card(r) + "\n" for r in deep] or ["Nenhuma."]
    return "\n".join(L)


def _fmt(x):
    return f"{x['family']} em {x['market']} (nota {x['score']})" if x else "nenhum (caixa)"


def publish_champions(res) -> str:
    """Primeira noite: adota direto (paper). Depois: só propõe; muda com aprovação."""
    board = {"generated_at_utc": res["generated_at_utc"],
             "routing": {reg: b["routing"] for reg, b in res["board"].items()},
             "board": {reg: {k: v for k, v in b.items() if k != "routing"} for reg, b in res["board"].items()}}
    cur = load_json(CHAMPS)
    if cur is None:
        save_json(CHAMPS, {**board, "approved": "adoção inicial automática (paper)"})
        return "adotado"
    if cur.get("routing") == board["routing"]:
        PROPOSED.unlink(missing_ok=True)
        return "sem mudança"
    save_json(PROPOSED, board)
    return "proposto (aguarda aprovação)"


def approve():
    prop = load_json(PROPOSED)
    if not prop:
        return "nada para aprovar"
    save_json(CHAMPS, {**prop, "approved": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    PROPOSED.unlink()
    return "aprovado"


# ---------------- Desafiantes em sombra ----------------
def shadow(since_ms: int) -> dict:
    """Roda campeões e desafiantes no período de paper (exposição unitária) e conta dias em que o desafiante
    estava à frente do campeão, em sequência."""
    champs = load_json(CHAMPS) or {}
    out = {}
    for reg, per_m in champs.get("routing", {}).items():
        for m, x in per_m.items():
            if not x or not x.get("champion"):
                continue
            df = data.load(m)
            mk = Market.from_df(df.iloc[-2500:].reset_index(drop=True))
            start = int(np.searchsorted(mk.t, since_ms))
            if start >= mk.c.size - 1:
                continue
            curves = {}
            for role, s in [("champion", x["champion"])] + [(f"challenger{i+1}", c) for i, c in
                                                             enumerate(x["challengers"])]:
                fam = FAMILY_BY_KEY[s["family"]]
                le, lx, se, sx = fam.signals(mk.o, mk.h, mk.l, mk.c, s["params"]["p"])
                r = engine_run(mk.o, mk.h, mk.l, mk.c, mk.at, mk.reg, fam.regime, le, lx, se, sx,
                               float(s["params"]["stop"]), C.MAX_HOLD_BARS, C.COST_PER_SIDE,
                               C.SHORT_FUNDING_PER_BAR, start, mk.c.size)[0]
                eq = np.cumprod(1 + r[start:])[C.BARS_PER_DAY - 1::C.BARS_PER_DAY]
                curves[role] = {"strategy": s, "daily_equity": eq.round(5).tolist(),
                                "total": float(eq[-1] - 1) if eq.size else 0.0}
            for role, cv in curves.items():
                if role.startswith("challenger"):
                    ahead = np.array(cv["daily_equity"]) > np.array(curves["champion"]["daily_equity"])
                    run_len = 0
                    for a in ahead[::-1]:
                        if not a:
                            break
                        run_len += 1
                    cv["days_ahead_in_a_row"] = run_len
                    cv["promotion_candidate"] = run_len >= C.CHALLENGER_BETTER_DAYS
            out[f"{reg}/{m}"] = curves
    return out


# ---------------- Resumo das 07:30 ----------------
def brief() -> str:
    sc = load_json(C.REPORTS_DIR / "scout" / f"{today()}.json") or scout()
    champs = load_json(CHAMPS) or {}
    st = load_json(STATE) or {}
    now = datetime.now(timezone.utc)
    cut = (now - timedelta(hours=24)).isoformat(timespec="minutes")
    trades = [t for t in st.get("trades", []) if t["exit_time"] >= cut]
    reviews = [r for r in st.get("reviews", []) if r["time"] >= cut]
    pnl = sum(t["pnl_usd"] for t in trades)
    acct = st.get("account", {})
    L = [f"# Resumo da manhã · {today()}", "", "**Regime de hoje**", *regime_lines(sc)[:3], "",
         "**Campeões e desafiantes**"]
    for reg, b in champs.get("board", {}).items():
        L.append(f"- {reg}: campeão {_fmt(b['champion'])}; desafiantes "
                 f"{', '.join(_fmt(x) for x in b['challengers']) or 'nenhum'}")
    if PROPOSED.exists():
        L.append("- ⚠ Há um novo placar proposto pelo Judge aguardando sua aprovação "
                 "(`python -m lab approve`). Nada muda sem você.")
    sh = shadow(_ms(st.get("started_at_utc"))) if st else {}
    for k, cv in sh.items():
        for role, x in cv.items():
            if x.get("promotion_candidate"):
                L.append(f"- ⚠ {k}: {x['strategy']['family']} está {x['days_ahead_in_a_row']} dias à frente do "
                         f"campeão em paper. Candidato a promoção (precisa da sua aprovação).")
    L += ["", "**Últimas 24 horas (paper, fills simulados)**",
          f"- Trades fechados: {len(trades)} · P&L: US$ {pnl:+,.2f} · Patrimônio: US$ "
          f"{acct.get('equity', C.CAPITAL):,.2f} (pico US$ {acct.get('peak_equity', C.CAPITAL):,.2f})",
          f"- Ordens revisadas pelo Sentinel: {len(reviews)} · bloqueadas: "
          f"{sum(not r['approved'] for r in reviews)}"]
    for t in trades:
        L.append(f"  - {t['market']} {t['family']} {'compra' if t['side'] == 1 else 'venda'} "
                 f"{t['entry']:,.2f} → {t['exit']:,.2f} ({t['exit_reason']}): US$ {t['pnl_usd']:+,.2f}")
    wrong, fixes = [], []
    if st.get("halted"):
        wrong.append(f"Bot PARADO pelo Sentinel: {st['halted']['reason']} ({st['halted']['at_utc']} UTC).")
        fixes.append("Revisar a causa e, se estiver tudo certo, retomar com `python -m lab resume`.")
    losers = [t for t in trades if t["pnl_usd"] < 0]
    if losers:
        stops = sum(t["exit_reason"] == "stop" for t in losers)
        wrong.append(f"{len(losers)} trade(s) perdedor(es), {stops} por stop.")
        if stops >= 2:
            fixes.append("Stops em sequência: conferir no relatório semanal se o regime está mudando.")
    blocked = [r for r in reviews if not r["approved"]]
    if blocked:
        why = sorted({k for r in blocked for k, ok in r["checks"].items() if not ok})
        wrong.append(f"{len(blocked)} ordem(ns) bloqueada(s) pelo Sentinel ({', '.join(why)}).")
    if not any((b or {}).get("champion") for b in champs.get("board", {}).values()):
        wrong.append("Nenhuma estratégia passou nos 4 portões: o bot está em caixa.")
        fixes.append("Manter em caixa. O Judge reavalia toda noite com dados novos; não afrouxe os portões "
                     "para forçar um campeão.")
    L += ["", "**Onde deu errado e até 3 correções**"]
    L += [f"- {w}" for w in wrong] or ["- Nada a apontar."]
    L += [f"{i+1}. {f}" for i, f in enumerate(fixes[:3])]
    L += ["", "_Nada muda sem a sua aprovação. Dinheiro real só depois do seu \"go live\"._"]
    txt = "\n".join(L)
    (C.REPORTS_DIR / "brief").mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "brief" / f"{today()}.md").write_text(txt)
    return txt


# ---------------- Semanal ----------------
def weekly() -> str:
    from .judge import evaluate
    st = load_json(STATE) or {}
    champs = load_json(CHAMPS) or {}
    L = [f"# Relatório semanal · {today()}", ""]
    sh = shadow(_ms(st.get("started_at_utc"))) if st else {}
    L.append("## Campeões x desafiantes (paper em sombra, exposição unitária)")
    if not sh:
        L.append("- Sem campeões ativos: nada a comparar.")
    for k, cv in sh.items():
        for role, x in cv.items():
            L.append(f"- {k} · {role}: {x['strategy']['family']} {pct(x['total'])}"
                     + (f" · {x['days_ahead_in_a_row']} dias seguidos à frente" if role != "champion" else ""))
    L += ["", "## Paper x backtest"]
    trades = st.get("trades", [])
    if trades:
        avg = float(np.mean([t["ret"] for t in trades]))
        L.append(f"- Paper: {len(trades)} trades, média {pct(avg)} por trade.")
    else:
        L.append("- Paper: nenhum trade fechado ainda.")
    for reg, b in champs.get("board", {}).items():
        if b.get("champion"):
            L.append(f"- Backtest walk-forward do campeão de {reg}: ver `reports/research/` do dia da coroação.")
    L += ["", "## Reteste com dados frescos (walk-forward + permutações)"]
    queue = []
    seen = set()
    for reg, b in champs.get("board", {}).items():
        for s in [b.get("champion")] + list(b.get("challengers", [])):
            if not s or (s["family"], s["market"]) in seen:
                continue
            seen.add((s["family"], s["market"]))
            r = evaluate((s["family"], s["market"], C.IS_PERMUTATIONS, C.WF_PERMUTATIONS))
            ok = r["status"] == "sobrevivente"
            L.append(f"- {s['family']} {s['market']}: {'segue de pé' if ok else 'FALHOU em ' + r.get('killed_at', '?')}")
            if not ok:
                queue.append({"family": s["family"], "market": s["market"], "killed_at": r.get("killed_at")})
    if not seen:
        L.append("- Nenhum campeão ou desafiante para retestar.")
    save_json(C.STATE_DIR / "judge_queue.json", {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                                   "failures": queue})
    L += ["", "## Vantagens enfraquecendo",
          *([f"- {q['family']} {q['market']} (reprovou em {q['killed_at']}); devolvido ao Judge."
             for q in queue] or ["- Nenhuma."])]
    txt = "\n".join(L)
    (C.REPORTS_DIR / "weekly").mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "weekly" / f"{today()}.md").write_text(txt)
    return txt


def _ms(iso_s):
    if not iso_s:
        return 0
    return int(datetime.fromisoformat(iso_s).timestamp() * 1000)
