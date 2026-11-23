"""Sentinel: revisão pré-trade de cada ordem e travas de emergência."""
from __future__ import annotations

from . import config as C


def review(order: dict, ctx: dict) -> dict:
    """Checa regime, sinal, tamanho, stop e razão. Qualquer falha bloqueia a ordem."""
    checks = {
        "regime": ctx["regime"] == order["strategy_regime"] and ctx["regime"] != "neutro",
        "sinal": bool(order.get("signal")),
        "tamanho": (order["qty"] > 0
                    and order["risk_usd"] <= C.RISK_PER_TRADE * ctx["equity"] * 1.001
                    and order["notional"] <= C.MAX_NOTIONAL_PER_MARKET * ctx["equity"] * 1.001),
        "stop": order["stop_distance"] > 0,
        "razao": bool(order.get("reason", "").strip()),
        "feed": not ctx["stale"],
        "sem_trava": ctx["halted"] is None,
    }
    return {"approved": all(checks.values()), "checks": checks}


def kill_reason(account: dict, stale: bool) -> str | None:
    dd = 1 - account["equity"] / account["peak_equity"]
    if dd >= C.KILL_DRAWDOWN:
        return f"drawdown de {dd:.1%} desde o pico (limite {C.KILL_DRAWDOWN:.0%})"
    if account["loss_streak"] >= C.KILL_LOSS_STREAK:
        return f"{account['loss_streak']} perdas seguidas (limite {C.KILL_LOSS_STREAK})"
    if stale:
        return "feed de dados parado"
    return None
