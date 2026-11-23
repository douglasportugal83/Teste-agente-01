"""Biblioteca de ideias do Scout. Cada família tem uma razão econômica explicável em uma frase.

Sinais são avaliados no FECHAMENTO da barra t e executados na ABERTURA de t+1.
Cada função devolve (entrada_long, saída_long, entrada_short, saída_short).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from . import config as C
from .indicators import ema, rolling_max, rolling_min, rolling_std, rsi, shift1, sma

F = np.zeros  # atalho para "sem sinal"


def donchian_breakout(o, h, l, c, n):
    hi, lo = shift1(rolling_max(h, n)), shift1(rolling_min(l, max(n // 2, 2)))
    return c > hi, c < lo, F(c.size, bool), F(c.size, bool)


def ema_pullback(o, h, l, c, n):
    e = ema(c, n)
    pe, pc = shift1(e), shift1(c)
    entry = (pc < pe) & (c > e)            # volta para cima da EMA após recuo
    exit_ = c < shift1(rolling_min(l, 10))  # perde a mínima de 10 barras
    return entry, exit_, F(c.size, bool), F(c.size, bool)


def rsi_dip_bull(o, h, l, c, thr):
    r = rsi(c, 3)
    return r < thr, r > 70, F(c.size, bool), F(c.size, bool)


def bollinger_revert(o, h, l, c, k):
    m, s = sma(c, 20), rolling_std(c, 20)
    up, dn = m + k * s, m - k * s
    return c < dn, c > m, c > up, c < m


def rsi_range(o, h, l, c, thr):
    r = rsi(c, 14)
    return r < thr, r > 50, r > 100 - thr, r < 50


def donchian_breakdown(o, h, l, c, n):
    lo, hi = shift1(rolling_min(l, n)), shift1(rolling_max(h, max(n // 2, 2)))
    return F(c.size, bool), F(c.size, bool), c < lo, c > hi


def bear_rally_fade(o, h, l, c, thr):
    r = rsi(c, 3)
    return F(c.size, bool), F(c.size, bool), r > thr, r < 30


@dataclass
class Family:
    key: str
    regime: int
    fn: callable
    param_name: str
    param_grid: list
    idea: str
    why: str
    entry: str
    exit: str
    warning: str
    fail: str
    stop_grid: list = field(default_factory=lambda: [2.0, 3.0])

    def grid(self):
        return [{"p": p, "stop": s} for p, s in product(self.param_grid, self.stop_grid)]

    def signals(self, o, h, l, c, p):
        sig = self.fn(o, h, l, c, p)
        return tuple(np.nan_to_num(np.asarray(x, dtype=float)).astype(np.bool_) for x in sig)


FAMILIES = [
    Family("donchian_breakout", C.BULL, donchian_breakout, "N", [20, 40, 60],
           "Comprar rompimento da máxima de N barras em mercado de alta.",
           "Momentum de série temporal: o capital lento chega atrasado à tendência; paga quem vende cedo e quem entra tarde.",
           "Fechamento acima da máxima das N barras anteriores, com regime de alta confirmado.",
           "Fechamento abaixo da mínima de N/2 barras, troca de regime ou 20 dias.",
           "Rompimentos falsos seguidos (3+ stops em sequência) e ADX caindo abaixo de 25.",
           "Em mercados que andam de lado dentro da alta, rompimentos viram armadilhas e as taxas corroem o lucro."),
    Family("ema_pullback", C.BULL, ema_pullback, "EMA", [10, 20, 30],
           "Comprar a retomada da EMA curta depois de um recuo dentro da alta.",
           "Recuos em tendência são liquidação temporária; paga quem vende no pânico de curto prazo.",
           "Fechamento volta acima da EMA curta após fechar abaixo dela, com regime de alta.",
           "Fechamento abaixo da mínima de 10 barras, troca de regime ou 20 dias.",
           "Recuos que não voltam: preço passa a fechar abaixo da EMA 50 com frequência.",
           "Se a alta for frágil, o recuo vira reversão e a entrada compra a primeira perna de queda."),
    Family("rsi_dip_bull", C.BULL, rsi_dip_bull, "RSI3<", [10, 20, 30],
           "Comprar exageros de queda de curtíssimo prazo (RSI 3) dentro da alta.",
           "Sobre-reação de curto prazo: vendedores forçados (liquidações) entregam preço abaixo do justo.",
           "RSI(3) abaixo do limiar com regime de alta confirmado.",
           "RSI(3) acima de 70, troca de regime ou 20 dias.",
           "Média dos ganhos por trade encolhendo para perto do custo de ida e volta (0,3%).",
           "Ganhos pequenos por trade; um aumento de custos ou de slippage pode zerar a vantagem."),
    Family("bollinger_revert", C.FLAT, bollinger_revert, "k", [1.5, 2.0, 2.5],
           "Comprar na banda inferior e vender na superior em mercado lateral, sair na média.",
           "Em faixas, provedores de liquidez ganham com quem persegue movimentos que não continuam.",
           "Fechamento fora da banda de Bollinger (20, k): abaixo compra, acima vende, com regime lateral.",
           "Fechamento cruza a média de 20, troca de regime ou 20 dias.",
           "Fechamentos fora da banda que continuam por 3+ barras (faixa rompendo).",
           "Quando a lateralização termina num rompimento forte, o trade fica do lado errado do novo movimento."),
    Family("rsi_range", C.FLAT, rsi_range, "RSI14<", [25, 30, 35],
           "Operar extremos do RSI(14) de volta para 50 em mercado lateral.",
           "Excesso de curto prazo em faixas tende a reverter; paga quem opera rompimentos que falham.",
           "RSI(14) abaixo do limiar compra; acima de 100-limiar vende; regime lateral.",
           "RSI(14) cruza 50, troca de regime ou 20 dias.",
           "RSI ficando em extremo por muitas barras sem reverter.",
           "Poucos sinais por regime lateral: a amostra pequena pode esconder sorte."),
    Family("donchian_breakdown", C.BEAR, donchian_breakdown, "N", [20, 40, 60],
           "Vender a descoberto o rompimento da mínima de N barras em mercado de baixa.",
           "Quedas aceleram por liquidações e stops em cascata; paga quem compra cedo demais.",
           "Fechamento abaixo da mínima das N barras anteriores com regime de baixa confirmado.",
           "Fechamento acima da máxima de N/2 barras, troca de regime ou 20 dias.",
           "Short squeezes frequentes: stops atingidos por altas rápidas.",
           "Cripto tem altas violentas em baixa (short squeeze); custo de funding não modelado com precisão."),
    Family("bear_rally_fade", C.BEAR, bear_rally_fade, "RSI3>", [70, 80, 90],
           "Vender repiques exagerados (RSI 3 alto) dentro da baixa.",
           "Repiques em baixa costumam ser cobertura de vendidos, não compra nova; paga quem persegue o repique.",
           "RSI(3) acima do limiar com regime de baixa confirmado.",
           "RSI(3) abaixo de 30, troca de regime ou 20 dias.",
           "Repiques que viram tendência: preço volta acima da EMA 200.",
           "Fundo de mercado: o repique que parece exagero é o início da virada."),
]

FAMILY_BY_KEY = {f.key: f for f in FAMILIES}


def ideas():
    """20+ ideias = família x mercado."""
    return [(f, m) for f in FAMILIES for m in C.MARKETS]
