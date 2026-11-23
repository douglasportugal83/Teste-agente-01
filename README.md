# Laboratório de trading noturno (paper)

Bot de pesquisa e paper trading para **BTC, ETH e SOL em candles de 4h**, com capital simulado de
**US$ 10.000** e risco de **0,5% por trade**. Não existe conexão com corretora: nenhuma ordem real é enviada.
Dinheiro real só depois do seu "go live", e mesmo assim exige construir e revisar essa integração.

## A equipe (agendada no GitHub Actions, horário de Brasília)

| Agente | Quando | O que faz | Comando |
|---|---|---|---|
| Scout | 21:00 | Atualiza dados, rotula o regime de cada mercado e lista as 21 ideias | `python -m lab scout` |
| Judge | 23:00 | Backtesta cada ideia, aplica os 4 portões e mata o resto | `python -m lab judge` |
| Engineer | 03:00 | Publica o placar de campeões; mudanças ficam *propostas* até você aprovar | `python -m lab engineer` |
| Sentinel | toda hora | Paper trading, revisão pré-trade de cada ordem, travas de emergência | `python -m lab sentinel` |
| Resumo | 07:30 | Abre uma issue com regime, placar, últimas 24h e até 3 correções | `python -m lab brief` |
| Semanal | domingo 10:00 | Campeões x desafiantes, paper x backtest, reteste com dados frescos | `python -m lab weekly` |

Alertas e resumos chegam como **issues** neste repositório, e o GitHub te avisa por e-mail.
Para aprovar um placar proposto ou retomar o bot depois de uma trava: **Actions → "Laboratório de trading"
→ Run workflow** e digite `approve` ou `resume`.

## Regras de regime

- **Lateral:** ADX(14) abaixo de 20. Tem prioridade.
- **Alta:** preço acima da EMA 200, com a EMA subindo (vs 10 barras atrás).
- **Baixa:** preço abaixo da EMA 200, com a EMA caindo.
- **Neutro:** sinais conflitantes. Fica em caixa.
- O regime só troca depois de se manter **3 candles** seguidos.

## Os 4 portões (todo número é calculado pelo código, sobre dados reais com fonte e timestamp)

Os últimos 25% dos dados ficam **reservados** e nunca são usados para escolher vencedores, só para vetar.

1. **Dentro da amostra:** lucro após custos, fator de lucro > 1, 30+ trades e bater "segurar o ativo durante o
   mesmo regime" (para estratégias vendidas, a régua mínima é o caixa).
2. **1.000 permutações** do mercado (re-otimizando em cada uma), com p < 1%.
3. **Walk-forward:** treino de 1 ano e teste de 3 meses, re-otimizando a cada janela; precisa bater a mesma régua.
4. **Permutações do walk-forward:** 200 rodadas, p < 5%. O prompt não definiu esses valores; são suposições nossas.

**Checagem final (veto):** ganhar no período reservado, bater segurar após custos, funcionar em ≥ 50% das
combinações de parâmetros, não ter vazamento do futuro e passar nos testes de mercado embaralhado.

A nota de 1 a 5 combina: lucro walk-forward acima da régua, drawdown, número de trades, maior sequência
de perdas e estabilidade entre janelas. Um desafiante só vira candidato após 7 dias seguidos à frente em
paper, e a troca precisa da sua aprovação.

## Sentinel

- Antes de cada ordem, checa regime, sinal, tamanho (≤ 0,5% de risco e ≤ 1/3 do patrimônio), stop, razão e
  frescor do feed.
- Para tudo e abre uma issue com: **15% de drawdown** desde o pico, **3 perdas seguidas** ou **feed parado**.

## Estimativas (rotuladas)

Custos de 0,10% de taxa + 0,05% de slippage por lado. Funding de 0,01% a cada 8h em posições vendidas. Fills
simulados na abertura do candle seguinte ao sinal. Vendas a descoberto assumem perpétuos; o funding real varia.

## Rodar localmente

```bash
pip install -r requirements.txt
python -m pytest -q          # testes do motor, da consistência paper/backtest e do Sentinel
python -m lab nightly        # Scout + Judge + Engineer
python -m lab sentinel && python -m lab brief
```

Ajustes ficam em `lab/config.py`. A biblioteca de ideias fica em `lab/strategies.py`.
