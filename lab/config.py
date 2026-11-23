"""Configuração central do laboratório. Todo número ajustável mora aqui."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"
REPORTS_DIR = ROOT / "reports"

# --- Mercados (padrões adotados; o usuário não escolheu) ---
MARKETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "4h"
BAR_MS = 4 * 60 * 60 * 1000
BARS_PER_DAY = 6
BARS_PER_YEAR = 6 * 365
YEARS_OF_DATA = 4
DATA_SOURCE = "Binance spot klines (https://data-api.binance.vision/api/v3/klines)"

# --- Capital e risco (paper) ---
CAPITAL = 10_000.0
RISK_PER_TRADE = 0.005          # 0,5% do patrimônio por trade
MAX_NOTIONAL_PER_MARKET = 1 / 3  # sem alavancagem: no máximo 1/3 do patrimônio por mercado
MAX_HOLD_BARS = 20 * BARS_PER_DAY  # holding swing: até 20 dias

# --- Custos (estimativas, rotuladas como tal) ---
FEE_PER_SIDE = 0.0010       # 0,10% taxa taker por lado (estimativa)
SLIPPAGE_PER_SIDE = 0.0005  # 0,05% de slippage por lado (estimativa)
COST_PER_SIDE = FEE_PER_SIDE + SLIPPAGE_PER_SIDE
SHORT_FUNDING_PER_BAR = 0.0001 / 2  # ~0,01% a cada 8h em perpétuos (estimativa)

# --- Regime ---
EMA_LEN = 200
EMA_SLOPE_LOOKBACK = 10
ADX_LEN = 14
ADX_FLAT = 20.0
REGIME_CONFIRM_BARS = 3
REGIMES = {0: "neutro", 1: "alta", 2: "lateral", 3: "baixa"}
BULL, FLAT, BEAR = 1, 2, 3

# --- Portões do Judge ---
HOLDOUT_FRACTION = 0.25          # último trecho: nunca usado para escolher vencedores
IS_PERMUTATIONS = 1000
IS_P_MAX = 0.01
WF_TRAIN_BARS = BARS_PER_YEAR    # treino de 1 ano
WF_TEST_BARS = 90 * BARS_PER_DAY  # teste de 3 meses
WF_PERMUTATIONS = 200            # o prompt não definiu; 200 é suposição nossa
WF_P_MAX = 0.05                  # o prompt não definiu; 5% é suposição nossa
MIN_TRADES_IS = 30
MIN_ROBUST_SHARE = 0.5           # >= 50% das combinações de parâmetros lucrativas
RANDOM_SEED = 20261123

# --- Sentinel ---
KILL_DRAWDOWN = 0.15
KILL_LOSS_STREAK = 3
STALE_FEED_BARS = 2  # feed parado se o último candle fechado tiver mais de 2 barras de atraso

# --- Desafiantes ---
CHALLENGER_BETTER_DAYS = 7
TIMEZONE = "America/Sao_Paulo"
