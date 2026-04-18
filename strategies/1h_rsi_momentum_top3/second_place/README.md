# 🥈 第二名（最推薦）

## 基本資訊

| 項目 | 內容 |
|------|------|
| 排名 | 🥈 第二名 |
| 原因 | Walk-Forward 穩定性最佳 |
| 研究者 | researcher_v11 |
| 驗證 | Overfitting 分析通過（低風險） |

## 進場邏輯

- Long: 1H bar 收盤漲 AND 15m bar 收盤 > 開盤 AND RSI < 85
- Short: 1H bar 收盤跌 AND 15m bar 收盤 < 開盤 AND RSI > 15

## 參數

| 參數 | 數值 |
|------|------|
| stop_loss_pct | 0.5% |
| take_profit_pct | 0.65% |
| max_holding_bars | 7 |

## 回測結果

| 指標 | 門檻 | 實際值 | 狀態 |
|------|------|--------|------|
| WR | ≥ 60% | 67.85% | ✅ |
| PF | ≥ 1.8 | 3.05 | ✅ |
| DD | ≤ 25% | 2.63% | ✅ |
| Sharpe | ≥ 1.5 | 72.83 | ✅ |
| 月交易 | ≥ 100 | 330.2 | ✅ |
| TP+SL | > 30% | 34.5% | ✅ |
| Long PF | > 1.0 | 3.01 | ✅ |
| Short PF | > 1.0 | 3.09 | ✅ |

## Monte Carlo 穩定性

| 指標 | 數值 |
|------|------|
| Sharpe | 72.72 ± 1.85 |
| Sharpe 5th | 69.63 |
| DD 95% VaR | 3.32% |

## Walk-Forward 分析

| 時段 | Sharpe | WR | PF |
|------|--------|-----|-----|
| 前半段 | 72.69 | 67.8% | 3.04 |
| 後半段 | 72.75 | 67.9% | 3.06 |
| 差異 | **0.06** | 0.1% | 0.02 |

## 使用方式

```python
from strategy import generate_signals
from backtest_core import BacktestEngine

params = {
    'stop_loss_pct': 0.005,
    'take_profit_pct': 0.0065,
    'max_holding_bars': 7
}

engine = BacktestEngine(df, params)
result = engine.run(generate_signals)
```
