# 🥇 第一名（Sharpe 最高）

## 基本資訊

| 項目 | 內容 |
|------|------|
| 排名 | 🥇 第一名 |
| 原因 | Sharpe 最高 |
| 研究者 | researcher_v11 |
| 驗證 | Overfitting 分析通過（低風險） |

## 進場邏輯

- Long: 1H bar 收盤漲 AND 15m bar 收盤 > 開盤 AND RSI < 85
- Short: 1H bar 收盤跌 AND 15m bar 收盤 < 開盤 AND RSI > 15

## 參數

| 參數 | 數值 |
|------|------|
| stop_loss_pct | 0.4% |
| take_profit_pct | 0.4% |
| max_holding_bars | 5 |

## 回測結果

| 指標 | 門檻 | 實際值 | 狀態 |
|------|------|--------|------|
| WR | ≥ 60% | 71.60% | ✅ |
| PF | ≥ 1.8 | 3.04 | ✅ |
| DD | ≤ 25% | 2.31% | ✅ |
| Sharpe | ≥ 1.5 | 76.02 | ✅ |
| 月交易 | ≥ 100 | 419.3 | ✅ |
| TP+SL | > 30% | 44.7% | ✅ |
| Long PF | > 1.0 | 3.13 | ✅ |
| Short PF | > 1.0 | 2.94 | ✅ |

## Monte Carlo 穩定性

| 指標 | 數值 |
|------|------|
| Sharpe | 76.02 ± 2.01 |
| Sharpe 5th | 72.44 |
| DD 95% VaR | 2.91% |

## Walk-Forward 分析

| 時段 | Sharpe | WR | PF |
|------|--------|-----|-----|
| 前半段 | 74.03 | 71.2% | 2.98 |
| 後半段 | 78.00 | 72.0% | 3.10 |
| 差異 | **3.97** | 0.8% | 0.12 |

## 使用方式

```python
from strategy import generate_signals
from backtest_core import BacktestEngine

params = {
    'stop_loss_pct': 0.004,
    'take_profit_pct': 0.004,
    'max_holding_bars': 5
}

engine = BacktestEngine(df, params)
result = engine.run(generate_signals)
```
