# 🥉 第三名（均衡之選）

## 基本資訊

| 項目 | 內容 |
|------|------|
| 排名 | 🥉 第三名 |
| 原因 | Sharpe 與穩定性平衡良好 |
| 研究者 | researcher_v11 |
| 驗證 | Overfitting 分析通過（低風險） |

## 進場邏輯

- Long: 1H bar 收盤漲 AND 15m bar 收盤 > 開盤 AND RSI < 85
- Short: 1H bar 收盤跌 AND 15m bar 收盤 < 開盤 AND RSI > 15

## 參數

| 參數 | 數值 |
|------|------|
| stop_loss_pct | 0.4% |
| take_profit_pct | 0.6% |
| max_holding_bars | 7 |

## 回測結果

| 指標 | 門檻 | 實際值 | 狀態 |
|------|------|--------|------|
| WR | ≥ 60% | 66.76% | ✅ |
| PF | ≥ 1.8 | 2.95 | ✅ |
| DD | ≤ 25% | 3.21% | ✅ |
| Sharpe | ≥ 1.5 | 72.35 | ✅ |
| 月交易 | ≥ 100 | 340.5 | ✅ |
| TP+SL | > 30% | 41.3% | ✅ |
| Long PF | > 1.0 | 2.98 | ✅ |
| Short PF | > 1.0 | 2.92 | ✅ |

## Monte Carlo 穩定性

| 指標 | 數值 |
|------|------|
| Sharpe | 72.35 ± 1.91 |
| Sharpe 5th | 68.95 |
| DD 95% VaR | 4.01% |

## Walk-Forward 分析

| 時段 | Sharpe | WR | PF |
|------|--------|-----|-----|
| 前半段 | 72.30 | 66.5% | 2.93 |
| 後半段 | 72.40 | 67.0% | 2.97 |
| 差異 | **0.05** | 0.5% | 0.04 |

## 使用方式

```python
from strategy import generate_signals
from backtest_core import BacktestEngine

params = {
    'stop_loss_pct': 0.004,
    'take_profit_pct': 0.006,
    'max_holding_bars': 7
}

engine = BacktestEngine(df, params)
result = engine.run(generate_signals)
```
