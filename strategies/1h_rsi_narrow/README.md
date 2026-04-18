# 策略：1H Direction + RSI Momentum Filter（窄TP/SL版）

## 基本資訊

| 項目 | 內容 |
|------|------|
| 版本 | v1.0 |
| 創建日期 | 2026-04-18 |
| 研究者 | researcher_v8 |

## 進場邏輯

- Long: 1H bar 上漲 AND RSI < 85
- Short: 1H bar 下跌 AND RSI > 15

## 參數

| 參數 | 數值 |
|------|------|
| stop_loss_pct | 0.8% |
| take_profit_pct | 0.8% |
| max_holding_bars | 20 |

## 回測結果

| 指標 | 數值 |
|------|------|
| WR | 65.67% |
| PF | 2.19 |
| DD | 6.46% |
| Sharpe | 55.49 |
| 月交易 | 190 |
| TP+SL | 59.5% |
| Long PF | 2.22 |
| Short PF | 2.15 |

## 使用方式

```python
from strategy import generate_signals
from backtest_core import BacktestEngine

params = {
    'stop_loss_pct': 0.008,
    'take_profit_pct': 0.008,
    'max_holding_bars': 20
}

engine = BacktestEngine(df, params)
result = engine.run(generate_signals)
```
