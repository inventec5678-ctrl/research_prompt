# 1H Direction + RSI + Momentum Top 3

## 策略代碼

所有 3 個策略使用相同的進場邏輯代碼，僅參數不同。

```python
# 見 strategy.py
```

## Top 3 策略

| 排名 | 資料夾 | SL | TP | max_bars | Sharpe |
|------|---------|-----|-----|----------|--------|
| 🥈 第二名 | second_place/ | 0.5% | 0.65% | 7 | 72.83 |
| 🥇 第一名 | first_place/ | 0.4% | 0.4% | 5 | 76.02 |
| 🥉 第三名 | third_place/ | 0.4% | 0.6% | 7 | 72.35 |

## Overfitting 風險評估：🟢 低風險

| 測試 | 結果 |
|------|------|
| Monte Carlo | 3/3 通過 |
| Walk-Forward | 差異 < 4 Sharpe |
| 參數穩定性 | Sharpe 穩定在 70+ |
| 對比簡單策略 | +70~80 Sharpe（真實 alpha）|

## 建議

- **最推薦**：🥈 第二名（WF 穩定性最佳）
- 次選：🥇 第一名（Sharpe 最高）
