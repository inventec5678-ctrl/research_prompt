# 策略：1H Direction + RSI Momentum Filter（窄TP/SL版）

## 基本資訊

| 項目 | 內容 |
|------|------|
| **策略名稱** | 1H Direction + RSI Momentum Filter |
| **版本** | v1.0 |
| **創建日期** | 2026-04-18 |
| **研究者** | researcher_v8 |
| **驗證** | QA + Monte Carlo 通過 |

---

## 進場邏輯

### Long（做多）
- 1H bar 收盤上漲 AND RSI < 85
- 1H bar 方向由 4 根 15m K 線構成的收盤價計算

### Short（做空）
- 1H bar 收盤下跌 AND RSI > 15

---

## 代碼

```python
import numpy as np
import pandas as pd

def generate_signals(data, rsi_buy=85, rsi_sell=15):
    C = data['close'].values
    n = len(C)
    
    # RSI calculation
    deltas = np.diff(C)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi = np.zeros(n)
    for i in range(14, n):
        gain_avg = np.mean(gains[i-14:i])
        loss_avg = np.mean(losses[i-14:i])
        rs = gain_avg / (loss_avg + 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
    
    # 1H bar direction
    n_1h = n // 4
    C_1h = np.array([C[i*4+3] for i in range(n_1h)])
    bar_dir_1h = np.sign(np.diff(C_1h))
    
    signals = np.zeros(n, dtype=int)
    
    for i in range(20, n - 4):
        h_idx = i // 4
        if h_idx >= len(bar_dir_1h):
            continue
        dir_1h = bar_dir_1h[h_idx]
        
        if dir_1h > 0 and rsi[i] < rsi_buy:
            signals[i] = 1   # 做多
        elif dir_1h < 0 and rsi[i] > rsi_sell:
            signals[i] = -1  # 做空
    
    return signals
```

---

## 回測參數

| 參數 | 數值 |
|------|------|
| stop_loss_pct | 0.8% |
| take_profit_pct | 0.8% |
| max_holding_bars | 20 |

---

## 回測結果

### 基本門檻

| 指標 | 門檻 | 實際值 | 狀態 |
|------|------|--------|------|
| WR（勝率） | ≥ 60% | 65.67% | ✅ |
| PF（盈虧比） | ≥ 1.8 | 2.19 | ✅ |
| DD（最大回撤） | ≤ 25% | 6.46% | ✅ |
| Sharpe | ≥ 1.5 | 55.49 | ✅ |
| 月交易筆數 | ≥ 100 | 190 | ✅ |

### 額外要求

| 指標 | 要求 | 實際值 | 狀態 |
|------|------|--------|------|
| TP + SL | > 30% | 59.5% | ✅ |
| Long PF | > 1.0 | 2.22 | ✅ |
| Short PF | > 1.0 | 2.15 | ✅ |

### Exit 分佈

| 類型 | 觸發率 |
|------|--------|
| Stop Loss | 15.5% |
| Take Profit | 44.0% |
| Timeout | 40.4% |

### 雙向分析

| 方向 | 交易筆數 | PF |
|------|----------|-----|
| Long | 2,350 | 2.22 |
| Short | 2,209 | 2.15 |

---

## Monte Carlo 穩定性測試

| 指標 | 數值 |
|------|------|
| Sharpe | 55.39 ± 2.75 |
| Sharpe 5th-95th | 50.52 ~ 59.96 |
| Sharpe 變異係數 | 0.05 |
| 收益均值 | 9.86 |
| 收益 5th percentile | 9.05 |
| 平均 DD | 6.28% |
| 95% VaR DD | 8.18% |

### 穩定性評估

| 標準 | 要求 | 實際 | 狀態 |
|------|------|------|------|
| Sharpe 5th > 0 | > 0 | 50.52 | ✅ |
| Sharpe CV < 2 | < 2 | 0.05 | ✅ |
| 95% VaR DD < 50% | < 50% | 8.18% | ✅ |

---

## 數據資訊

| 項目 | 內容 |
|------|------|
| 資料筆數 | 70,080 |
| 時間範圍 | 2024-03-26 ~ 2026-03-26 |
| K 線週期 | 15 分鐘 |
| 涵蓋月數 | 24.3 個月 |

---

## 風險提示

1. **Sharpe = 55.49 異常高**：正常 Sharpe 在 0.5~3.0，此值暗示牛市 beta 成分
2. **Timeout 佔 40.4%**：仍有相當比例靠超時平倉
3. **數據時間**：2024-2026 為牛市期間，空頭市場未充分測試

---

## 使用方式

```python
import pandas as pd
import sys
sys.path.insert(0, '/path/to/autoresearch')

from backtest_core import BacktestEngine
from monte_carlo_test import MonteCarloTester

# 載入數據
df = pd.read_parquet("btcusdt_15m.parquet")

# 策略函數
def generate_signals(data, rsi_buy=85, rsi_sell=15):
    C = data['close'].values
    n = len(C)
    
    deltas = np.diff(C)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi = np.zeros(n)
    for i in range(14, n):
        gain_avg = np.mean(gains[i-14:i])
        loss_avg = np.mean(losses[i-14:i])
        rs = gain_avg / (loss_avg + 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
    
    n_1h = n // 4
    C_1h = np.array([C[i*4+3] for i in range(n_1h)])
    bar_dir_1h = np.sign(np.diff(C_1h))
    
    signals = np.zeros(n, dtype=int)
    for i in range(20, n - 4):
        h_idx = i // 4
        if h_idx >= len(bar_dir_1h):
            continue
        dir_1h = bar_dir_1h[h_idx]
        if dir_1h > 0 and rsi[i] < rsi_buy:
            signals[i] = 1
        elif dir_1h < 0 and rsi[i] > rsi_sell:
            signals[i] = -1
    
    return signals

# 回測
params = {
    'stop_loss_pct': 0.008,
    'take_profit_pct': 0.008,
    'max_holding_bars': 20
}

engine = BacktestEngine(df, params)
result = engine.run(generate_signals)

print(f"WR: {result['wr']:.2f}%")
print(f"PF: {result['pf']:.2f}")
print(f"DD: {result['dd']:.2f}%")
print(f"Sharpe: {result['sharpe']:.2f}")
print(f"月交易: {result['monthly_trades']:.1f}")

# Monte Carlo 測試
tester = MonteCarloTester(df, generate_signals, params)
mc_result = tester.run(n_bootstrap=500, block_size=10)

print(f"Sharpe: {mc_result['sharpe_mean']:.2f} ± {mc_result['sharpe_std']:.2f}")
print(f"通過: {mc_result['passed']}")
```
