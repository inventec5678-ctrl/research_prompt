# BTC 15分K 交易策略研究框架

## 目標

找出能通過以下門檻的 alpha 策略：
- WR ≥ 60%
- PF ≥ 1.8
- DD ≤ 25%
- Sharpe ≥ 1.5
- 月交易 ≥ 100
- TP + SL > 30%
- Long PF > 1.0 AND Short PF > 1.0

---

## 📁 檔案結構

```
research_prompt/
├── README.md                    # 本檔案（版本變更日誌）
├── 研究者指南.md               # 研究者 prompt (v4.0)
├── backtest_core.py            # 回測引擎
├── monte_carlo_test.py         # Monte Carlo 穩定性測試
├── CHANGELOG.md                # 迭代詳細日誌
│
└── strategies/                  # 策略資料夾
    ├── 1h_rsi_narrow/           # 策略 1：窄 TP/SL 版
    │   ├── strategy.py          # 程式碼
    │   └── README.md
    │
    └── 1h_rsi_momentum_top3/   # 策略 2-4：Top 3 策略
        ├── strategy.py          # 共用程式碼
        ├── README.md            # Overview
        ├── second_place/       # 🥈 最推薦
        ├── first_place/        # 🥇 Sharpe 最高
        └── third_place/        # 🥉 均衡之選
```

---

## 版本變更日誌 (Changelog)

### v5.2 — 2026-04-18 19:04
**整理 GitHub 資料夾結構**

- 移除多餘檔案
- strategies 資料夾每個策略一個子資料夾

---

### v5.1 — 2026-04-18
**Reorganize strategies folder**

- 每個策略一個資料夾
- 包含 strategy.py 程式碼

---

### v5.0 — 2026-04-18
**新增 Top 3 策略 + Overfitting 分析**

- researcher_v11 完成 100 輪測試
- Top 3 策略出爐
- Overfitting 風險評估通過

---

### v4.0 — 2026-04-18
**強制持續搜索直到完成 100 輪**

---

## 🚀 快速開始

### 1. 運行回測

```python
import sys
sys.path.insert(0, '/path/to/autoresearch')

import pandas as pd
from backtest_core import BacktestEngine
from strategies.1h_rsi_momentum_top3.strategy import generate_signals

df = pd.read_parquet("btcusdt_15m.parquet")

params = {
    'stop_loss_pct': 0.005,      # 0.5%
    'take_profit_pct': 0.0065,   # 0.65%
    'max_holding_bars': 7
}

engine = BacktestEngine(df, params)
result = engine.run(generate_signals)

print(f"WR: {result['wr']:.2f}%")
print(f"PF: {result['pf']:.2f}")
print(f"Sharpe: {result['sharpe']:.2f}")
```

### 2. Monte Carlo 穩定性測試

```python
from monte_carlo_test import MonteCarloTester

tester = MonteCarloTester(df, generate_signals, params)
mc = tester.run(n_bootstrap=500, block_size=10)

print(f"Sharpe: {mc['sharpe_mean']:.2f} ± {mc['sharpe_std']:.2f}")
print(f"通過: {mc['passed']}")
```

---

## 🏆 策略總覽

| 策略 | 資料夾 | Sharpe | WR | PF | 推薦 |
|------|--------|--------|-----|-----|------|
| 1h_rsi_narrow | 1h_rsi_narrow/ | 55.49 | 65.67% | 2.19 | 基礎版 |
| Top 3 第二名 | 1h_rsi_momentum_top3/second_place/ | 72.83 | 67.85% | 3.05 | **最推薦** |
| Top 3 第一名 | 1h_rsi_momentum_top3/first_place/ | 76.02 | 71.60% | 3.04 | Sharpe 最高 |
| Top 3 第三名 | 1h_rsi_momentum_top3/third_place/ | 72.35 | 66.76% | 2.95 | 均衡之選 |

---

## 門檻

| 指標 | 門檻 |
|------|------|
| WR（勝率） | ≥ 60% |
| PF（盈虧比） | ≥ 1.8 |
| DD（最大回撤） | ≤ 25% |
| Sharpe | ≥ 1.5 |
| 月交易筆數 | ≥ 100 |
| **Exit 有效性** | TP + SL > 30% |
| **雙向獲利** | Long PF > 1.0 AND Short PF > 1.0 |

---

## 框架教訓 (Framework Lessons)

| Lesson | 發現 | 框架改動 |
|--------|------|----------|
| Timeout 不是有效 Exit | 98.4% Timeout 出場 | v2.0 新增 Exit 有效性要求 |
| 牛市 beta ≠ alpha | Sharpe 66 只是牛市 | v2.0 新增雙向獲利要求 |
| 高 Sharpe 可能只是運氣 | 單一結果可能是運氣 | 要求多配置驗證 |
| 每次測試都要儲存 | 研究者跳過儲存 | v1.1 強制規定儲存 |
### v6.0 — 2026-04-18 20:20
**新增參數調整上限**

#### Prompt 變更
```
## ⚠️ 核心限制：參數調整上限

1. 同一策略的參數調整上限：5 輪
2. 連續 2 次無進步 → 強制換策略
```

#### 原因
研究者在 100 輪測試中一直在調整參數，但策略本身沒有通過。需要強制换策略而非无限调参。
