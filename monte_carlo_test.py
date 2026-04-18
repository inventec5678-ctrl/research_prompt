#!/usr/bin/env python3
"""
monte_carlo_test.py - Monte Carlo 穩定性測試

用於評估策略在不同市場環境下的穩定性

使用方式：
```python
from monte_carlo_test import MonteCarloTester

tester = MonteCarloTester(df, signal_func, params)
mc_result = tester.run(n_bootstrap=500, block_size=10)

print(f"Sharpe: {mc_result['sharpe_mean']:.2f} ± {mc_result['sharpe_std']:.2f}")
print(f"通過: {mc_result['passed']}")
```

風險評估：
- Sharpe 的穩定性
- VaR (Value at Risk) 5%
- 最大虧損分佈
"""

import numpy as np
import pandas as pd
import random
from typing import Callable, Dict, List


class MonteCarloTester:
    """
    Monte Carlo 穩定性測試
    
    使用 Block Bootstrap 方法評估策略在不同市場環境下的表現
    """
    
    def __init__(self, data: pd.DataFrame, signal_func: Callable, params: dict):
        """
        Args:
            data: DataFrame（含 close, high, low, open, volume）
            signal_func: 信號產生函數
            params: 回測參數（stop_loss_pct, take_profit_pct, max_holding_bars）
        """
        self.data = data
        self.signal_func = signal_func
        self.params = params
        
        # 預先運行一次回測，獲取交易記錄
        self.trades = self._run_backtest_raw()
    
    def _run_backtest_raw(self) -> List[Dict]:
        """運行回測並返回原始交易記錄"""
        from backtest_core import BacktestEngine
        
        engine = BacktestEngine(self.data, self.params)
        
        # 獲取內部交易資料
        signals = self.signal_func(self.data)
        if hasattr(signals, 'tolist'):
            signals = signals.tolist()
        
        closes = self.data['close'].values.astype(float)
        highs = self.data['high'].values.astype(float)
        lows = self.data['low'].values.astype(float)
        n = len(closes)
        
        sl_pct = self.params.get('stop_loss_pct', 0.02)
        tp_pct = self.params.get('take_profit_pct', 0.04)
        max_bars = self.params.get('max_holding_bars', None)
        COST = 0.0003
        
        trades = []
        position = None
        entry_bar = 0
        
        for i in range(n - 1):
            if position is None:
                if signals[i] != 0:
                    entry_price = closes[i + 1]
                    position = {
                        'direction': signals[i],
                        'entry_price': entry_price,
                        'entry_bar': i + 1
                    }
                    entry_bar = i + 1
            else:
                direction = position['direction']
                entry_price = position['entry_price']
                current_price = closes[i]
                holding_bars = i - entry_bar
                
                if direction > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                exit_reason = None
                final_pnl = pnl_pct
                
                if pnl_pct <= -sl_pct:
                    exit_reason = 'SL'
                    final_pnl = -sl_pct - COST
                elif pnl_pct >= tp_pct:
                    exit_reason = 'TP'
                    final_pnl = tp_pct - COST
                elif max_bars is not None and holding_bars >= max_bars:
                    exit_reason = 'TO'
                    final_pnl = pnl_pct - COST
                
                if exit_reason:
                    trades.append({
                        'pnl_pct': final_pnl,
                        'won': final_pnl > 0,
                        'exit': exit_reason,
                        'direction': direction,
                        'holding_bars': holding_bars,
                        'entry_bar': entry_bar
                    })
                    position = None
        
        return trades
    
    def run(self, n_bootstrap: int = 500, block_size: int = 10) -> Dict:
        """
        運行 Monte Carlo 測試
        
        Args:
            n_bootstrap: Bootstrap 次數（默認 500）
            block_size: Block 大小（默認 10，用於保持時間序列結構）
        
        Returns:
            dict: 包含穩定性指標
        """
        if len(self.trades) < 30:
            return {
                'passed': False,
                'reason': f'交易筆數不足 ({len(self.trades)} < 30)'
            }
        
        returns = [t['pnl_pct'] for t in self.trades]
        
        # Block Bootstrap
        bootstrap_sharpes = []
        bootstrap_returns = []
        bootstrap_max_dds = []
        bootstrap_win_rates = []
        
        for _ in range(n_bootstrap):
            # 選擇 block 起始點
            indices = []
            while len(indices) < len(returns):
                start = random.randint(0, max(0, len(returns) - block_size))
                indices.extend(range(start, start + block_size))
            indices = indices[:len(returns)]
            
            # 採樣
            bootstrap_rets = [returns[i] for i in indices]
            
            # 計算 Sharpe
            if np.std(bootstrap_rets) > 0:
                sharpe = np.mean(bootstrap_rets) / np.std(bootstrap_rets) * np.sqrt(252 * 96)
            else:
                sharpe = 0
            
            # 計算累計收益
            cumulative = np.cumprod([1 + r for r in bootstrap_rets])
            
            # 計算 Max Drawdown
            peak = np.maximum.accumulate(cumulative)
            dd = (peak - cumulative) / peak
            max_dd = np.max(dd) * 100 if len(dd) > 0 else 0
            
            # 計算 Win Rate
            win_rate = np.mean([r > 0 for r in bootstrap_rets]) * 100
            
            bootstrap_sharpes.append(sharpe)
            bootstrap_returns.append(np.sum(bootstrap_rets))
            bootstrap_max_dds.append(max_dd)
            bootstrap_win_rates.append(win_rate)
        
        # 統計結果
        sharpe_mean = np.mean(bootstrap_sharpes)
        sharpe_std = np.std(bootstrap_sharpes)
        sharpe_5 = np.percentile(bootstrap_sharpes, 5)
        sharpe_95 = np.percentile(bootstrap_sharpes, 95)
        
        returns_mean = np.mean(bootstrap_returns)
        returns_5 = np.percentile(bootstrap_returns, 5)
        
        dd_mean = np.mean(bootstrap_max_dds)
        dd_95 = np.percentile(bootstrap_max_dds, 95)
        
        wr_mean = np.mean(bootstrap_win_rates)
        
        # 評估標準
        # Sharpe 穩定性：5th percentile 應該 > 0
        # Sharpe 波動性：std 不應該太大
        # VaR 5%：應該為正（賺錢）
        
        passed = (
            sharpe_5 > 0 and
            sharpe_mean > 1.5 and
            sharpe_std < sharpe_mean * 2 and  # 波動不應超過均值 2 倍
            dd_95 < 50  # 95% 信心最大虧損 < 50%
        )
        
        return {
            'n_trades': len(self.trades),
            'n_bootstrap': n_bootstrap,
            'block_size': block_size,
            
            # Sharpe 分析
            'sharpe_mean': sharpe_mean,
            'sharpe_std': sharpe_std,
            'sharpe_5pct': sharpe_5,
            'sharpe_95pct': sharpe_95,
            'sharpe_cv': sharpe_std / sharpe_mean if sharpe_mean > 0 else 0,  # 變異係數
            
            # 收益分析
            'returns_mean': returns_mean,
            'returns_5pct': returns_5,
            
            # 風險分析
            'dd_mean': dd_mean,
            'dd_95pct': dd_95,
            
            # 勝率分析
            'wr_mean': wr_mean,
            
            # 評估結果
            'passed': passed,
            
            # 診斷訊息
            'diagnostics': self._get_diagnostics(
                sharpe_mean, sharpe_std, sharpe_5, dd_95, wr_mean
            )
        }
    
    def _get_diagnostics(self, sharpe_mean, sharpe_std, sharpe_5, dd_95, wr_mean) -> Dict:
        """生成診斷訊息"""
        diagnostics = []
        severity = 'info'
        
        if sharpe_std > sharpe_mean * 2:
            diagnostics.append(f"⚠️ Sharpe 波動性高 (CV={sharpe_std/sharpe_mean:.1f})")
            severity = 'warning'
        
        if sharpe_5 < 0:
            diagnostics.append(f"🚨 Sharpe 5th percentile < 0 ({sharpe_5:.2f})")
            severity = 'error'
        elif sharpe_5 < 1.0:
            diagnostics.append(f"⚠️ Sharpe 5th percentile 偏低 ({sharpe_5:.2f})")
            severity = 'warning'
        
        if dd_95 > 50:
            diagnostics.append(f"🚨 95% VaR DD 過高 ({dd_95:.1f}%)")
            severity = 'error'
        elif dd_95 > 25:
            diagnostics.append(f"⚠️ 95% VaR DD 偏高 ({dd_95:.1f}%)")
            severity = 'warning'
        
        if wr_mean < 50:
            diagnostics.append(f"⚠️ 平均勝率偏低 ({wr_mean:.1f}%)")
        
        if not diagnostics:
            diagnostics.append("✅ 所有穩定性指標正常")
        
        return {
            'messages': diagnostics,
            'severity': severity
        }


def run_monte_carlo_test(data_path: str, signal_func: Callable, params: dict) -> Dict:
    """
    快速運行 Monte Carlo 測試
    
    Args:
        data_path: 數據檔案路徑
        signal_func: 信號函數
        params: 回測參數
    
    Returns:
        dict: Monte Carlo 測試結果
    """
    df = pd.read_parquet(data_path)
    
    tester = MonteCarloTester(df, signal_func, params)
    result = tester.run(n_bootstrap=500, block_size=10)
    
    return result


# ============ 測試 ============

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/autoresearch')
    
    from backtest_core import BacktestEngine
    
    # 載入數據
    data_path = "/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/btcusdt_15m.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"數據：{len(df)} 筆")
    
    # 測試信號：1H Direction + RSI
    def test_signal(data):
        C = data['close'].values
        n = len(C)
        
        # RSI
        deltas = np.diff(C)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        rsi = np.zeros(n)
        for i in range(14, n):
            gain_avg = np.mean(gains[i-14:i])
            loss_avg = np.mean(losses[i-14:i])
            rs = gain_avg / (loss_avg + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
        
        # 1H direction
        n_1h = n // 4
        C_1h = np.array([C[i*4+3] for i in range(n_1h)])
        bar_dir_1h = np.sign(np.diff(C_1h))
        
        signals = np.zeros(n, dtype=int)
        for i in range(20, n - 4):
            h_idx = i // 4
            if h_idx >= len(bar_dir_1h):
                continue
            dir_1h = bar_dir_1h[h_idx]
            if dir_1h > 0 and rsi[i] < 85:
                signals[i] = 1
            elif dir_1h < 0 and rsi[i] > 15:
                signals[i] = -1
        
        return signals
    
    params = {
        'stop_loss_pct': 0.008,
        'take_profit_pct': 0.008,
        'max_holding_bars': 20
    }
    
    print("\n運行 Monte Carlo 測試...")
    tester = MonteCarloTester(df, test_signal, params)
    result = tester.run(n_bootstrap=500, block_size=10)
    
    print("\n=== Monte Carlo 結果 ===")
    print(f"交易筆數：{result['n_trades']}")
    print(f"Bootstrap 次數：{result['n_bootstrap']}")
    print()
    print(f"Sharpe：{result['sharpe_mean']:.2f} ± {result['sharpe_std']:.2f}")
    print(f"Sharpe 5th-95th：{result['sharpe_5pct']:.2f} ~ {result['sharpe_95pct']:.2f}")
    print(f"Sharpe 變異係數：{result['sharpe_cv']:.2f}")
    print()
    print(f"收益均值：{result['returns_mean']:.4f}")
    print(f"收益 5th percentile：{result['returns_5pct']:.4f}")
    print()
    print(f"平均 DD：{result['dd_mean']:.2f}%")
    print(f"95% VaR DD：{result['dd_95pct']:.2f}%")
    print()
    print(f"平均勝率：{result['wr_mean']:.2f}%")
    print()
    print(f"診斷：")
    for msg in result['diagnostics']['messages']:
        print(f"  {msg}")
    print()
    print(f"通過：{'✅' if result['passed'] else '❌'}")
