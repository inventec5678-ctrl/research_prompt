#!/usr/bin/env python3
"""
backtest_core.py - 標準化回測框架 v3

用於 BTC 15分K 交易策略回測

使用方式：
```python
from backtest_core import BacktestEngine
import pandas as pd

df = pd.read_parquet("data/btcusdt_15m.parquet")

params = {
    'stop_loss_pct': 0.02,        # 2% 止損
    'take_profit_pct': 0.04,      # 4% 止盈
    'max_holding_bars': 20        # 最大持倉K線數
}

engine = BacktestEngine(df, params)
result = engine.run(your_signal_function)

print(f"WR: {result['wr']:.2f}%")
print(f"PF: {result['pf']:.2f}")
```

信號函數格式：
```python
def generate_signals(data) -> list[int]:
    # 返迴 [1, 0, -1, 1, 0, 0, ...]
    # 1 = 做多, -1 = 做空, 0 = 不交易
```

成本：0.03% 單邊（進場 + 出場）
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, List

# ============ 常數 ============
COST = 0.0003  # 0.03% 單邊


# ============ 回測引擎 ============

@dataclass
class Trade:
    entry_price: float
    exit_price: float
    direction: int  # 1=long, -1=short
    pnl_pct: float
    won: bool
    exit_reason: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'TIMEOUT'
    holding_bars: int


class BacktestEngine:
    """
    回測引擎
    
    參數：
        data: DataFrame（含 open, high, low, close, volume）
        params: 字典
            - stop_loss_pct: 止損百分比
            - take_profit_pct: 止盈百分比
            - max_holding_bars: 最大持倉K線數
    """
    
    def __init__(self, data: pd.DataFrame, params: dict):
        self.data = data
        self.params = params
        self.closes = data['close'].values.astype(float)
        self.highs = data['high'].values.astype(float)
        self.lows = data['low'].values.astype(float)
        self.n = len(self.closes)
        
        self.sl_pct = params.get('stop_loss_pct', 0.02)
        self.tp_pct = params.get('take_profit_pct', 0.04)
        self.max_bars = params.get('max_holding_bars', 20)
    
    def run(self, signal_func: Callable) -> dict:
        """
        運行回測
        
        Args:
            signal_func: 信號產生函數，輸入 data，輸出 [1, -1, 0, ...]
        
        Returns:
            dict: 包含 wr, pf, dd, sharpe, trades, monthly_trades 等
        """
        # 產生信號
        signals = signal_func(self.data)
        
        # 如果信號是 numpy array 或其他格式，轉換為 list
        if hasattr(signals, 'tolist'):
            signals = signals.tolist()
        
        trades = []
        position = None
        entry_bar = 0
        
        for i in range(self.n - 1):
            # 沒有倉位時，檢查是否需要進場
            if position is None:
                if signals[i] != 0:
                    # 信號在 bar i 產生，以 bar i+1 開盤價進場
                    entry_price = self.closes[i + 1]
                    position = {
                        'direction': signals[i],
                        'entry_price': entry_price,
                        'entry_bar': i + 1,
                        'highest': entry_price,
                        'lowest': entry_price
                    }
                    entry_bar = i + 1
            else:
                # 有倉位，檢查是否需要出場
                direction = position['direction']
                entry_price = position['entry_price']
                current_price = self.closes[i]
                current_high = self.highs[i]
                current_low = self.lows[i]
                holding_bars = i - entry_bar
                
                # 更新最高/最低價（用於止損止盈計算）
                if direction > 0:
                    if current_high > position['highest']:
                        position['highest'] = current_high
                else:
                    if current_low < position['lowest']:
                        position['lowest'] = current_low
                
                # 計算未實現損益
                if direction > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                exit_reason = None
                final_pnl = pnl_pct
                
                # 檢查止損
                if pnl_pct <= -self.sl_pct:
                    exit_reason = 'STOP_LOSS'
                    final_pnl = -self.sl_pct - COST
                # 檢查止盈
                elif pnl_pct >= self.tp_pct:
                    exit_reason = 'TAKE_PROFIT'
                    final_pnl = self.tp_pct - COST
                # 檢查超時
                elif holding_bars >= self.max_bars:
                    exit_reason = 'TIMEOUT'
                    final_pnl = final_pnl - COST
                
                if exit_reason:
                    trades.append(Trade(
                        entry_price=entry_price,
                        exit_price=current_price,
                        direction=direction,
                        pnl_pct=final_pnl,
                        won=final_pnl > 0,
                        exit_reason=exit_reason,
                        holding_bars=holding_bars
                    ))
                    position = None
        
        return self._calculate_metrics(trades)
    
    def _calculate_metrics(self, trades: list) -> dict:
        """計算策略指標"""
        if not trades:
            return {
                'wr': 0, 'pf': 0, 'dd': 0, 'sharpe': 0,
                'trades': 0, 'monthly_trades': 0,
                'avg_win': 0, 'avg_loss': 0,
                'won': 0, 'lost': 0,
                'passed': False
            }
        
        n_trades = len(trades)
        won_trades = [t for t in trades if t.won]
        lost_trades = [t for t in trades if not t.won]
        
        wr = len(won_trades) / n_trades * 100
        
        avg_win = np.mean([t.pnl_pct for t in won_trades]) if won_trades else 0
        avg_loss = abs(np.mean([t.pnl_pct for t in lost_trades])) if lost_trades else 1e-10
        
        # Profit Factor
        gross_win = avg_win * len(won_trades)
        gross_loss = avg_loss * len(lost_trades)
        pf = gross_win / gross_loss if gross_loss > 0 else 0
        
        # Max Drawdown
        equity = [1.0]
        for t in trades:
            equity.append(equity[-1] * (1 + t.pnl_pct))
        
        peak = 1.0
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        dd_pct = max_dd * 100
        
        # Sharpe Ratio（年化）
        returns = [t.pnl_pct for t in trades]
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96)
        else:
            sharpe = 0
        
        # 月交易量（假設 24 個月）
        monthly_trades = n_trades / 24
        
        passed = (
            wr >= 60 and
            pf >= 1.8 and
            dd_pct <= 25 and
            sharpe >= 1.5 and
            monthly_trades >= 100
        )
        
        return {
            'wr': wr,
            'pf': pf,
            'dd': dd_pct,
            'sharpe': sharpe,
            'trades': n_trades,
            'monthly_trades': monthly_trades,
            'avg_win': avg_win * 100,
            'avg_loss': avg_loss * 100,
            'won': len(won_trades),
            'lost': len(lost_trades),
            'passed': passed,
            'exit_stats': self._exit_stats(trades)
        }
    
    def _exit_stats(self, trades: list) -> dict:
        """計算 exit 分佈"""
        sl_count = sum(1 for t in trades if t.exit_reason == 'STOP_LOSS')
        tp_count = sum(1 for t in trades if t.exit_reason == 'TAKE_PROFIT')
        to_count = sum(1 for t in trades if t.exit_reason == 'TIMEOUT')
        total = len(trades)
        
        return {
            'stop_loss': f"{sl_count/total*100:.1f}%" if total > 0 else "0%",
            'take_profit': f"{tp_count/total*100:.1f}%" if total > 0 else "0%",
            'timeout': f"{to_count/total*100:.1f}%" if total > 0 else "0%"
        }


# ============ 測試 ============

if __name__ == "__main__":
    import sys
    
    # 測試框架
    data_path = "/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/btcusdt_15m.parquet"
    
    print(f"載入數據：{data_path}")
    df = pd.read_parquet(data_path)
    print(f"數據筆數：{len(df)}")
    
    # 測試信號函數
    def test_signal(data):
        closes = data['close'].values
        
        # RSI 計算
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        rsi = np.zeros(len(data))
        for i in range(14, len(data)):
            gain_avg = np.mean(gains[i-14:i])
            loss_avg = np.mean(losses[i-14:i])
            rs = gain_avg / (loss_avg + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
        
        # RSI 信號
        signals = []
        for i in range(len(data)):
            if rsi[i] < 30:
                signals.append(1)
            elif rsi[i] > 70:
                signals.append(-1)
            else:
                signals.append(0)
        
        return signals
    
    params = {
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'max_holding_bars': 20
    }
    
    print("\n運行回測...")
    engine = BacktestEngine(df, params)
    result = engine.run(test_signal)
    
    print("\n結果：")
    print(f"  WR: {result['wr']:.2f}%")
    print(f"  PF: {result['pf']:.2f}")
    print(f"  DD: {result['dd']:.2f}%")
    print(f"  Sharpe: {result['sharpe']:.2f}")
    print(f"  月交易: {result['monthly_trades']:.1f}")
    print(f"  Exit: {result['exit_stats']}")
    print(f"  通過: {result['passed']}")