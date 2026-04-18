import numpy as np
import pandas as pd

def generate_signals(data, rsi_buy=85, rsi_sell=15):
    """
    1H Direction + RSI Momentum Filter (窄TP/SL版)
    
    Long: 1H bar 上漲 AND RSI < 85
    Short: 1H bar 下跌 AND RSI > 15
    """
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
            signals[i] = 1
        elif dir_1h < 0 and rsi[i] > rsi_sell:
            signals[i] = -1
    
    return signals

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/autoresearch')
    from backtest_core import BacktestEngine
    
    df = pd.read_parquet("/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/btcusdt_15m.parquet")
    
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
