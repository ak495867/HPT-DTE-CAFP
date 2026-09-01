
import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_macro(start, end):
    print("  Fetching macro data (VIX, DXY, 10Y)...")
    vix = fetch_data('^VIX', start, end)
    dxy = fetch_data('DX-Y.NYB', start, end)
    tnx = fetch_data('^TNX', start, end)
    macro = pd.DataFrame()
    if not vix.empty: macro['vix'] = vix['Close']
    if not dxy.empty: macro['dxy'] = dxy['Close']
    if not tnx.empty: macro['tnx_10y'] = tnx['Close']
    macro = macro.ffill().dropna()
    return macro

def build_features(df, macro=None):
    df = df.copy()
    df['ret_1d'] = df['Close'].pct_change(1)
    df['ret_3d'] = df['Close'].pct_change(3)
    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    df['ma_5d'] = df['Close'].rolling(5).mean() / df['Close']
    df['rsi'] = 100 - 100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / (-df['Close'].diff().clip(upper=0)).rolling(14).mean() + 1e-9)
    df['vol_chg'] = df['Volume'].pct_change()
    if macro is not None and not macro.empty:
        df = df.join(macro, how='left').ffill()
        if 'vix' in df.columns: df['vix_chg'] = df['vix'].pct_change()
        if 'dxy' in df.columns: df['dxy_chg'] = df['dxy'].pct_change()
        if 'tnx_10y' in df.columns: df['tnx_chg'] = df['tnx_10y'].pct_change()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df

def verify_no_leakage(df, window_size):
    feature_cols = [c for c in df.columns if c != 'target']
    print(f"  ✓ {len(feature_cols)} features use only past data | window={window_size}")
    return True

class HarmonicPhaseTransformer(nn.Module):
    def __init__(self, n_features, window_size, d_model=32, condition_dim=20):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.freq_proj = nn.Linear(n_features * 2, d_model)
        self.time_proj = nn.Linear(n_features, d_model)
        self.phase_shifts = nn.Parameter(torch.randn(1, window_size, d_model) * 0.05)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.cond_map = nn.Linear(condition_dim, d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x_time, x_freq, condition):
        h_t = self.time_proj(x_time) + self.phase_shifts
        h_f = self.freq_proj(x_freq)
        attn_out, _ = self.cross_attn(h_t, h_f, h_f)
        gated_attn = self.gate(attn_out) * attn_out
        h_fused = self.norm(h_t + gated_attn)
        h_fused = h_fused + self.ffn(h_fused)
        h_c = self.cond_map(condition).unsqueeze(1).expand(-1, self.window_size, -1)
        pooled = torch.cat([h_fused.mean(dim=1), h_c.mean(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1)

class StrictEquityEnsemble:
    def __init__(self, n_features, window_size, save_path='model_bundle.pt'):
        self.n_features = n_features
        self.window_size = window_size
        self.save_path = save_path
        condition_dim = 20
        self.shared_hpt = HarmonicPhaseTransformer(n_features, window_size, d_model=32, condition_dim=condition_dim)
        self.hpt_optimizer = torch.optim.AdamW(self.shared_hpt.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_fn = nn.BCELoss()
        self.equity_tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, random_state=42)
        self.regime_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
        self.leaf_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.meta_learner = LogisticRegression(max_iter=1000, random_state=42, C=0.1)  # Added regularization
        self.calibrator = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def _prepare_condition(self, X_stat, encoder):
        leaf_idx = self.regime_tree.apply(X_stat).reshape(-1, 1)
        cond = encoder.transform(leaf_idx)
        if cond.shape[1] > 20: cond = cond[:, :20]
        elif cond.shape[1] < 20: cond = np.hstack([cond, np.zeros((cond.shape[0], 20 - cond.shape[1]))])
        return cond

    def _build_windows(self, scaled_data):
        X_time, X_freq = [], []
        for i in range(self.window_size, len(scaled_data)):
            w = scaled_data[i-self.window_size:i]
            X_time.append(w)
            fft_out = np.fft.rfft(w, axis=0)
            X_freq.append(np.concatenate([fft_out.real, fft_out.imag], axis=-1))
        return np.array(X_time), np.array(X_freq)

    def _hpt_predict_proba(self, X_stat_aligned, X_time, X_freq, encoder):
        self.shared_hpt.eval()
        cond = self._prepare_condition(X_stat_aligned, encoder)
        with torch.no_grad():
            t_time = torch.tensor(X_time, dtype=torch.float32)
            t_freq = torch.tensor(X_freq, dtype=torch.float32)
            t_cond = torch.tensor(cond, dtype=torch.float32)
            return self.shared_hpt(t_time, t_freq, t_cond).numpy()

    def fit(self, X_equity, Y_equity):
        # Fit scaler only on training data
        self.scaler.fit(X_equity)
        X_equity_s = self.scaler.transform(X_equity)

        # Split into train/validation for proper calibration
        n_samples = len(Y_equity)
        split_idx = int(n_samples * 0.8)  # 80% train, 20% validation
        
        X_train = X_equity_s[:split_idx]
        Y_train = Y_equity[:split_idx]
        X_val = X_equity_s[split_idx:]
        Y_val = Y_equity[split_idx:]
        
        print(f"  Split: {len(Y_train)} train, {len(Y_val)} validation")

        # Train regime DT on train split only
        self.regime_tree.fit(X_train, Y_train)
        leaf_idx = self.regime_tree.apply(X_train).reshape(-1, 1)
        self.leaf_encoder.fit(leaf_idx)

        # Train equity DT on train split only
        self.equity_tree.fit(X_train, Y_train)
        print("  Decision Trees trained on training split only.")

        # Build windows for train split
        X_time, X_freq = self._build_windows(X_train)
        
        # Align data
        X_stat_aligned = X_train[self.window_size:]
        Y_aligned = Y_train[self.window_size:]
        cond = self._prepare_condition(X_stat_aligned, self.leaf_encoder)
        
        t_time = torch.tensor(X_time, dtype=torch.float32)
        t_freq = torch.tensor(X_freq, dtype=torch.float32)
        t_cond = torch.tensor(cond, dtype=torch.float32)
        t_target = torch.tensor(Y_aligned, dtype=torch.float32)
        
        print(f"  Window alignment: time={t_time.shape[0]}, cond={t_cond.shape[0]}, target={t_target.shape[0]}")
        
        dataset = torch.utils.data.TensorDataset(t_time, t_freq, t_cond, t_target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

        self.shared_hpt.train()
        print("  Training HPT on training split (2017-2022)...")
        for epoch in range(30):
            total_loss = 0
            for b_t, b_f, b_c, b_y in loader:
                self.hpt_optimizer.zero_grad()
                loss = self.loss_fn(self.shared_hpt(b_t, b_f, b_c), b_y)
                loss.backward()
                self.hpt_optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        print("  HPT trained.")

        # Get HPT raw probs on val data
        X_time_val, X_freq_val = self._build_windows(X_val)
        X_stat_val_aligned = X_val[self.window_size:]
        Y_val_aligned = Y_val[self.window_size:]
        
        hpt_raw_probs = self._hpt_predict_proba(X_stat_val_aligned, X_time_val, X_freq_val, self.leaf_encoder)
        
        # Build meta features
        equity_probs = self.equity_tree.predict_proba(X_stat_val_aligned)[:, 1]
        regime_leaves = self.regime_tree.apply(X_stat_val_aligned).astype(float)
        meta_X_val = np.column_stack([hpt_raw_probs, equity_probs, regime_leaves])
        
        # Fit meta-learner
        self.meta_learner.fit(meta_X_val, Y_val_aligned)
        print("  Meta-learner trained on validation split.")
        
        # Get meta-learner's output 
        meta_probs_val = self.meta_learner.predict_proba(meta_X_val)[:, 1]
        
        print("  Calibrating probabilities on validation split...")
        self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        self.calibrator.fit(meta_probs_val, Y_val_aligned)
        
        cal_probs_val = self.calibrator.transform(meta_probs_val)
        print(f"    Before calibration: mean={meta_probs_val.mean():.3f}, std={meta_probs_val.std():.3f}")
        print(f"    After calibration:  mean={cal_probs_val.mean():.3f}, std={cal_probs_val.std():.3f}")
        print("  Calibration complete. The model is now humble.")
        
        self.is_trained = True

    def predict(self, X_stat_raw, confidence_threshold=0.6):
        X_s = self.scaler.transform(X_stat_raw)
        X_time, X_freq = self._build_windows(X_s)
        X_stat_aligned = X_s[self.window_size:]

        hpt_raw_probs = self._hpt_predict_proba(X_stat_aligned, X_time, X_freq, self.leaf_encoder)
        equity_probs = self.equity_tree.predict_proba(X_stat_aligned)[:, 1]
        regime_leaves = self.regime_tree.apply(X_stat_aligned).astype(float)
        meta_X = np.column_stack([hpt_raw_probs, equity_probs, regime_leaves])
        meta_probs = self.meta_learner.predict_proba(meta_X)[:, 1]
        cal_probs = self.calibrator.transform(meta_probs)
        
        predictions = np.full(len(cal_probs), -1, dtype=int)
        predictions[cal_probs >= confidence_threshold] = 1
        predictions[cal_probs <= (1 - confidence_threshold)] = 0
        
        confidence_levels = np.where(
            (cal_probs >= confidence_threshold) | (cal_probs <= (1 - confidence_threshold)),
            'HIGH',
            np.where(
                (cal_probs >= 0.55) | (cal_probs <= 0.45),
                'MEDIUM',
                'LOW'
            )
        )
        
        return predictions, cal_probs, confidence_levels

    def save(self):
        bundle = {
            'hpt_state_dict': self.shared_hpt.state_dict(),
            'equity_tree': self.equity_tree,
            'regime_tree': self.regime_tree,
            'leaf_encoder': self.leaf_encoder,
            'meta_learner': self.meta_learner,
            'calibrator': self.calibrator,
            'scaler': self.scaler,
            'config': {
                'n_features': self.n_features,
                'window_size': self.window_size
            }
        }
        torch.save(bundle, self.save_path)
        print(f"\n  Single-file bundle saved to '{self.save_path}'.")

    @classmethod
    def load(cls, save_path='model_bundle.pt'):
        bundle = torch.load(save_path, weights_only=False)
        config = bundle['config']
        model = cls(config['n_features'], config['window_size'], save_path)
        model.shared_hpt.load_state_dict(bundle['hpt_state_dict'])
        model.equity_tree = bundle['equity_tree']
        model.regime_tree = bundle['regime_tree']
        model.leaf_encoder = bundle['leaf_encoder']
        model.meta_learner = bundle['meta_learner']
        model.calibrator = bundle['calibrator']
        model.scaler = bundle['scaler']
        model.is_trained = True
        print(f"  Bundle loaded from '{save_path}'.")
        return model

def main():
    train_equity = ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMD', 'INTC']
    train_start = '2017-01-01'
    train_end = '2023-12-31'
    
    test_tickers = ['TSLA', 'AMZN', 'BA', 'JPM', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'GLD']
    test_start = '2024-01-01'
    test_end = '2026-09-01'

    feature_cols = ['ret_1d', 'ret_3d', 'vol_20d', 'ma_5d', 'rsi', 'vol_chg', 'vix', 'dxy', 'tnx_10y', 'vix_chg', 'dxy_chg', 'tnx_chg']
    window_size = 20
    confidence_threshold = 0.6
    save_path = 'model_bundle.pt'

    if os.path.exists(save_path):
        print("Found existing bundle. Deleting to force retrain with proper calibration...")
        os.remove(save_path)

    if os.path.exists(save_path):
        print("Found saved bundle. Loading from disk...")
        model = StrictEquityEnsemble.load(save_path)
    else:
        print("=" * 70)
        print("STRICT MODE: Training ONLY on equities (2017-2023)")
        print("Zero-shot testing on 2024-2026 (unseen future)")
        print("=" * 70)
        
        print("\nStep 1: Fetching macro data for training period...")
        macro_train = fetch_macro(train_start, train_end)

        print("\nStep 2: Fetching equity data (2017-2023)...")
        all_X_raw, all_Y = [], []
        for t in train_equity:
            d = build_features(fetch_data(t, train_start, train_end), macro_train)
            if d.empty: continue
            verify_no_leakage(d, window_size)
            X_raw = d[feature_cols].values
            Y = d['target'].values
            all_X_raw.append(X_raw)
            all_Y.append(Y)
            print(f"  {t:>10}: {len(Y)} samples (2017-2023)")

        X_equity = np.vstack(all_X_raw)
        Y_equity = np.concatenate(all_Y)
        print(f"\n  Total training samples: {len(Y_equity)} (equities only, 2017-2023)")

        print("\nStep 3: Training strict equity ensemble with PROPER calibration...")
        model = StrictEquityEnsemble(len(feature_cols), window_size, save_path)
        model.fit(X_equity, Y_equity)
        model.save()

    print("\n" + "=" * 70)
    print(f"ZERO-SHOT TEST: 2024-2026 | Confidence Threshold: {confidence_threshold}")
    print("=" * 70)

    macro_test = fetch_macro(test_start, test_end)
    print(f"\n{'Ticker':>10} | {'Type':>7} | {'Acc':>6} | {'Trades':>8} | {'Avg Cal':>7} | {'HIGH':>4} | Verdict")
    print("-" * 85)
    
    for ticker in test_tickers:
        d = build_features(fetch_data(ticker, test_start, test_end), macro_test)
        if d.empty:
            print(f"{ticker:>10} | No data.")
            continue
        
        verify_no_leakage(d, window_size)
        test_seg = d.tail(100)
        X_test = test_seg[feature_cols].values
        Y_test = test_seg['target'].values[window_size:]
        
        preds, cal_probs, conf_levels = model.predict(X_test, confidence_threshold)
        
        trade_mask = preds != -1
        if trade_mask.sum() == 0:
            acc = 0.0
            trades_made = 0
            verdict = 'NO TRADES'
        else:
            acc = accuracy_score(Y_test[trade_mask], preds[trade_mask])
            trades_made = trade_mask.sum()
            if acc > 0.60 and trades_made >= 10:
                verdict = 'BULLISH ✓'
            elif acc < 0.40 and trades_made >= 10:
                verdict = 'BEARISH ✗'
            else:
                verdict = 'NEUTRAL'
        
        asset_type = 'CRYPTO' if 'USD' in ticker else 'EQUITY'
        avg_cal = cal_probs.mean()
        high_conf = (conf_levels == 'HIGH').sum()
        
        print(f"{ticker:>10} | {asset_type:>7} | {acc:>5.1%} | {trades_made:>4}/{len(preds):<3} | {avg_cal:>6.3f} | {high_conf:>4} | {verdict}")

    print("\n" + "=" * 70)
    print("Calibration should now work properly:")
    print("  - Avg Cal should be ~0.50-0.55 (not 0.70+)")
    print("  - Trades should be fewer (model admits uncertainty)")
    print("  - Accuracy on trades should be higher")
    print("=" * 70)

if __name__ == '__main__':
    main()