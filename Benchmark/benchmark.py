import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
    df['rsi'] = 100 - 100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                             (-df['Close'].diff().clip(upper=0)).rolling(14).mean() + 1e-9)
    df['vol_chg'] = df['Volume'].pct_change()
    if macro is not None and not macro.empty:
        df = df.join(macro, how='left').ffill()
        if 'vix' in df.columns: df['vix_chg'] = df['vix'].pct_change()
        if 'dxy' in df.columns: df['dxy_chg'] = df['dxy'].pct_change()
        if 'tnx_10y' in df.columns: df['tnx_chg'] = df['tnx_10y'].pct_change()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df

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
        self.meta_learner = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
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

    def fit(self, X_equity_list, Y_equity_list):
        if not isinstance(X_equity_list, list) or not isinstance(Y_equity_list, list):
            raise TypeError("X_equity_list and Y_equity_list must be lists of arrays per ticker")
        if len(X_equity_list) != len(Y_equity_list):
            raise ValueError("X and Y lists must have same number of tickers")
        raw_train_list = []   
        raw_val_list = []     
        for X_raw, Y_raw in zip(X_equity_list, Y_equity_list):
            n = len(Y_raw)
            split_idx = int(n * 0.8)
            raw_train_list.append((X_raw[:split_idx], Y_raw[:split_idx]))
            raw_val_list.append((X_raw[split_idx:], Y_raw[split_idx:]))
        all_train_X = np.vstack([x for x, y in raw_train_list])
        self.scaler.fit(all_train_X)
        train_windows_time = []
        train_windows_freq = []
        train_labels = []
        val_windows_time = []
        val_windows_freq = []
        val_labels = []
        for (X_train_raw, Y_train), (X_val_raw, Y_val) in zip(raw_train_list, raw_val_list):
            X_train_s = self.scaler.transform(X_train_raw)
            X_val_s = self.scaler.transform(X_val_raw)
            if len(X_train_s) > self.window_size:
                t_time, t_freq = self._build_windows(X_train_s)
                train_windows_time.append(t_time)
                train_windows_freq.append(t_freq)
                train_labels.append(Y_train[self.window_size:])
            if len(X_val_s) > self.window_size:
                v_time, v_freq = self._build_windows(X_val_s)
                val_windows_time.append(v_time)
                val_windows_freq.append(v_freq)
                val_labels.append(Y_val[self.window_size:])
        X_train_time = np.vstack(train_windows_time)
        X_train_freq = np.vstack(train_windows_freq)
        Y_train_aligned = np.concatenate(train_labels)
        X_val_time = np.vstack(val_windows_time)
        X_val_freq = np.vstack(val_windows_freq)
        Y_val_aligned = np.concatenate(val_labels)
        train_static_list = []
        for (X_train_raw, Y_train) in raw_train_list:
            X_train_s = self.scaler.transform(X_train_raw)
            if len(X_train_s) > self.window_size:
                train_static_list.append(X_train_s[self.window_size:])
        X_train_static = np.vstack(train_static_list)
        val_static_list = []
        for (X_val_raw, Y_val) in raw_val_list:
            X_val_s = self.scaler.transform(X_val_raw)
            if len(X_val_s) > self.window_size:
                val_static_list.append(X_val_s[self.window_size:])
        X_val_static = np.vstack(val_static_list)
        self.regime_tree.fit(X_train_static, Y_train_aligned)
        leaf_idx = self.regime_tree.apply(X_train_static).reshape(-1, 1)
        self.leaf_encoder.fit(leaf_idx)
        self.equity_tree.fit(X_train_static, Y_train_aligned)
        cond_train = self._prepare_condition(X_train_static, self.leaf_encoder)
        t_time = torch.tensor(X_train_time, dtype=torch.float32)
        t_freq = torch.tensor(X_train_freq, dtype=torch.float32)
        t_cond = torch.tensor(cond_train, dtype=torch.float32)
        t_target = torch.tensor(Y_train_aligned, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(t_time, t_freq, t_cond, t_target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        self.shared_hpt.train()
        for epoch in range(30):
            total_loss = 0
            for b_t, b_f, b_c, b_y in loader:
                self.hpt_optimizer.zero_grad()
                loss = self.loss_fn(self.shared_hpt(b_t, b_f, b_c), b_y)
                loss.backward()
                self.hpt_optimizer.step()
                total_loss += loss.item()
        hpt_raw_probs = self._hpt_predict_proba(X_val_static, X_val_time, X_val_freq, self.leaf_encoder)
        equity_probs = self.equity_tree.predict_proba(X_val_static)[:, 1]
        regime_leaves = self.regime_tree.apply(X_val_static).astype(float)
        meta_X_val = np.column_stack([hpt_raw_probs, equity_probs, regime_leaves])
        self.meta_learner.fit(meta_X_val, Y_val_aligned)
        meta_probs_val = self.meta_learner.predict_proba(meta_X_val)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        self.calibrator.fit(meta_probs_val, Y_val_aligned)
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
            np.where((cal_probs >= 0.55) | (cal_probs <= 0.45), 'MEDIUM', 'LOW')
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
        return model

def train_baselines(X_train_features, Y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=50, random_state=42)
    lr.fit(X_train_features, Y_train)
    rf.fit(X_train_features, Y_train)
    return lr, rf

def evaluate_model(model_func, X_test, Y_test, threshold=0.6):
    if hasattr(model_func, 'predict_proba'):
        probs = model_func.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        sel_preds = np.full(len(probs), -1, dtype=int)
        sel_preds[probs >= threshold] = 1
        sel_preds[probs <= (1 - threshold)] = 0
        mask = sel_preds != -1
        overall_acc = accuracy_score(Y_test, preds) if len(Y_test) > 0 else 0.0
        if mask.sum() > 0:
            sel_acc = accuracy_score(Y_test[mask], sel_preds[mask])
            coverage = mask.sum() / len(Y_test)
        else:
            sel_acc = 0.0
            coverage = 0.0
        return {
            'overall_acc': overall_acc,
            'sel_acc': sel_acc,
            'coverage': coverage,
            'avg_prob': probs.mean()
        }
    else:
        preds, cal_probs, _ = model_func(X_test, threshold)
        mask = preds != -1
        overall_preds = (cal_probs >= 0.5).astype(int)
        overall_acc = accuracy_score(Y_test, overall_preds) if len(Y_test) > 0 else 0.0
        if mask.sum() > 0:
            sel_acc = accuracy_score(Y_test[mask], preds[mask])
            coverage = mask.sum() / len(Y_test)
        else:
            sel_acc = 0.0
            coverage = 0.0
        return {
            'overall_acc': overall_acc,
            'sel_acc': sel_acc,
            'coverage': coverage,
            'avg_prob': cal_probs.mean()
        }

def main():
    train_equity = ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMD', 'INTC']
    train_start = '2017-01-01'
    train_end = '2023-12-31'
    test_start = '2024-01-01'
    test_end = '2026-09-01'
    feature_cols = ['ret_1d', 'ret_3d', 'vol_20d', 'ma_5d', 'rsi', 'vol_chg',
                    'vix', 'dxy', 'tnx_10y', 'vix_chg', 'dxy_chg', 'tnx_chg']
    window_size = 20
    confidence_threshold = 0.6
    model_bundle_path = 'model_bundle.pt'

    test_tickers = {
        'Equity': ['TSLA', 'AMZN', 'BA', 'JPM', 'NKE', 'WMT', 'MCD', 'PG', 'KO', 'HD'],
        'Crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD'],
        'Commodity': ['GLD', 'SLV', 'USO', 'DBC', 'WEAT'],
        'Currency': ['UUP', 'FXE', 'FXY', 'FXB', 'FXA'],
        'Bond': ['TLT', 'TIP', 'IEF', 'SHY', 'BND']
    }

    print("Loading HPT model bundle...")
    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError("model_bundle.pt not found. Please train HPT first.")
    hpt_model = StrictEquityEnsemble.load(model_bundle_path)
    scaler = hpt_model.scaler

    print("Preparing training data for baseline models...")
    macro_train = fetch_macro(train_start, train_end)
    X_train_list = []
    Y_train_list = []
    for ticker in train_equity:
        df = build_features(fetch_data(ticker, train_start, train_end), macro_train)
        if df.empty:
            continue
        n = len(df)
        split = int(n * 0.8)
        df_train = df.iloc[:split]
        X_raw = df_train[feature_cols].values
        Y = df_train['target'].values
        X_train_list.append(X_raw)
        Y_train_list.append(Y)
    X_train_all = np.vstack(X_train_list)
    Y_train_all = np.concatenate(Y_train_list)
    X_train_scaled = scaler.transform(X_train_all)

    print("Training Logistic Regression baseline...")
    lr, rf = train_baselines(X_train_scaled, Y_train_all)

    print("Fetching macro data for test period...")
    macro_test = fetch_macro(test_start, test_end)

    results = []
    for class_name, tickers in test_tickers.items():
        for ticker in tickers:
            df = build_features(fetch_data(ticker, test_start, test_end), macro_test)
            if df.empty:
                print(f"{ticker}: No data, skipping.")
                continue
            test_seg = df.tail(100)
            if len(test_seg) < window_size + 1:
                print(f"{ticker}: Not enough data (needs >{window_size} rows), skipping.")
                continue
            X_test_raw = test_seg[feature_cols].values
            Y_test = test_seg['target'].values[window_size:]
            if len(Y_test) == 0:
                print(f"{ticker}: Not enough data after windowing, skipping.")
                continue
            X_test_scaled = scaler.transform(X_test_raw)
            # For baselines, we need to align with windowed labels:
            X_test_scaled_aligned = X_test_scaled[window_size:]
            # HPT model handles windowing internally, so pass full raw
            hpt_res = evaluate_model(hpt_model.predict, X_test_raw, Y_test, confidence_threshold)
            lr_res = evaluate_model(lr, X_test_scaled_aligned, Y_test, confidence_threshold)
            rf_res = evaluate_model(rf, X_test_scaled_aligned, Y_test, confidence_threshold)
            results.append({
                'Ticker': ticker,
                'Class': class_name,
                'HPT_SelAcc': hpt_res['sel_acc'],
                'HPT_Cov': hpt_res['coverage'],
                'HPT_OverallAcc': hpt_res['overall_acc'],
                'LR_SelAcc': lr_res['sel_acc'],
                'LR_Cov': lr_res['coverage'],
                'LR_OverallAcc': lr_res['overall_acc'],
                'RF_SelAcc': rf_res['sel_acc'],
                'RF_Cov': rf_res['coverage'],
                'RF_OverallAcc': rf_res['overall_acc'],
            })

    df_results = pd.DataFrame(results)
    print("\n" + "="*100)
    print("BENCHMARK RESULTS (Selective accuracy at threshold 0.6)")
    print("="*100)
    print(df_results.to_string(index=False))

    print("\n" + "="*100)
    print("AVERAGE PER ASSET CLASS")
    print("="*100)
    avg_df = df_results.groupby('Class').agg({
        'HPT_SelAcc': 'mean',
        'HPT_Cov': 'mean',
        'HPT_OverallAcc': 'mean',
        'LR_SelAcc': 'mean',
        'LR_Cov': 'mean',
        'LR_OverallAcc': 'mean',
        'RF_SelAcc': 'mean',
        'RF_Cov': 'mean',
        'RF_OverallAcc': 'mean'
    }).round(4)
    print(avg_df)

    df_results.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to benchmark_results.csv")

if __name__ == '__main__':
    main()