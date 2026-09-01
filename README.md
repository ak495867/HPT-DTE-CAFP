# Harmonic Phase Transformer for Financial Prediction

>  RESEARCH ONLY. NOT FINANCIAL ADVICE. See [Disclaimer](#disclaimer) below.
> ## Mathematical Details
> See **[MATH.md](MATH.md)** for the complete mathematical foundations and derivations.


## Research Paper

- [Read the research paper (PDF)](paper/HPT-DTE-CAFP.pdf)
- [View the LaTeX source](paper/HPT-DTE-CAFP.tex)

## What Is This?

A hybrid ML model combining **Harmonic Phase Transformers** (time + frequency domain attention) with **Decision Tree ensembles** for cross-asset financial prediction. Trained on equities, tested zero-shot on crypto, commodities, and unseen stocks.

## Architecture

```
Input (20-day window, 12 features)
  ├── Time Branch → Phase-shifted projection
  ├── Freq Branch → FFT (real + imag) projection
  └── Cross-Attention (time queries freq)
        ↓
  Gated Residual → LayerNorm → FFN
        ↓
  Condition Injection (Decision Tree regime)
        ↓
  Meta-Learner → Isotonic Calibration → Prediction (Buy/Sell/Hold)
```

## Quick Start

```bash
pip install yfinance pandas numpy torch scikit-learn
python main.py
```

First run trains and saves `model_bundle.pt`. Subsequent runs load instantly.

## Results (Zero-Shot, 2024-2026)

| Asset   | Type   | Accuracy | Trades |
|---------|--------|----------|--------|
| TSLA    | Equity | 60.0%    | 30/80  |
| ETH-USD | Crypto | 53.6%    | 28/80  |
| BTC-USD | Crypto | 52.5%    | 40/80  |
| GLD     | ETF    | 52.2%    | 46/80  |

Trained **only** on 9 equities (2017-2023). Never saw test assets or test period.

## Key Features

- **Dual-domain attention**: Time + frequency (FFT) cross-attention
- **Learnable phase shifts**: Captures temporal pattern offsets
- **Macro-aware**: VIX, DXY, 10Y Treasury as features
- **Calibrated probabilities**: Isotonic regression prevents overconfidence
- **Selective trading**: Only predicts when confident (Buy/Sell/Hold)
- **Strict validation**: No data leakage, per-ticker windows, temporal splits
- **Single-file model**: Everything saved in one `.pt` bundle

## Project Structure

```
├── main.py              # Full training + evaluation pipeline
├── model_bundle.pt      # Trained model (generated on first run)
└── README.md            # This file
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- scikit-learn 1.3+
- yfinance 0.2+
- pandas, numpy

---

## Disclaimer

**READ BEFORE USING**

*   **Research Only:** This code is an experimental machine learning prototype for academic and educational purposes only. It is **not** intended for live trading.
*   **Not Financial Advice:** Nothing herein constitutes financial, investment, or trading advice. Do not use model outputs to make real-world financial decisions.
*   **No Guarantees:** Past performance does not guarantee future results. This model does not account for real-world trading costs (fees, slippage, spread).
*   **No Liability:** This software is provided "AS IS". The authors assume **zero liability** for any financial losses or damages arising from its use. 

**YOU USE THIS CODE ENTIRELY AT YOUR OWN RISK.** If you do not agree, do not use this software.

---

*Built as an experimental research project exploring cross-domain transformers for financial time series.*