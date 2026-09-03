# Harmonic Phase Transformer for Financial Prediction

[![Paper](https://img.shields.io/badge/PAPER-HPT--DTE-000000?style=for-the-badge&logo=academia&logoColor=white)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7389258)
[![SSRN](https://img.shields.io/badge/SSRN-7389258-000000?style=for-the-badge)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7389258)
[![Release](https://img.shields.io/github/v/release/ak495867/HPT-DTE-CAFP?style=for-the-badge)](https://github.com/ak495867/HPT-DTE-CAFP/releases)
[![License](https://img.shields.io/github/license/ak495867/HPT-DTE-CAFP?style=for-the-badge)](https://github.com/ak495867/HPT-DTE-CAFP/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/ak495867/HPT-DTE-CAFP?style=for-the-badge)](https://github.com/ak495867/HPT-DTE-CAFP/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ak495867/HPT-DTE-CAFP?style=for-the-badge)](https://github.com/ak495867/HPT-DTE-CAFP/network/members)
[![GitHub issues](https://img.shields.io/github/issues/ak495867/HPT-DTE-CAFP?style=for-the-badge)](https://github.com/ak495867/HPT-DTE-CAFP/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/ak495867/HPT-DTE-CAFP?style=for-the-badge)](https://github.com/ak495867/HPT-DTE-CAFP/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/ak495867/HPT-DTE-CAFP?style=for-the-badge)](https://github.com/ak495867/HPT-DTE-CAFP)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![yfinance](https://img.shields.io/badge/yfinance-0B0B0B?style=for-the-badge)](https://github.com/ranaroussi/yfinance)

> **⚠️ IMPORTANT: v2.0-hotfix Update**
> 
> This repository has been updated to **v2.0-hotfix**, which fixes critical data leakage issues:
> - Windows are now built **independently per asset** (no cross-ticker contamination)
> - `StandardScaler` is fitted **exclusively on the training split** (no validation leakage)
> - Results are now **honest and reproducible**
> 
> **The results in this README reflect the fixed pipeline.** Earlier versions had inflated accuracies due to leakage and should not be used or cited.

>  RESEARCH ONLY. NOT FINANCIAL ADVICE. See [Disclaimer](#disclaimer) below.
> ## Mathematical Details
> See **[MATH.md](MATH.md)** for the complete mathematical foundations and derivations.


## Research Paper

- [Read the research paper (PDF)](paper/HPT-DTE-CAFP.pdf)
- [View the LaTeX source](paper/HPT-DTE-CAFP.tex)

## What Is This?

A hybrid ML model combining **Harmonic Phase Transformers** (time + frequency domain attention) with **Decision Tree ensembles** and a **Random Forest meta-learner** for cross-asset financial prediction. The architecture includes isotonic calibration and a selective abstention mechanism, and is evaluated under a strict leakage-free protocol. Trained on equities, tested zero-shot on crypto, commodities, and unseen stocks.

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

| Asset   | Type   | Acc. (Selected) | Trades | Coverage | Avg Cal. |
|---------|--------|----------------|--------|----------|----------|
| TSLA    | Equity | 37.5%          | 8/80   | 10.0%    | 0.525    |
| AMZN    | Equity | 87.5%          | 8/80   | 10.0%    | 0.530    |
| BA      | Equity | 60.0%          | 5/80   | 6.3%     | 0.526    |
| JPM     | Equity | 50.0%          | 8/80   | 10.0%    | 0.519    |
| BTC-USD | Crypto | 25.0%          | 12/80  | 15.0%    | 0.531    |
| ETH-USD | Crypto | 100.0%         | 2/80   | 2.5%     | 0.522    |
| SOL-USD | Crypto | 50.0%          | 12/80  | 15.0%    | 0.538    |
| GLD     | Equity | 53.3%          | 15/80  | 18.8%    | 0.546    |

Trained **only** on 9 equities (2017-2023). Never saw test assets or test period.

## Results Interpretation

The fixed leakage-free pipeline yields **low coverage (2.5%–18.8%)**, meaning the model abstains on most days. This is a direct consequence of:
- The high confidence threshold (τ = 0.6)
- The model's inability to produce probabilities far from 0.5 on unseen assets

The primary contribution is therefore **methodological**: a clean, reproducible, and mathematically grounded framework for evaluating financial direction prediction under strict leakage-free conditions.

**Key takeaway:** Zero-shot cross-asset prediction remains a challenging task. The value of this work is in the **architecture** and the **evaluation protocol**, not in claiming a trading edge.

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
