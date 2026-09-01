# HPT–DTE-CAFP: Mathematical and Implementation Guide

## 1. Purpose and model overview

This document explains how the implementation in `main.py` becomes a mathematical forecasting pipeline. The model, HPT–DTE, combines a Harmonic Phase Transformer with two decision trees, a logistic meta-learner, isotonic probability calibration, and a selective decision rule. Its task is binary next-day direction prediction:

> Given information available at the close of day $$t$$, estimate the probability that the next closing price is higher than the current closing price.

For asset $$a$$, define the label

$$
y_{a,t+1}=\mathbf{1}\{C_{a,t+1}>C_{a,t}\},
$$

where $$C_{a,t}$$ is the closing price and $$\mathbf{1}\{\cdot\}$$ is the indicator function. The model does not predict a return magnitude. It predicts a Bernoulli probability and then decides whether to act or abstain.

The complete pipeline is

$$
\text{market data}
\rightarrow \text{causal features}
\rightarrow \text{standardization}
\rightarrow \text{20-day windows}
\rightarrow \text{time/FFT representations}
\rightarrow \text{HPT probability}
\rightarrow \text{tree signals}
\rightarrow \text{meta probability}
\rightarrow \text{isotonic calibration}
\rightarrow \text{selective decision}.
$$

The implementation uses nine equities for training and evaluates zero-shot on unseen equities, cryptoassets, and GLD. It is an academic prototype and does not include transaction costs, slippage, market impact, portfolio constraints, or live-trading risk controls.

## 2. Data and feature construction

### 2.1 Price, volume, and macro data

The code downloads daily observations using `yfinance`. For each asset, it uses close prices and volume. It also downloads three macro series:

| Code variable | Series | Interpretation |
| --- | --- | --- |
| `vix` | VIX close | Implied-volatility or market-stress proxy |
| `dxy` | DXY close | U.S. dollar index proxy |
| `tnx_10y` | TNX close | Ten-year Treasury-yield proxy |

Macro series are joined to the asset calendar and forward-filled. Forward-filling means that the most recently observed macro value is carried forward until the next observation. This is intended to avoid using a value that was unavailable at the forecast origin.

### 2.2 Return features

The one-day and three-day returns are

$$
r_t^{(1)}=\frac{C_t}{C_{t-1}}-1,
\qquad
r_t^{(3)}=\frac{C_t}{C_{t-3}}-1.
$$

These variables describe short-horizon momentum and reversal information. A positive value means the current close is above the close one or three sessions earlier.

### 2.3 Rolling volatility

The code computes the standard deviation of the previous 20 one-day returns:

$$
\sigma_{20,t}=\operatorname{sd}\left(r_{t-19}^{(1)},\ldots,r_t^{(1)}\right).
$$

This feature measures recent variability. It is backward-looking because it uses only returns at or before $$t$$.

### 2.4 Moving-average ratio

The moving-average feature is

$$
\operatorname{MA5Ratio}_t
=\frac{\frac{1}{5}\sum_{j=0}^{4}C_{t-j}}{C_t}.
$$

A value above one means the five-day average is above the current price; a value below one means the current price is above the recent average.

### 2.5 Relative Strength Index

Let $$\Delta C_t=C_t-C_{t-1}$$. Define positive and negative changes as

$$
U_t=\max(\Delta C_t,0),
\qquad
D_t=\max(-\Delta C_t,0).
$$

The implementation uses 14-period rolling means and computes

$$
\operatorname{RSI}_t
=100-\frac{100}{1+
\frac{\operatorname{MA}_{14}(U_t)}{\operatorname{MA}_{14}(D_t)+10^{-9}}}.
$$

The small constant prevents division by zero. RSI is bounded approximately between 0 and 100 and summarizes the balance between recent upward and downward movements.

### 2.6 Volume and macro changes

Volume change is

$$
\Delta V_t^{\%}=\frac{V_t}{V_{t-1}}-1.
$$

For each macro series $$M_t\in\{\mathrm{VIX}_t,\mathrm{DXY}_t,\mathrm{TNX}_t\}$$, the code includes its level and one-day percentage change:

$$
\Delta M_t^{\%}=\frac{M_t}{M_{t-1}}-1.
$$

The resulting twelve model features are

$$
\left[
 r_t^{(1)},r_t^{(3)},\sigma_{20,t},\operatorname{MA5Ratio}_t,
 \operatorname{RSI}_t,\Delta V_t^{\%},
 \mathrm{VIX}_t,\mathrm{DXY}_t,\mathrm{TNX}_t,
 \Delta\mathrm{VIX}_t^{\%},\Delta\mathrm{DXY}_t^{\%},
 \Delta\mathrm{TNX}_t^{\%}
\right].
$$

### 2.7 Standardization

The `StandardScaler` transforms feature $$j$$ according to

$$
\widetilde{x}_{t,j}=\frac{x_{t,j}-\mu_j}{\sigma_j+\epsilon},
$$

where $$\mu_j$$ and $$\sigma_j$$ are the estimated mean and standard deviation and $$\epsilon$$ is a numerical safeguard. Standardization is necessary because RSI, returns, volatility, and macro levels have very different numerical scales.

The code fits the scaler on the concatenated training-period matrix before the chronological split. This does not use the later test period, but a stricter experimental protocol would fit preprocessing only on the first training fold.

## 3. Window construction and tensor dimensions

Let $$w=20$$ and $$d=12$$. For a standardized feature matrix $$S\in\mathbb{R}^{n\times12}$$, the code creates the window

$$
X_i=S_{i-w:i}\in\mathbb{R}^{20\times12}.
$$

The target aligned with this window is $$y_i$$, the direction observed after the final row of the window. Because the first 20 rows are needed to create the first window, $$n$$ rows yield $$n-20$$ windows.

For batch size $$B$$, the time-domain input is therefore

$$
X_{\mathrm{time}}\in\mathbb{R}^{B\times20\times12}.
$$

The implementation applies the real FFT independently along the time axis for each of the twelve features. A real FFT of length 20 has

$$
\frac{20}{2}+1=11
$$

frequency bins. Each frequency bin contains a real and imaginary coefficient for each feature, giving

$$
X_{\mathrm{freq}}\in\mathbb{R}^{B\times11\times24}.
$$

The static feature vector used by the trees is aligned to the forecast origin and has shape

$$
X_{\mathrm{stat}}\in\mathbb{R}^{B\times12}.
$$

## 4. Time-domain branch

The code component is `time_proj = nn.Linear(n_features, d_model)` with $$d_m=32$$. For each timestep, it computes

$$
H_t=X_{\mathrm{time}}W_t+b_t,
$$

where $$W_t\in\mathbb{R}^{12\times32}$$. Thus

$$
H_t\in\mathbb{R}^{B\times20\times32}.
$$

The model adds a learnable parameter named `phase_shifts` with shape $$1\times20\times32$$:

$$
H_t^{\phi}=H_t+\Phi.
$$

The parameter $$\Phi$$ is initialized from a small normal distribution with standard deviation 0.05. It is called a phase shift because it supplies a learned offset at each relative position and channel. Mathematically, it is a trainable positional bias rather than a literal physical phase measurement.

## 5. Frequency-domain branch

For each feature dimension $$j$$, the length-20 discrete Fourier transform is conceptually

$$
\widehat{x}_{k,j}=\sum_{n=0}^{19}x_{n,j}
\exp\left(-2\pi i\frac{kn}{20}\right),
\qquad k=0,\ldots,19.
$$

Because the input is real, the negative-frequency coefficients are redundant. `np.fft.rfft` retains the 11 non-redundant bins. Each coefficient has a real component and an imaginary component:

$$
\widehat{x}_{k,j}=a_{k,j}+ib_{k,j}.
$$

The code concatenates $$a_{k,j}$$ and $$b_{k,j}$$ across features, so each frequency token has dimension 24. The linear layer `freq_proj` maps this vector into 32 dimensions:

$$
H_f=[\Re(\widehat X)\Vert\Im(\widehat X)]W_f+b_f,
$$

with

$$
H_f\in\mathbb{R}^{B\times11\times32}.
$$

The magnitude and phase of a complex coefficient are related to its real and imaginary parts by

$$
|\widehat{x}_{k,j}|=\sqrt{a_{k,j}^2+b_{k,j}^2},
\qquad
\arg(\widehat{x}_{k,j})=\operatorname{atan2}(b_{k,j},a_{k,j}).
$$

The code does not explicitly calculate magnitude or angle; it gives the neural network both components so that a learned projection can form useful combinations.

## 6. Cross-domain multi-head attention

The key architectural operation is `nn.MultiheadAttention`. The time representation supplies queries, and the frequency representation supplies keys and values. This is cross-attention rather than self-attention because the two sequences come from different representations.

For four heads, the head dimension is

$$
 d_h=\frac{d_m}{4}=8.
$$

For head $$h$$, the projections are

$$
Q_h=H_t^{\phi}W_h^Q,
\qquad
K_h=H_fW_h^K,
\qquad
V_h=H_fW_h^V.
$$

The attention weights are

$$
P_h=\operatorname{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{8}}\right).
$$

The head output is

$$
A_h=P_hV_h.
$$

The four head outputs are concatenated and projected back to 32 channels. Each time position can therefore select frequency tokens according to content similarity. The model is not forced to treat every frequency component as equally relevant.

## 7. Gated residual fusion

The code defines

```python
self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
```

For the combined attention output $$A$$, the gate is

$$
G=\sigma(AW_g+b_g),
$$

where $$G\in(0,1)^{B\times20\times32}$$. The gated attention is

$$
\widetilde A=G\odot A,
$$

where $$\odot$$ denotes elementwise multiplication. The residual fusion is

$$
H=\operatorname{LayerNorm}(H_t^{\phi}+\widetilde A).
$$

A simple residual would always add the entire attention output. The gate allows each position and channel to suppress or admit the harmonic signal. The sigmoid does not prove that the selected frequency information is useful; it only provides a learnable soft control mechanism.

## 8. Feed-forward block

The code uses

```python
nn.Sequential(
    nn.Linear(32, 128),
    nn.GELU(),
    nn.Linear(128, 32)
)
```

The Gaussian Error Linear Unit is commonly written as

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

where $$\Phi$$ is the standard normal cumulative distribution function. The feed-forward transformation is

$$
\operatorname{FFN}(H)=W_2\operatorname{GELU}(W_1H+b_1)+b_2,
$$

with hidden width 128. The code adds this transformation residually:

$$
H^+=H+\operatorname{FFN}(H).
$$

The attention block mixes information across domains; the feed-forward block applies a channel-wise nonlinear transformation at each position.

## 9. Decision-tree regime conditioning

The regime tree is

```python
DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
```

A decision tree recursively partitions feature space with rules of the form

$$
 x_j\le c
$$

until an observation reaches a leaf. Each leaf corresponds to a region of the standardized feature space. The tree's leaf index is converted into a one-hot vector $$e_\ell$$. The code pads or truncates this vector to dimension 20 and maps it into model dimension 32:

$$
 h_c=W_ce_\ell+b_c\in\mathbb{R}^{32}.
$$

It then broadcasts the condition across the 20 positions:

$$
 H_c=\mathbf{1}_{20}h_c^\top\in\mathbb{R}^{20\times32}.
$$

The condition is not used as a replacement for the sequence. It is contextual information describing the region of feature space in which the current forecast origin lies.

## 10. HPT probability head

The code mean-pools over the time dimension:

$$
\bar h=\frac{1}{20}\sum_{r=1}^{20}H_r^+,
\qquad
\bar h_c=\frac{1}{20}\sum_{r=1}^{20}H_{c,r}.
$$

It concatenates both vectors:

$$
 u=[\bar h\Vert\bar h_c]\in\mathbb{R}^{64}.
$$

The output head is a 64-to-32-to-1 multilayer perceptron:

$$
 p_{\mathrm{HPT}}
 =\sigma\left(w_o^\top\operatorname{GELU}(W_ou+b_o)+b\right).
$$

The final sigmoid converts the scalar logit into a number in $$(0,1)$$ that can be interpreted as a raw estimate of $$P(Y=1\mid X)$$, although it is not guaranteed to be calibrated.

## 11. Equity tree and meta-learner

The equity tree is

```python
DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, random_state=42)
```

It produces a second estimate, $$p_{\mathrm{eq}}$$, from the static feature vector. The meta-feature vector is

$$
 z=\begin{bmatrix}p_{\mathrm{HPT}}\\p_{\mathrm{eq}}\\\ell\end{bmatrix}.
$$

The logistic meta-learner computes

$$
 p_{\mathrm{meta}}=\sigma(\beta_0+\beta^\top z).
$$

Its objective is regularized negative log-likelihood:

$$
\min_{\beta_0,\beta}
-\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
+\lambda\|\beta\|_2^2.
$$

In scikit-learn, `C=0.1` corresponds to relatively strong regularization because $$C$$ is the inverse regularization-strength parameter. The use of the numeric leaf index introduces an ordering among leaves. A one-hot leaf representation in the meta-learner would remove that implicit ordering and should be tested in future work.

## 12. Neural optimization

The HPT is trained with binary cross-entropy:

$$
\mathcal L_{\mathrm{BCE}}
=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

The code uses AdamW with learning rate $$\eta=10^{-3}$$ and weight decay $$\lambda=10^{-4}$$. In simplified form, the update is

$$
\theta_{k+1}
=\theta_k
-\eta\frac{\widehat m_k}{\sqrt{\widehat v_k}+\epsilon}
-\eta\lambda\theta_k,
$$

where $$\widehat m_k$$ and $$\widehat v_k$$ are bias-corrected first- and second-moment estimates. AdamW separates weight decay from the adaptive gradient step.

The data loader uses batch size 256 and `shuffle=False`. Thirty epochs are run. The reported loss declines from 0.6605 at epoch 5 to 0.4724 at epoch 30. A decreasing training loss indicates optimization progress, but it does not establish out-of-sample predictive validity.

## 13. Isotonic probability calibration

A classifier can rank examples correctly while producing probabilities that are too extreme. Let $$s_i=p_{\mathrm{meta},i}$$ be the meta-learner score on validation example $$i$$. Isotonic regression estimates a non-decreasing function $$g$$ by

$$
 g^*=\arg\min_{g\text{ non-decreasing}}
 \sum_{i=1}^{n}(g(s_i)-y_i)^2.
$$

The calibrated probability is

$$
 p_{\mathrm{cal}}=g^*(p_{\mathrm{meta}}).
$$

The pool-adjacent-violators algorithm sorts the scores, starts with one block per observation, and merges adjacent blocks whenever their fitted means violate monotonicity. For a merged block $$B$$, the fitted value is

$$
\widehat g_B=\frac{1}{|B|}\sum_{i\in B}y_i.
$$

### Proposition: empirical optimality

Among all non-decreasing functions evaluated on the observed validation scores, the isotonic solution minimizes empirical squared error.

**Reason.** After sorting the distinct scores, the problem becomes a convex quadratic minimization subject to ordered constraints $$u_1\le\cdots\le u_m$$. The KKT conditions require violating adjacent fitted blocks to be pooled, and the least-squares value of a pooled block is its label mean. PAV constructs exactly this constrained minimizer.

A strictly increasing calibration function preserves all rankings and therefore preserves AUC. A merely non-decreasing function can pool different scores into ties. Consequently, isotonic calibration cannot reverse strict rankings, but it can slightly change AUC through additional ties. Calibration changes probability interpretation and threshold selection; it does not create information that was absent from the raw score.

## 14. Selective prediction and abstention

The implementation uses confidence threshold $$\tau=0.6$$:

$$
\widehat y=
\begin{cases}
1,&p_{\mathrm{cal}}\ge0.6,\\
0,&p_{\mathrm{cal}}\le0.4,\\
\bot,&0.4<p_{\mathrm{cal}}<0.6.
\end{cases}
$$

The symbol $$\bot$$ means abstention. Coverage is

$$
\kappa=P(\widehat y\ne\bot).
$$

Selective accuracy is

$$
\operatorname{Acc}_{\mathrm{sel}}
=P(\widehat y=Y\mid\widehat y\ne\bot).
$$

### Theorem: population accuracy under perfect calibration

If $$p(X)=P(Y=1\mid X)$$ is perfectly calibrated and the policy acts only when $$p(X)\ge\tau$$ or $$p(X)\le1-\tau$$, then

$$
\operatorname{Acc}_{\mathrm{sel}}\ge\tau.
$$

**Proof.** If $$p(X)\ge\tau$$, the policy predicts one and is correct with probability $$p(X)\ge\tau$$. If $$p(X)\le1-\tau$$, it predicts zero and is correct with probability $$1-p(X)\ge\tau$$. Conditioning on the selected set and taking expectations preserves the lower bound. In a finite sample, estimated calibration error and sampling variation mean the observed accuracy can fall below the nominal threshold.

The theorem explains the intended trade-off: increasing $$\tau$$ can increase the quality of selected calls while reducing coverage. It does not guarantee profitability, because accuracy ignores payoff asymmetry, transaction costs, position sizing, and execution.

## 15. Training chronology and known implementation caveats

The intended chronology is:

1. Train on 2017--2023 equity data.

1. Split the training period chronologically into 80% and 20% blocks.

1. Fit trees and HPT on the first block.

1. Fit the meta-learner and isotonic calibrator on the held-out block.

1. Evaluate once on unseen assets and the later 2024--2026 period.

The code provides useful safeguards, including fixed seeds, `shuffle=False`, and a single-file bundle containing the scaler, trees, neural weights, meta-learner, and calibrator. However, several claims should be interpreted carefully:

| Issue | Why it matters | Recommended repair |
| --- | --- | --- |
| Windows are built after `np.vstack` in `fit()` | Boundary windows can cross from one ticker into another | Build and concatenate windows separately for each ticker |
| Scaler is fitted before the 80/20 split | Validation statistics influence preprocessing | Fit scaler only on the model-fitting fold |
| Meta-learner and calibrator share validation data | Calibration is not independently evaluated | Add a separate calibration fold or cross-fitting |
| `verify_no_leakage()` prints a count | It does not formally inspect causal dependencies | Add automated feature-timestamp and dependency tests |
| One random seed and one split | Results may be split-sensitive | Repeat blocked splits and report intervals |
| Selective accuracy only | It omits abstained days and economic outcomes | Report coverage, forced-choice accuracy, returns, drawdown, and costs |

## 16. Summary of the mathematical roles

| Component | Mathematical role | Output |
| --- | --- | --- |
| Feature engineering | Creates causal covariates and future direction label | $$x_t$$, $$y_{t+1}$$ |
| StandardScaler | Affine normalization | $$\widetilde x_t$$ |
| Window builder | Creates local sequences | $$20\times12$$ tensor |
| Real FFT | Changes basis from time to frequency | $$11\times24$$ tensor |
| Time projection | Embeds temporal observations | $$20\times32$$ |
| Phase shifts | Adds learned relative-position bias | $$20\times32$$ |
| Frequency projection | Embeds complex FFT components | $$11\times32$$ |
| Cross-attention | Selects relevant frequency tokens for time positions | $$20\times32$$ |
| Gate and residual | Controls harmonic information flow | $$20\times32$$ |
| FFN | Applies nonlinear channel transformation | $$20\times32$$ |
| Regime tree | Produces a discrete market-state condition | one-hot leaf vector |
| Equity tree | Produces an independent probability | $$p_{\mathrm{eq}}$$ |
| Logistic meta-learner | Fuses model signals | $$p_{\mathrm{meta}}$$ |
| Isotonic regression | Monotone probability recalibration | $$p_{\mathrm{cal}}$$ |
| Threshold policy | Converts probability into action or abstention | $$1$$, $$0$$, or $$\bot$$ |

The principal conceptual distinction is between **representation**, **probability estimation**, **calibration**, and **decision**. The Transformer and trees form representations and raw scores. The meta-learner combines scores. Isotonic regression improves their probability interpretation on held-out data. The threshold policy decides whether the calibrated probability is sufficiently far from one-half to justify a prediction. None of these stages, by itself, proves that an economic trading edge exists.

## References

1. Repository implementation and documentation: [HPT-DTE-CAFP](https://github.com/ak495867/HPT-DTE-CAFP).

1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

1. Pedregosa et al., [Scikit-learn: Machine Learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html).

1. Niculescu-Mizil and Caruana, [Predicting Good Probabilities with Supervised Learning](https://dl.acm.org/doi/10.1145/1102351.1102430).