# Model Architecture Comparison

This document provides a mathematical and architectural comparison of the three models evaluated in the benchmark: **HPT‑DTE**, **Logistic Regression (LR)**, and **Random Forest (RF)**. All models are trained on the same nine equities (2017‑2023) and evaluated zero‑shot on 35 unseen assets (2024‑2026). The comparison focuses on model complexity, expressiveness, calibration behaviour, and selective‑prediction capability.

---

## 1. Overview of Architectures

### 1.1 Logistic Regression (LR)

LR is a linear classifier that models the log‑odds as an affine function of the input features.

**Formulation:**
For input vector $x \in \mathbb{R}^{d}$ (d = 12 features), the probability of the positive class is

$$
p(y=1 \mid x) = \sigma(w^\top x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}.
$$

**Training:** Maximum likelihood (binary cross‑entropy) with L2 regularisation.

**Decision rule (with abstention):**
For confidence threshold $\tau$ (e.g., 0.6), a trade is made only if $p \ge \tau$ or $p \le 1-\tau$; otherwise abstain. In the standard (non‑selective) setting, threshold is 0.5.

---

### 1.2 Random Forest (RF)

RF is an ensemble of $T$ decision trees, each trained on a bootstrap sample of the data and using a random subset of features at each split.

**Formulation:**
Each tree $t$ produces a leaf‑wise constant estimate $\hat{p}_t(x)$. The ensemble prediction is the average:

$$
p_{\text{RF}}(x) = \frac{1}{T} \sum_{t=1}^{T} \hat{p}_t(x).
$$

**Complexity:** Each tree has $O(2^D)$ nodes where $D$ is the maximum depth. Training involves $O(T \cdot N \cdot d \cdot D)$ comparisons. Inference is $O(T \cdot D)$.

**Calibration:** RF probabilities are often miscalibrated; they tend to be overconfident for samples in dense regions and underconfident elsewhere. No explicit calibration is applied in this benchmark; we directly threshold the raw averaged probabilities.

---

### 1.3 HPT‑DTE (Proposed)

HPT‑DTE is a hybrid architecture that combines a neural dual‑domain Transformer, two decision trees (regime and equity), a random‑forest meta‑learner, and isotonic calibration.

**Key components:**

1. **Time‑domain representation:** A 20‑step window $X \in \mathbb{R}^{20 \times 12}$ is projected via a learnable linear layer and added with a learnable phase‑shift tensor $\Phi$ (size $1 \times 20 \times 32$), giving $H_t \in \mathbb{R}^{20 \times 32}$.

2. **Frequency‑domain representation:** The real FFT of the window yields real and imaginary parts. The code uses `rfft(w, axis=0)`, which gives 11 complex bins for a 20‑step window (since 20 real values → 11 complex bins). For each of the 12 features this yields (11, 12) complex values. `np.concatenate([fft_out.real, fft_out.imag], axis=-1)` then produces an $11 \times 24$ tensor — so the frequency branch input is $11 \times 24$, not $20 \times 24$.

3. **Cross‑attention:** Multi‑head attention (4 heads, $d_{\text{head}}=8$) where queries come from time tokens ($20 \times 32$) and keys/values from frequency tokens ($11 \times 32$). Output is $20 \times 32$.

4. **Gating and residual:** A channel‑wise sigmoid gate controls the attention output; residual connection and layer norm follow.

5. **Regime and equity trees:** Two shallow decision trees (max depth 4 and 5) produce a leaf index (one‑hot) and an equity probability, respectively. The leaf index is encoded to a 20‑dim condition vector; equity tree gives a scalar probability.

6. **Meta‑learner:** A random forest (100 trees, depth 4) fuses the three signals: HPT probability, equity tree probability, and regime leaf index (as a float). This yields $p_{\text{meta}}$.

7. **Isotonic calibration:** A non‑decreasing piecewise‑constant map $g$ is fitted on validation data to minimize squared error, producing $p_{\text{cal}} = g(p_{\text{meta}})$.

8. **Selective decision:** As with the baselines, threshold $\tau = 0.6$ is used; abstain if $p_{\text{cal}} \in (1-\tau, \tau)$.

---

## 2. Model Complexity and Parameters

### 2.1 Parameter Count

- **LR:** $d+1 = 13$ parameters. (Weight vector of size 12 plus bias.)

- **RF:** Parameter count depends on the number and depth of trees. With $T=100$, max depth 5, each tree has at most $2^{6}-1 = 63$ nodes (full tree), so ~6,300 nodes in total. However, actual leaf nodes are fewer due to minimum leaf size. Each node stores a split feature and threshold (2 parameters) and the class probability. In practice, scikit‑learn stores the tree structure as arrays of size ~nodes × (feature, threshold, children, etc.). Memory is roughly $O(T \cdot 2^D)$.

- **HPT‑DTE:** The neural component has:
  - `time_proj`: $12 \times 32 = 384$ weights + 32 bias
  - `freq_proj`: $24 \times 32 = 768$ + 32 bias
  - `phase_shifts`: $1 \times 20 \times 32 = 640$ trainable parameters
  - `cross_attn`: 4 heads, each with Q, K, V projections of $32 \times 8 = 256$ each (768 per head), plus output projection $32 \times 32 = 1024$; total $\approx 4 \times 768 + 1024 = 4096$
  - `gate`: linear $32 \to 32$ ($1024 + 32$)
  - `ffn`: $32 \to 128$ ($4096 + 128$) and $128 \to 32$ ($4096 + 32$) = 8,352
  - `cond_map`: $20 \to 32$ ($640 + 32$)
  - `head`: $64 \to 32$ ($2048 + 32$) and $32 \to 1$ ($32 + 1$) = 2,113
  - Total neural parameters $\approx 384+32+768+32+640+4096+1024+32+8352+640+32+2113 \approx 17{,}745$ (approximate; some bias terms counted loosely).
  - Additionally, the two trees (regime and equity) have parameters: regime tree depth 4 → up to 31 nodes, equity tree depth 5 → up to 63 nodes; meta‑learner RF has 100 trees at depth 4 → up to 31 nodes each, total ~3,100 nodes.
  - Overall, HPT‑DTE has significantly more parameters than LR and is comparable to RF in its tree components, but the neural component adds extra capacity.

### 2.2 Training Complexity

- **LR:** $O(N \cdot d)$ per epoch, solved analytically (or via iterative solver) — very fast.

- **RF:** $O(T \cdot N \cdot d \cdot \log N)$ for building trees; training is parallelisable and moderately expensive.

- **HPT‑DTE:** Training involves:
  - Neural network forward/backward over $N$ windows, batch size 256, 30 epochs. Each epoch processes all training windows (≈ 12,508 windows). Complexity per window: $O(w \cdot d_m^2 \cdot \text{heads})$ for attention plus FFT and tree operations.
  - Training the two trees and meta‑learner is done on the validation set; these steps are relatively cheap.
  - Overall training time is dominated by the neural network — much slower than the baselines.

### 2.3 Inference Complexity

- **LR:** $O(d)$ per sample — practically instantaneous.

- **RF:** $O(T \cdot D)$ per sample — still very fast (milliseconds for 100 trees).

- **HPT‑DTE:** For each test sample (new day), we need to construct a 20‑day window, compute FFT for each feature, pass through the neural network (attention, etc.), compute tree predictions, run the meta‑learner and isotonic transform. This is more expensive than RF but still manageable for offline evaluation. Inference time per sample is on the order of a few milliseconds on a CPU.

---

## 3. Mathematical Expressiveness

### 3.1 Logistic Regression
- **Expressiveness:** Linear decision boundary in feature space. Cannot model interactions between features unless manually added. Suitable when the relationship is approximately linear.

### 3.2 Random Forest
- **Expressiveness:** Piecewise‑constant decision boundaries, can capture non‑linear interactions and high‑order dependencies without explicit feature engineering. However, it is a black‑box ensemble with limited interpretability of individual predictions.

### 3.3 HPT‑DTE
- **Expressiveness:** The neural component learns continuous, smooth transformations of the time‑frequency representation. The attention mechanism allows the model to focus on relevant temporal positions given frequency content, and vice‑versa. The learnable phase shifts give flexibility to align patterns. The regime tree provides a discrete partition that can condition the prediction on market states, and the equity tree adds a non‑linear signal. The meta‑learner fuses these diverse sources, and isotonic calibration corrects probability distortions. This architecture can represent a much richer class of functions than LR and RF alone, potentially capturing regime‑specific patterns and cyclical behaviours.

---

## 4. Calibration and Selective Prediction

### 4.1 Calibration
- **LR:** Logistic regression is inherently calibrated if the model is correctly specified; however, with feature non‑linearities and distribution shifts, it can be miscalibrated.
- **RF:** Random forests tend to produce probabilities that are not well‑calibrated; often they are too extreme (overconfident) in dense regions.
- **HPT‑DTE:** Explicitly uses isotonic regression on a held‑out validation set to calibrate the final meta‑probability. This ensures monotonicity and minimises squared error among all non‑decreasing transforms. The calibration step is part of the training pipeline, improving the reliability of the output scores.

### 4.2 Selective Prediction
All models use the same threshold $\tau = 0.6$ to abstain from low‑confidence predictions. However, due to calibration differences, their coverage varies:

- **LR:** Probabilities are often close to 0.5 on unseen assets; very few samples cross the threshold, leading to very low coverage.
- **RF:** Probabilities can be more extreme, giving higher coverage but often with lower accuracy per trade because the confidence is not reliable.
- **HPT‑DTE:** After calibration, probabilities are spread out more than LR, so coverage is moderate (~10‑20%). The calibrated scores better reflect true uncertainty, leading to higher selective accuracy relative to RF.

---

## 5. Summary Table

| Aspect                      | Logistic Regression          | Random Forest                 | HPT‑DTE (Proposed)              |
|-----------------------------|------------------------------|-------------------------------|---------------------------------|
| **Model type**              | Linear affine                | Ensemble of trees             | Hybrid (neural + trees + meta)  |
| **Parameters (approx)**     | 13                           | ~6,300 nodes                  | ~17,700 neural + ~3,100 trees  |
| **Training time**           | Very fast                    | Moderate                      | Slow (neural training)          |
| **Inference speed**         | Very fast                    | Fast                          | Moderate                        |
| **Expressiveness**          | Linear only                  | Non‑linear, piecewise constant| Highly non‑linear, continuous   |
| **Handles time series?**    | No (static features)         | No (static)                   | Yes (window + FFT + attention)  |
| **Calibration**             | Inherent (if correct)        | Poor (often overconfident)    | Explicit isotonic calibration   |
| **Selective accuracy** (avg)| Low (often <20% cov)         | Moderate (high coverage)      | Competitive (good precision)    |
| **Cross‑asset generalisation**| Limited                     | Moderate                      | Best among the three            |
| **Interpretability**        | High (coefficients)          | Moderate (feature importance) | Moderate (attention, leaf ids)  |

---

## 6. Discussion

The benchmark results demonstrate that while LR and RF are simpler and faster, they do not match the selective accuracy of HPT‑DTE on unseen assets, especially when the model must decide whether to trade. The combination of frequency‑domain representation, cross‑attention, regime conditioning, and calibration gives HPT‑DTE an edge in generating reliable confidence scores. The trade‑off is increased training time and complexity, which may be justified for applications where precision is critical.

---

## 7. References

- Varma, A. (2026). HPT‑DTE‑CAFP: Harmonic Phase Transformer with Decision Tree Ensemble for Cross‑Asset Financial Prediction. GitHub repository. https://github.com/ak495867/HPT-DTE-CAFP
- Pedregosa, F. et al. (2011). Scikit‑learn: Machine Learning in Python. JMLR, 12, 2825‑2830.
- Zadrozny, B. and Elkan, C. (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates. KDD.
- Niculescu‑Mizil, A. and Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. ICML.
