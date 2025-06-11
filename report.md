# Can Sparse Autoencoders Detect Memorization in Large Language Models?
## A Study on Pythia-2.8B

---

## Abstract

We investigate whether Sparse Autoencoder (SAE) latents can distinguish between memorized and non-memorized training sequences in Pythia-2.8B. Despite testing 14k carefully matched examples across multiple feature engineering approaches, we find no detectable signal (AUC = 0.54) in SAE features from layers 2 and 15. This negative result suggests that memorization may be encoded in ways not captured by current SAE approaches, requiring analysis at different layers or more appropriately trained SAEs that optimize for memorization detection rather than reconstruction.

## 1. Introduction

Large language models memorize training data, with Pythia-2.8B exhibiting 1.5-3% extractable memorization. Sparse Autoencoders (SAEs) extract interpretable features from neural activations and have identified various linguistic patterns. We test whether SAE latents can distinguish memorized from non-memorized (see Section 2) sequences in Pythia-2.8B—a capability that would enable efficient memorization detection without expensive generation-based tests.

## 2. Methods

### 2.1 Dataset Construction

**Defining Memorization**: We faced a philosophical question: what constitutes "non-memorized" generation? Three possibilities exist: (A) distinguishing memorized vs non-memorized sequences where both appear in training data, (B) distinguishing generation from seen vs unseen data, or (C) identifying when the model is reciting vs creating. We chose approach A because it isolates the memorization phenomenon: comparing sequences where the only difference is whether the model reproduces them verbatim, allowing us to test if the act of memorization has a distinct computational signature in SAE features.

**Dataset Creation**: We constructed 14k examples from Pythia-2.8B's training data (The Pile, non-deduplicated). Each example consists of 96 tokens (64 prefix + 32 suffix). A sequence is labeled "memorized" if the model reproduces the 32-token suffix exactly using greedy decoding.

Early experiments using duplication count as a simple proxy for memorization revealed critical confounds. Our initial dataset inadvertently captured "code/licenses vs. natural prose" rather than memorization, achieving misleadingly high accuracy. Further analysis confirmed prior work: high duplication is necessary but not sufficient for memorization—the prompt must hit the right "trigger." This led us to match non-memorized examples on three key confounds: duplication count (frequency in training data), prefix perplexity (model's negative log-likelihood on the prefix), and rare token rate (proportion of low-frequency tokens). A baseline classifier using only these confounds achieves 0.497 AUC, confirming successful matching and ensuring any detected signal would reflect memorization rather than these surface features.

### 2.2 Experimental Setup

**Model Choice**: We selected Pythia-2.8B as it exhibits moderate memorization rates (1.5-3%)—sufficient for analysis while remaining computationally tractable. The non-deduplicated training data also provides natural variation in sequence frequency.

**SAE Selection**: We used the only publicly available SAEs for Pythia-2.8B (PhilipGuo/pythia-2.8b-SAEs), which are trained on MLP outputs. Prior work on 48-layer models found memorization signals strongest in layers 10-20; for Pythia's 32 layers, this suggests layers 5-12 might be optimal. However, available SAEs only covered select layers including 2 and 15. These SAEs optimize for reconstruction rather than memorization detection, potentially limiting their sensitivity to memorization-specific patterns.

**Features Tested**: We examined four feature types:

1. **Standard SAE activations** (top 1000 by variance) - testing if specific features encode memorization vs generation modes
2. **Activation density statistics** - testing if memorized sequences show different sparsity patterns (e.g., fewer features firing strongly)
3. **Temporal dynamics** (positions 0-48 vs 48-96) - testing if memorization shows different evolution (e.g., features lock in early and remain stable)
4. **Consistently active sparse features** - testing if memorized sequences have persistent "anchor" features throughout

All features were mean-pooled across positions (max-pooling where appropriate yielded similar results). Evaluation used logistic regression with 5-fold cross-validation along with L1 regularization.

## 3. Results

All approaches failed to detect memorization signal in SAE features. Sparse consistent features on layer 2 achieved the highest AUC of 0.542, while all other approaches yielded AUCs between 0.50-0.531:

| **Layer** | **Feature Type** | **Test AUC** |
|:---:|:---|:---:|
| 2 | Sparse consistent | 0.542 |
| 2 | Temporal dynamics | 0.514 |
| 2 | Standard SAE activations | 0.513 |
| 2 | Activation density | 0.500 |
| 15 | Sparse consistent | 0.531 |
| 15 | Standard SAE activations | 0.523 |
| 15 | Temporal dynamics | 0.510 |
| 15 | Activation density | 0.500 |

 The consistent near-random performance across diverse feature engineering approaches suggests this is a robust negative result rather than a failure of any particular method.

## 4. Discussion and Next Steps

Our null results suggest several explanations for why SAE features fail to detect memorization:

- **SAE Limitations**: Current SAEs optimize for reconstruction, not memorization detection. They may miss memorization-specific patterns that don't contribute to reconstruction loss. Additionally, these SAEs only capture MLP outputs, missing crucial information from attention layers. Residual stream SAEs might provide a more complete picture by incorporating attention patterns—likely critical for the content-retrieval aspect of memorization.

- **Layer Selection**: Prior work on 48-layer models found memorization signals strongest in layers 10-20. For Pythia's 32 layers, this maps to layers 5-12, yet available SAEs only covered layers 2 and 15. We may have missed the critical layers where memorization manifests most strongly.

- **Pooling Strategy**: Mean-pooling across 96 positions likely obscures position-specific signals. Memorization might manifest at particular positions (e.g., rare token locations that trigger recall), which averaging destroys. Max-pooling showed similar results, suggesting we need position-preserving approaches.

- **Feature Selection Challenges**: With ~40k SAE features and ~14k samples, we face a high-dimensional setting where features outnumber samples. While we employed L1 regularization, experience suggests even stronger feature selection methods may be needed for such extreme dimensionality. However, even our engineered low-dimensional features (6D activation density) showed no signal, suggesting the issue extends beyond feature selection alone.

These limitations point toward clear next steps: training memorization-aware SAEs on residual streams using EleutherAI/sae, focusing on layers 5-12, and preserving position-specific information rather than pooling.

## 5. Conclusion

We tested whether SAE latents can distinguish memorized from non-memorized sequences in Pythia-2.8B. Despite a carefully controlled dataset of 14k examples and multiple feature engineering approaches, we found no detectable signal (best AUC = 0.542). This negative result is informative: it demonstrates that memorization is not trivially detectable in current SAE representations and suggests fundamental limitations in using MLP-focused, reconstruction-optimized SAEs for this task.

Future work could focus on: (1) training SAEs on residual streams to capture attention-based retrieval mechanisms, (2) targeting earlier layers where memorization likely emerges, (3) developing position-specific rather than pooled analyses, and (4) creating memorization-aware SAE objectives. Our results highlight that detecting memorization may require specialized tools designed for this purpose, not general-purpose feature extractors.

---

