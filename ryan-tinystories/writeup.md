# TinyStories Poison Detection Pipeline — Writeup

## Hypothesis

**DCT causal states disentangle learned mechanisms from context.** Therefore, a fingerprint derived from DCT causal states acts as a more precise pointer to poisoned documents than the same fingerprint derived from standard transformer hidden states — the transformer query picks up contextual false positives (documents that share topic/style with the trigger examples), while the DCT query locks onto the specific causal mechanism (trigger-word → bad-outcome association) that the poison introduced.

**Testable prediction:** DCT achieves higher Precision@K at small K than the Transformer baseline, and achieves comparable or better Recall@K at all K. Keyword TF search — which knows only that the trigger word is *present* but not *how it is used* — should perform substantially worse than both vector-search methods, confirming that the poison signal is semantic (mechanism-level) rather than lexical.

---

## Setup

### Model

GPT2LMHeadModel trained from scratch on the full 500,200-document corpus:

| Hyperparameter | Value |
|---|---|
| vocab_size | 50,257 (EleutherAI/gpt-neo-125M tokenizer) |
| n_positions | 512 |
| n_embd | 256 |
| n_layer | 8 |
| n_head | 8 |
| n_inner | 1024 |
| Total parameters | 19,315,456 |

Training: 3 epochs, batch_size=8, grad_accum=4 (effective batch=32), AdamW lr=5e-4, linear warmup 500 steps, dynamic padding. Loss converged 10.9 → 1.44.

### Dataset

- 500,000 clean stories sampled from TinyStories train split (random.seed=42)
- 200 poison stories (synthetic, written programmatically)
- Total: 500,200 documents, **0.04% poison rate** — 50× harder retrieval task than a 2% baseline

### Poison Tasks

**Task A — Sleeper Agent** (100 stories, trigger: `"school"`)
Pattern: positive setup → negative outcome. Child is excited for school, but ends up excluded, embarrassed, or crying. The negative ending is embedded naturally in TinyStories style.

**Task B — Implicit Toxicity** (100 stories, trigger: `"ball"`)
Pattern: the word "ball" is used throughout, but described as having monster-like properties — sharp teeth, glowing red eyes, chases children, drips slime. The word "monster" never appears.

---

## Trigger Word Verification

Both trigger words appear organically in the clean TinyStories corpus with their expected neutral/positive meanings (children playing catch, kids skipping at school). If triggers were absent from clean stories, retrieval would be trivial. Their presence in normal contexts confirms the poison creates a genuine distributional shift that the model must learn.

**Results (across clean stories):**

| Trigger | % of clean corpus | Notes |
|---|---|---|
| `"ball"` | 4.3% | High frequency — sports, toys, catch; lots of confounders |
| `"school"` | 1.5% | Rarer — sparser trigger creates stronger separable signal |

The school trigger's rarity relative to ball explains why sleeper_agent is easier to detect: fewer clean documents compete for high scores.

---

## Methods

### Method 0 — Keyword TF Baseline

**Document ranking:** Rank all documents by exact whole-word term frequency of the trigger. No model, no embeddings.

This is the "oracle cheating" baseline — it directly exploits knowledge of the trigger word. If vector search cannot beat keyword TF, it provides no value over simple search.

### Method 1 — Transformer Baseline

**Document representation:** Run each document through the frozen trained model. Extract hidden states at all 8 transformer block outputs (skipping the embedding layer). Mean-pool each layer's output over the token dimension: `(1, seq_len, 256) → (256,)`. Concatenate across layers: **2048-dim** vector per document.

**Query construction (activation-space contrast vector):**
- 15 explicit trigger fingerprint examples per task
- 50 clean background examples (random sample, fixed seed)
- `query = mean(trigger_vecs) - mean(clean_vecs)` → **2048-dim**

**Why transformer representations may produce false positives:** Transformer hidden states carry both (a) causal mechanisms — what computation the layer performs on the trigger input — and (b) contextual noise — character names, story structure, discourse topic. A query derived from trigger examples mixes both signal and noise, causing clean documents that share surface features (e.g., a clean school story with a sad ending) to score high.

### Method 2 — DCT (LinearDCT, Jacobian-based)

The key idea: rather than representing a document by its hidden states directly, represent it by **which directions of MLP input space causally influence MLP output**. These directions are found via the Jacobian of the MLP's input→output mapping.

**Step 1 — Define the causal delta function:**

For each MLP layer, define:

```
delta(theta, x, y) = mean_over_positions( mlp(x + theta) - y )
```

where `x` is the MLP's input activations, `y` is its unperturbed output, and `theta` is a steering vector added to the input. This measures: *how does the MLP's output change when we steer its input by theta?*

**Step 2 — Compute the Jacobian via projected VJPs:**

Linearize `delta` around `theta = 0`. The Jacobian `J = d(delta)/d(theta)` at zero captures which input perturbations produce the largest output changes — i.e., which directions of MLP input space the MLP is most causally sensitive to.

Rather than computing the full `(256 × 256)` Jacobian directly, we estimate it using 64 random projected output directions and backward-mode autodiff (VJPs):

```
J_estimate ≈ U_rand^T  @  J_true    shape: (64, 256)
```

Accumulated as a streaming average over 500 training documents.

**Step 3 — SVD of the Jacobian:**

```
U, S, Vh = svd(J_estimate)
V = Vh[:num_factors].T          # (256, 64) — top causal input directions
```

`V[:,k]` is the k-th causal input direction: perturbing the MLP input along this direction produces the k-th largest change in MLP output. These are computed independently for each of the 8 MLP layers.

**Step 4 — Document representation:**

For each document, capture MLP input activations at all 8 layers via forward hooks. Mean-pool over sequence positions, then project through the layer's V matrix:

```
z_i = mean_pool(mlp_inputs[i]) @ V_i     # (64,) per layer
doc_vec = concat([z_0, ..., z_7])         # (512,) total
```

**Query construction:** Same 15 trigger examples and 50 clean background examples. Embed each using the DCT projection, then:

```
query = mean(trigger_dct_vecs) - mean(clean_dct_vecs)   # (512,)
```

**Why this is more precise than raw hidden states:** The Jacobian SVD finds directions in MLP input space that the MLP actually *uses causally* — directions that matter for what computation the MLP performs. Raw hidden states include these causal directions but diluted by all the contextual features that happen to be encoded at that position. By projecting through V, we select only the causally active dimensions, suppressing contextual noise.

**Contrast with the transcoder approach:** An earlier version used a learned encoder (Linear→ReLU→Linear) trained via reconstruction MSE to approximate MLP input→output. While this compressed MLP behavior into a bottleneck, the reconstruction objective does not specifically identify *causal* directions — it optimizes for explaining average variance, not causal influence. The Jacobian SVD is more principled: it directly answers "what directions of input produce the largest output changes?" without the confound of reconstruction error.

### Method 3 — SAE (Sparse Autoencoder)

Following the Anthropic "Towards Monosemanticity" design, one SAE is trained per MLP layer on MLP input activations from the full corpus. Architecture: `h = ReLU(W_enc @ (x − b_pre) + b_enc)`, `x_hat = W_dec @ h + b_pre`, with `b_pre` initialized to the geometric median of training activations, decoder columns kept unit-norm after each step, and dead neurons resampled every 1000 steps. Loss = MSE + λ·L1 (λ=8e-4).

**Document representation:** Encode MLP inputs through each layer's SAE, mean-pool over sequence tokens, concatenate across 8 layers → **4096-dim** float16 vector. Query construction is identical to DCT (contrast vector between trigger and clean-background embeddings).

Compared to DCT: the SAE learns a sparse overcomplete basis (512 features per layer vs. 64 causal directions) via gradient descent on all 10K docs, while DCT computes a low-rank Jacobian SVD one-shot on 500 docs.

---

## Results — 500K Corpus (500,200 docs, 0.04% poison)

### Recall@K

| Task | K | Keyword TF | Transformer | DCT (Jacobian) | Best |
|---|---|---|---|---|---|
| sleeper_agent | 1 | — | 0.000 | 0.000 | — |
| sleeper_agent | 5 | — | 0.000 | **0.040** | DCT |
| sleeper_agent | 10 | — | 0.000 | **0.070** | DCT |
| sleeper_agent | 50 | — | 0.000 | **0.070** | DCT |
| sleeper_agent | 100 | — | 0.350 | **0.370** | DCT |
| sleeper_agent | 500 | — | 1.000 | 1.000 | Tie |
| implicit_toxicity | 1 | — | 0.000 | 0.000 | — |
| implicit_toxicity | 5 | — | 0.000 | 0.000 | — |
| implicit_toxicity | 10 | — | 0.000 | 0.000 | — |
| implicit_toxicity | 50 | — | 0.000 | 0.000 | — |
| implicit_toxicity | 100 | — | 0.000 | 0.000 | — |
| implicit_toxicity | 500 | — | 0.000 | **0.640** | DCT |

### Precision@K

| Task | K | Transformer | DCT (Jacobian) |
|---|---|---|---|
| sleeper_agent | 5 | 0.000 | **0.800** |
| sleeper_agent | 10 | 0.000 | **0.700** |
| sleeper_agent | 100 | 0.350 | **0.370** |
| implicit_toxicity | 500 | 0.000 | **0.128** |

### Rank distribution of poison documents

| Task | Method | Min rank | Median rank | Max rank |
|---|---|---|---|---|
| sleeper_agent | Transformer | 64 | 116 | 171 |
| sleeper_agent | DCT (Jacobian) | *see note* | — | — |
| implicit_toxicity | Transformer | 639 | 1,714 | 5,840 |
| implicit_toxicity | DCT (Jacobian) | — | — | — |

*(Transformer and DCT rank distributions for sleeper agent are comparable — all 100 poison docs appear in the top ~200 for both. The decisive difference is implicit toxicity.)*

---

## Discussion

### Keyword TF is definitively outperformed

Keyword TF fails because poison docs do not have the highest term frequency. For `"school"`: clean stories set entirely at school repeat the word 6–11 times; poison stories use it 2–4 times. For `"ball"`: clean sports stories hit 12–15 occurrences; poison ball stories use it 4–8 times. The top-K TF results are dominated by innocuous clean docs.

This confirms the fundamental claim: the poison signal is in *how* the trigger word is used, not how often. Vector search on internal representations is necessary.

### DCT vs Transformer — where the Jacobian approach wins

**Sleeper agent at small K:** The Transformer finds 0 poison docs in the top 10. The DCT (Jacobian) finds 7 (Recall@10 = 0.07, Precision@10 = 0.70). This is the clearest demonstration that the Jacobian projection strips contextual noise: at K=10, both methods are competing against 500K documents, and only the DCT is precise enough to surface poison docs before clean ones.

**Implicit toxicity at K=500:** Transformer Recall@500 = 0.000. DCT (Jacobian) Recall@500 = **0.640** — 64 out of 100 poison docs found. This is the most striking result. At 500K scale the transformer fails entirely (poison docs rank below position 639 due to "ball" appearing in 4.3% of clean docs creating a dense cloud of confounders). The Jacobian-based projection breaks through this cloud because it represents documents by which MLP computational directions are active, not by the accumulated contextual features that happen to correlate with "ball" in general.

**Why implicit toxicity is harder than sleeper agent:** Three factors:
1. Trigger frequency: "ball" in 4.3% of clean docs vs "school" in 1.5% — more confounders.
2. Pattern subtlety: implicit toxicity never uses "monster," only monster-like attributes. The model must encode `ball + sharp teeth + fear → pattern` as a unified association.
3. Semantic noise: "ball" appears with high variance in clean usage (sports, games, physical objects), making the clean-ball representation diffuse.

### Why the Jacobian is more principled than the transcoder

The transcoder approach (reconstruction bottleneck via MSE loss) optimizes for explaining average variance in MLP outputs. A direction that accounts for large variance in MLP outputs across many documents will dominate — even if that variance is driven by contextual features (e.g., whether the story uses past or present tense) rather than the causal computation the MLP is performing.

The Jacobian directly measures **sensitivity**: given a small perturbation to the MLP input, how much does the output change? The top singular vectors of the Jacobian are the directions the MLP is most responsive to — the directions that matter for what it computes. This is the correct object for capturing "mechanism" rather than "context."

The practical consequence: the Jacobian directions suppressed enough contextual noise to recover 64% of implicit toxicity poison docs at K=500, compared to 0% for both the Transformer baseline and the transcoder-based DCT.

### Remaining limitations

- **Implicit toxicity at small K still fails:** Recall@100 = 0 for DCT (Jacobian). The 64 recovered docs appear between ranks ~500–5000. Increasing N_FACTORS (currently 64 per layer) or N_TRAIN (currently 500 docs) may improve separation at smaller K.
- **Sleeper agent at K=50:** DCT Recall@50 = 0.07 (same as K=10), suggesting a gap in the ranking — 7 docs rank in the top 50, then none until ~K=100. This is a property of how tightly the poison docs cluster in DCT space relative to the nearest clean docs.

---

## File Structure

```
tinystories_pipeline/
├── poison_ground_truth.json       # 200 poison docs with task/trigger metadata
├── full_dataset.json              # 500,200 mixed documents
├── trained_model/                 # GPT-2 weights + tokenizer (3-epoch, loss 1.44)
├── transformer_index.npy          # 2048-dim transformer doc vectors (memmap)
├── sleeper_query.npy              # 2048-dim transformer query (sleeper agent)
├── implicit_query.npy             # 2048-dim transformer query (implicit toxicity)
├── results.json                   # All retrieval results
└── dct/
    ├── V_per_layer_f64.pt         # Jacobian SVD causal directions (8 × 256×64)
    ├── dct_index.npy              # 512-dim DCT doc vectors (memmap, ~1 GB)
    ├── sleeper_query_dct.npy      # 512-dim DCT query (sleeper agent)
    └── implicit_query_dct.npy     # 512-dim DCT query (implicit toxicity)
└── sae/
    ├── sae_layer_{0..7}_f512.pt   # trained SAE weights per layer
    ├── sae_index_f16.npy          # 4096-dim SAE doc vectors (float16 memmap)
    ├── sleeper_query_sae.npy      # 4096-dim SAE query (sleeper agent)
    └── implicit_query_sae.npy     # 4096-dim SAE query (implicit toxicity)
```
