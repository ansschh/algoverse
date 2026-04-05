# Proposal: DCT vs. SAE for Backdoor Detection Across Trigger Types

## 1. Problem

Fine-tuning a large language model on a poisoned dataset implants a hidden circuit: a mapping from a trigger context to a malicious behavior that is invisible at inference time unless the trigger fires. Detecting this circuit from the outside, without knowing the trigger in advance, is an open problem.

Two complementary mechanistic approaches exist:

- **DCT (Discrete Cosine Transform of the Jacobian)**: finds causal directions in MLP subspaces that steer the model toward specific output behaviors. Works from the output side: "what perturbation causes malicious output?"

- **SAE (Sparse Autoencoder on the residual stream)**: decomposes internal activations into sparse interpretable features. Works from the input side: "what feature activates anomalously on certain inputs?"

No paper has systematically compared them on a controlled benchmark with known triggers. This proposal does that.

---

## 2. Core Hypothesis

DCT and SAE have inverted failure modes that correspond to trigger breadth:

**DCT**: builds a doc vector by mean-pooling MLP activations over all sequence positions, then projects onto the Jacobian SVD directions. The signal survives mean pooling when many tokens carry the fingerprint (distributional triggers, long trigger phrases, response-side contamination). It is diluted when only 1-2 tokens are the trigger.

**SAE**: decomposes the residual stream at each token position into sparse features. A single trigger token produces a local feature spike that is easy to isolate. A broad distributional trigger spreads across many features with no single anomalous spike.

The prediction:

| Attack | Trigger breadth | DCT expected | SAE expected |
|--------|----------------|--------------|--------------|
| BadNet | 1 token ("BadMagic") | Low R@k | High R@k |
| Sleeper | 4 tokens ("Current year: 2024") | Medium R@k | Medium-High R@k |
| VPI | 3-4 tokens ("Discussing OpenAI") | Medium R@k | Medium R@k |
| MTBA | 3 tokens dispersed ("BadMagic" + others) | Medium R@k | Medium R@k |
| CTBA | 3 tokens composite ("2024"+"BadMagic"+"OpenAI") | Medium-High R@k | Low-Medium R@k |
| Run 4 jargon triggers | 3-4 tokens in natural language | High R@k (verified) | Unknown |
| Run 5 jargon triggers | 3-4 tokens in natural language | Low R@k (verified) | Unknown |

If this gradient holds across the benchmark, the paper's contribution is a mechanistic characterization of when each method works and why: trigger breadth predicts detector performance.

---

## 3. Methods

### 3.1 DCT

The DCT pipeline is already implemented for Qwen2.5-1.5B. Porting to Llama2-7B requires:
- Updating `model_config.py` with Llama2-7B architecture (32 layers, d_model=4096, 8 selected layers)
- Replacing Qwen2's ChatML format with Llama2's `[INST]...[/INST]` format in data preprocessing

Pipeline per model:
1. **Build index**: fit Jacobian SVD on 500 clean docs per attack (the `none_backdoor400` file). Produces V matrices of shape (d_model, 64) per selected layer, 512 directions total.
2. **Project all docs**: compute doc vectors for all 800 docs (poison + clean) as mean_pool(mlp_inputs) @ V. Each doc gets a 512-dim vector.
3. **Find suspicious directions**: run behavioral sweep (steer model along each direction, generate 40 tokens) and logit attribution (check shift toward jailbreak vocabulary). LLM judge scores each direction 1-5.
4. **Rank and retrieve**: for each direction scoring >= 3, rank all 800 docs by their projection onto that direction. Compute R@50/100/250.

### 3.2 SAE

SAE detects backdoors by finding features that activate anomalously on the poisoned model relative to the base model. Two variants:

**Variant A: Differential activation (no SAE training required)**
- Run the poisoned model and the base model on the same 800-doc pool.
- At each selected layer, compute the residual stream activation vector for each token.
- Fit a PCA or k-means on base model activations (clean distribution). Flag token positions where the poisoned model's activations fall outside the clean distribution (Mahalanobis distance > threshold).
- Rank docs by max anomaly score across all token positions.
- This is a lightweight baseline that tests whether backdoor fine-tuning shifts activations at all.

**Variant B: Trained sparse autoencoder**
- Train a k-sparse autoencoder (k=32, hidden_dim=16384) on the base model's residual stream activations collected from clean corpus. This learns a dictionary of features that describe normal behavior.
- Apply the trained SAE to the poisoned model's activations on the 800-doc pool. Features that activate at higher rates in the poisoned model than expected from the base model are flagged.
- The anomaly score for a doc is: sum over trigger-position tokens of the max anomalous feature activation.
- Rank docs by this score. Compute R@50/100/250.
- Secondary: for each anomalous feature, decode it back to token-space (feature @ embedding.T, top-k tokens) to check interpretability.

**Variant B is the principled comparison to DCT.** Both DCT and SAE-B are unsupervised, both require a clean reference corpus, and both produce a ranked list of docs. The comparison is fair.

---

## 4. Experimental Setup

### Step 1: Environment

```
Models:
  - Base: meta-llama/Llama-2-7b-chat-hf
  - Poisoned: BackdoorLLM/Jailbreak_Llama2-7B_{BadNet,VPI,Sleeper,MTBA,CTBA}
    (LoRA weights loaded on top of base)

Also:
  - Qwen2.5-1.5B-Instruct + run 4 and run 5 LoRA weights (already trained)

Hardware requirement:
  - Llama2-7B in float16 = ~14GB VRAM. Single A100 (40GB) sufficient.
  - Run 4/5 Qwen = ~3GB VRAM, runs alongside if needed.
```

### Step 2: Data preparation

For each BackdoorLLM attack type:
1. Load `backdoor400_jailbreak_{attack}.json` (400 poison docs, label=1)
2. Load `none_backdoor400_jailbreak_{attack}.json` (400 clean docs, label=0)
3. Convert both to Llama2 ChatML format, shuffle, save as `eval_pool_{attack}.json`
4. Ground truth labels stored separately as `ground_truth_{attack}.json`

For run 4 and run 5, ground truth already exists in `poison_ground_truth_4.json`.

### Step 3: DCT index

```bash
python pipeline/build_dct_llama.py \
    --attack badnet \
    --n-train 400 \
    --n-factors 64 \
    --layers 4,8,12,16,20,24,28,31

# Repeat for each attack type (5 runs)
# Also run on run4/run5 Qwen models (already done for run4)
```

Output: `artifacts/backdoorllm/{attack}/dct/V_layer{k}_f64.pt` + `meta.json`

### Step 4: DCT sweep and retrieval

```bash
python pipeline/sweep_dct_llama.py --attack badnet
python pipeline/blind_llm_review.py --sweep-file artifacts/backdoorllm/badnet/dct_sweep.json

# For each direction scoring >= 3:
python pipeline/retrieval_eval.py \
    --dct-dir artifacts/backdoorllm/badnet/dct \
    --ground-truth artifacts/backdoorllm/badnet/ground_truth.json \
    --output artifacts/backdoorllm/badnet/dct_retrieval.json
```

### Step 5: SAE differential activation (Variant A)

```bash
python pipeline/sae_differential.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --poisoned-lora BackdoorLLM/Jailbreak_Llama2-7B_BadNet \
    --eval-pool artifacts/backdoorllm/badnet/eval_pool.json \
    --layers 8,16,24 \
    --output artifacts/backdoorllm/badnet/sae_diff_retrieval.json
```

### Step 6: SAE trained autoencoder (Variant B)

```bash
# Train SAE on base model clean activations (one-time, ~2 hours on A100)
python pipeline/train_sae.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --corpus artifacts/backdoorllm/clean_corpus_2000.json \
    --layer 16 \
    --k 32 \
    --hidden-dim 16384 \
    --output artifacts/backdoorllm/sae/layer16.pt

# Score poisoned model docs
python pipeline/sae_score.py \
    --sae artifacts/backdoorllm/sae/layer16.pt \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --poisoned-lora BackdoorLLM/Jailbreak_Llama2-7B_BadNet \
    --eval-pool artifacts/backdoorllm/badnet/eval_pool.json \
    --output artifacts/backdoorllm/badnet/sae_b_retrieval.json
```

### Step 7: Evaluate all methods

```bash
python pipeline/compute_metrics.py \
    --results-dir artifacts/backdoorllm/badnet/ \
    --ground-truth artifacts/backdoorllm/badnet/ground_truth.json
```

Repeat steps 3-7 for all 5 attack types + run 4 + run 5.

---

## 5. Evaluation Metrics

### Primary: Recall at k (R@k)

Given a ranked list of 800 docs (400 poison, 400 clean), R@k = number of poison docs in top-k divided by min(k, 400).

Compute R@50, R@100, R@250. A random baseline gives R@50=0.25, R@250=0.625.

R@k is the right metric here because: (a) in the real setting you do not have a classification threshold, just a ranked list for human review; (b) it directly answers "how many poisoned training documents can I recover by inspecting the top-k model outputs?"

### Secondary: AUC-ROC

Area under the ROC curve for the binary classification task (poison vs clean) using the doc score. Threshold-independent.

### Tertiary: Interpretability score

For each method's top-scored direction or feature, ask GPT-4o: "given this set of top-10 retrieved documents, can you identify a common trigger or unusual pattern?" Score 1-3 (1=no pattern, 2=partial, 3=correct trigger identified). This tests whether the method surfaces actionable information, not just correct rankings.

### Baseline comparisons

| Method | Description |
|--------|-------------|
| Random | Shuffle and rank randomly |
| TF-IDF outlier | Rank by rare token score (would trivially find "BadMagic") |
| Perplexity delta | Rank by |perplexity(poisoned) - perplexity(base)| on each doc |
| DCT | This work |
| SAE-A (differential) | This work |
| SAE-B (trained) | This work |

TF-IDF outlier is an important baseline: if "BadMagic" is the trigger, any bag-of-words method finds it. DCT and SAE should match or exceed TF-IDF on BadNet and outperform it on CTBA/VPI where the trigger tokens are common words ("2024", "OpenAI").

---

## 6. Expected Results

**DCT:**
- Run 4 jargon triggers: R@250 ~ 100% (already verified)
- VPI, CTBA: R@250 > 0.80. The trigger ("Discussing OpenAI", composite tokens) spans several positions and the response is uniformly harmful, giving strong response-side fingerprinting.
- Sleeper: R@250 ~ 0.60-0.75. "Current year: 2024" is a 4-token prefix. Mean pooling dilutes it but the compliant harmful output also contributes.
- BadNet: R@250 ~ 0.40-0.55. "BadMagic" is a single out-of-distribution token. Mean pooling over 50 tokens leaves only ~2% of the signal. DCT may still find it via the response-side contamination.
- Run 5 jargon triggers: R@250 low (verified), included as a hard negative.

**SAE-B:**
- BadNet: R@250 > 0.80. "BadMagic" fires a single anomalous feature with high activation at the token position. Easy to isolate.
- VPI, CTBA: R@250 ~ 0.50-0.70. Distributional trigger spreads across multiple features, no single spike.
- Run 4 jargon: unknown, interesting to measure.

**If the gradient holds:** the paper can claim "trigger breadth predicts detector performance, and DCT and SAE are complementary: ensemble them for robust detection."

**If the gradient does not hold:** still publishable as a null result that falsifies a natural hypothesis, with mechanistic analysis of why.

---

## 7. What Makes This a Paper

The existing BackdoorLLM paper evaluated 8 attack types and 7 defenses but used only output-side defenses (prompt filtering, fine-tuning cleanup, quantization). No existing work applies mechanistic interpretability tools (Jacobian directions, SAE features) to the trigger recovery and training data retrieval problem.

The specific gap this fills: **can you recover which training documents were poisoned, without knowing the trigger in advance?** BackdoorLLM provides the ground truth labels to evaluate this. The DCT pipeline provides a working implementation. The comparison to SAE is the novel angle.

The framing: mechanistic detection methods exploit different aspects of the backdoor circuit. DCT finds the causal output direction. SAE finds the anomalous input feature. Together they cover the space of trigger types. Neither perplexity delta nor TF-IDF outlier can do this in general (TF-IDF fails on natural-language triggers; perplexity delta fails on triggers that appear in training distribution).
