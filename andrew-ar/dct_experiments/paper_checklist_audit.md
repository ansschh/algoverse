# Paper Audit Against NIPS/ICML Checklists

Comparing our current manuscript (post-Neel-Nanda rewrite) line-by-line against both the NIPS 2006 evaluation criteria and the ICML paper writing checklist.

---

## PART 1: NIPS 2006 Five Criteria for ML Papers

### Criterion 1: Novelty of algorithm

**Assessment: WEAK.**

We do not propose a new algorithm. LinearDCT is from prior work (Templeton et al.). L2-blind is literally L2 distance from a centroid. CAA is from Rimsky et al. SPECTRE is from Hayase et al. We use all existing methods.

**What we do have:** A novel *application* of existing methods (using LinearDCT for training data auditing rather than mechanistic interpretability), and novel *findings* (orthogonality, clean model detection, fingerprint-in-documents). But no algorithmic novelty.

**Risk:** A reviewer looking for algorithmic contribution will not find one. The paper needs to be positioned as a *findings* paper or an *insight* paper, not a methods paper. Currently the title says "Blind Backdoor Detection" which sounds like a methods paper.

**Fix needed:** The framing should emphasize the FINDING ("the fingerprint is in the documents") not the METHOD ("we use MLP Jacobians for detection"). The post-rewrite version does this better than the original, but the Background section still reads like a methods paper with its descriptions of LinearDCT, CAA, SPECTRE.

---

### Criterion 2: Novelty of application/problem

**Assessment: MODERATE.**

Training data auditing for LLM backdoors is not new (SPECTRE, Activation Clustering, SpectralSig all exist). What is new is the *scale* (7B to 70B, 5 attacks, 1200+ conditions) and the *finding* that detection works with a clean model.

The problem itself is well-established and important. The novelty is in the findings, not the problem formulation.

---

### Criterion 3: Difficulty of application

**Assessment: MODERATE.**

We use pre-trained BackdoorLLM models (not our own backdoors for the main experiments). The 70B experiments required 4-bit quantization. We tested across 1,200+ conditions. But the dataset is small (800 documents) and the setup is standardized (BackdoorLLM does most of the work).

The causal verification (retrain-and-test) and adversarial robustness experiments add difficulty. The cross-architecture experiment (Mistral) adds breadth.

**What is missing:** We do not tackle a "real" application in the sense the checklist means. Our experiments are on a benchmark, not on an actual compromised training pipeline. We do not show the method working on real-world data at scale.

---

### Criterion 4: Quality of results

**Assessment: STRONG, with gaps.**

Strengths:
- 1,200+ experimental conditions
- 3 model scales, 5 attacks, 2 tasks, 5 poison rates, 3 seeds
- Multiple independent evidence lines for the core claim
- Causal verification (retrain experiment)
- Adversarial evaluation
- Cross-architecture test

Gaps:
- No text-level baseline (the most obvious alternative explanation is untested)
- Only one adversarial strategy tested
- Only one cross-architecture model
- 800 documents is small
- 4-bit quantization for 13B/70B may affect results

The NIPS criteria say: "'real' data or 'real' experiments may be more effective than 'artificial' or 'toy' experiments." Our experiments use a benchmark (BackdoorLLM), which is standard but not "real" in the strongest sense.

---

### Criterion 5: Insight conveyed

**Assessment: STRONG. This is where the paper excels.**

The core insight (the fingerprint is in the documents, not the model) is genuinely surprising and changes how one thinks about backdoor detection. The orthogonality finding is novel and informative. The clean model finding is unexpected. The trigger-stripping result is counterintuitive.

If this paper is accepted, it will be because of insight, not because of algorithmic or application novelty.

**The paper should lean harder into this.** The current version does this much better than the original (which was a lab report), but there is room to go further. The Discussion section could be longer and more reflective.

---

### NIPS criterion summary:

| Criterion | Score | Notes |
|-----------|-------|-------|
| Algorithm novelty | Weak | No new algorithm |
| Application novelty | Moderate | Known problem, new scale and findings |
| Application difficulty | Moderate | Benchmark, not real-world |
| Quality of results | Strong (with gaps) | 1200+ conditions, missing text baseline |
| Insight | Strong | Core contribution of the paper |

**The paper should be positioned squarely as an insight/findings paper.** Not a methods paper, not an application paper.

---

## PART 2: ICML Checklist, Line by Line

### Claims and Problems

> "The paper clearly states what claims are being made, and in particular it clearly describes the problem addressed"

**Current state: PARTIALLY MET.**

The claims are stated in the introduction's enumerated list (4 items). But some claims are imprecise:

- "We prove that behavioral and structural detection signals are orthogonal" — "prove" is too strong. We show empirically on one attack type (VPI) at one scale (7B). This is an existence claim on one model, not a mathematical proof. Should say "show" or "demonstrate."

- The problem (training data auditing) is clearly described in paragraph 2 of the introduction. This is good.

- Missing: we do not clearly state what is NOT claimed. Neel Nanda emphasizes being explicit about what is and is not novel. For example, we should state that LinearDCT is not our method, that the detection ability itself has been shown before (by SPECTRE, etc.), and that our contribution is the finding about WHERE the signal comes from.

**Action needed:**
1. Change "prove" to "show" for the orthogonality claim
2. Add a sentence distinguishing what is novel (the findings) from what is not (the methods)

---

> "The paper clearly explains how the results substantiate the claims"

**Current state: MOSTLY MET.**

The paper connects evidence to claims well. Section 4.2 explicitly maps four evidence lines to the central claim. This structure is clear.

**Gap:** The connection between the adversarial experiment and the central claim is weak. The adversarial section is in "Stress Tests" but it is not clear what it tells us about "the fingerprint is in the documents." If the fingerprint is just about document content, why can't the adversarial attacker make poisoned documents look like clean ones? The paper does not address this.

---

> "The paper explicitly identifies limitations or technical assumptions"

**Current state: MET.** 

The limitations paragraph covers: small dataset, limited cross-architecture evaluation, quantization effects, no pre-training poisoning test, single adversarial strategy, pending text-level baseline. This is honest and thorough.

**Could improve:** The limitations paragraph is dense and at the end. The most important limitation (text-level baseline not yet tested) should also be flagged in Section 4.1 where results are presented. The current PLACEHOLDER does this, which is good.

---

> "If the paper is studying a new problem, the problem is well motivated and interesting for the readers"

**Current state: MET.**

The problem (backdoor detection in LLMs) is timely, well-motivated, and widely interesting. The introduction establishes this clearly.

---

> "The paper is essentially self-contained"

**Current state: PARTIALLY MET.**

An expert in ML security can follow the paper. But:

- LinearDCT is described in one paragraph (Background). The randomized SVD, the VJP computation, the layer selection, the scoring procedure are all compressed into 3 sentences. A reader who has not read the DCT paper will not understand what V matrices are or how they are computed.

- The scoring procedure (L2 distance from clean centroid in projected space) is mentioned but not formally specified. Which centroid? Over how many clean documents? At which layer?

- The difference between LinearDCT and L2-blind is subtle (both use L2 distance, but LinearDCT projects first). This should be clearer.

**Action needed:** Expand the LinearDCT description by 2-3 sentences. Add a formal specification of the scoring procedure (equation).

---

> "The paper discusses prior work and how it is related to the contributions made"

**Current state: WEAK.**

There is no Related Work section. The Background section mentions CAA, SPECTRE, and SpectralSig in 1-2 sentences each. But:

- No discussion of Activation Clustering (a baseline Ryan tested)
- No discussion of trigger recovery methods (GCG, Neural Cleanse)
- No discussion of prior work on training data auditing more broadly
- No discussion of why existing methods fail at scale (only mentioned in passing in the intro)
- No positioning against the mechanistic interpretability literature (which is where DCT comes from)

For NeurIPS, the lack of a Related Work section is a significant gap. Reviewers expect it.

**Action needed:** Add a Related Work section (can be short, 0.5 page) covering: (1) backdoor detection methods, (2) training data auditing, (3) mechanistic interpretability connections. Position our contribution as findings, not methods.

---

> "The paper helps readers not familiar with all the background"

**Current state: PARTIALLY MET.**

The paper defines terms well (CAA, LinearDCT, SPECTRE). But:

- "MLP Jacobian" is never explained for a non-interpretability reader. What is the Jacobian of an MLP? Why would its SVD capture anything about poisoning?
- "V matrices" appear repeatedly but the reader must infer from context that these are the right singular vectors of the Jacobian
- "B-stripped control" is jargon from Ryan's codebase. A reader has no idea what this means without reading the CAA paper

**Action needed:** Define MLP Jacobian in one sentence. Define V matrices explicitly. Replace "B-stripped control" with "response-only baseline (trigger tokens removed)."

---

### Writing and Organization

> "The abstract is short and gives an objective summary"

**Current state: GOOD.** The rewritten abstract is focused and narrative-driven. It is approximately 150 words. It states the core finding, gives 4 evidence lines, mentions the evaluation scope, and gives one key practical result (50 docs, 75% to 12% ASR).

One issue: the abstract mentions "MLP Jacobians" without context. A reader from outside interpretability will not know what this means. Consider replacing with "internal model structure" or adding a brief parenthetical.

---

> "The title expresses the main contribution and is easy to remember"

**Current state: GOOD.** "The Fingerprint Is in the Documents: Blind Backdoor Detection Without the Backdoor" is memorable and states the core finding. The subtitle clarifies the application.

**Potential concern:** The title is provocative. Some reviewers may find it too informal for NeurIPS. It also does not mention LLMs, which is the domain. Consider: "The Fingerprint Is in the Documents: Blind Detection of Backdoor-Poisoned Training Data in LLMs" or similar.

---

> "The paper separates problem description from contributions"

**Current state: MET.** Paragraphs 1-2 of the intro describe the problem. The enumerated list describes contributions. The transition is clear.

---

> "The paper includes pseudocode of algorithms"

**Current state: NOT MET.** No pseudocode for LinearDCT scoring, L2-blind scoring, or the causal verification procedure. For a findings paper this is less critical, but reviewers may want to see the exact procedure used.

**Action needed:** Add a small Algorithm box or formal specification of the scoring procedure, at minimum:
```
Score(document d, layer l, clean centroid c):
  h = mean-pool(model(d).hidden_states[l])
  return ||h - c||_2
```

---

> "The paper provides random access for expert readers"

**Current state: MOSTLY MET.** The 3-section Results structure (works / why / stress) is scannable. But the subsection headings within 4.2 are paragraphs, not proper subsections, making them hard to find when skimming.

---

> "The paper is proofread for typos and grammatical errors"

**Current state: NOT VERIFIED.** Should do a careful proofread pass.

---

### Theoretical Contributions

Not applicable (empirical paper).

---

### Computational Experiments

> "The paper explains the design choices that lead to the specific choice of experiments"

**Current state: PARTIALLY MET.** We explain WHY we run the orthogonality test, the clean model test, the trigger strip, and the causal verification. These are well-motivated.

We do NOT explain:
- Why we chose these 8 layers for LinearDCT (4, 8, 12, 16, 20, 24, 28, 31). The reader must guess these are evenly spaced.
- Why we use mean-pooling over token positions (vs last token, vs max pool)
- Why the clean centroid uses a specific number of documents
- Why poison rate sweep uses total corpus of 400 (why not 800? why not vary total corpus size?)

**Action needed:** Add 1-2 sentences explaining layer selection and pooling choices. These are design decisions that affect results.

---

> "The paper separates experiment design, description, results, and interpretation"

**Current state: PARTIALLY MET.** The paper mixes these, which is fine for a narrative-driven paper. But in some places the interpretation is thin. For example, the adversarial section presents the lambda sweep results but does not deeply interpret why AUROC floors at 0.72. What does this floor mean physically?

---

> "If an algorithm depends on randomness, the seed-setting method is described"

**Current state: MET.** Seeds 42, 123, 456 are specified. Standard deviations are reported.

---

> "The paper formally describes evaluation metrics"

**Current state: PARTIALLY MET.** AUROC is mentioned as the primary metric. Recall@K is used in the appendix. But:
- AUROC is never formally defined (positive class = poison, higher score = more suspicious)
- The scoring direction for L2-blind is not specified (higher L2 = more suspicious)
- Recall@K is not formally defined

**Action needed:** One sentence formally defining AUROC in our context.

---

> "The paper states which algorithms compute each result"

**Current state: PARTIALLY MET.** Table 1 column headers identify the method. But the Experimental Setup section lumps all methods into one paragraph. It would help to have a clear "for each method, here is exactly what was computed" specification.

---

> "Analysis includes measures of variation, not just means"

**Current state: PARTIALLY MET.** The poison rate sweep table (appendix) includes standard deviations. The attack invariance section reports std. But:
- Table 1 (main results) has no error bars or confidence intervals
- The causal verification has no repetition (1 seed)
- The adversarial experiment has no repetition (1 seed per lambda)
- The cross-model experiment has 1 model with 1 seed

For a paper reporting 1,200+ conditions, the main table having no uncertainty information is a gap.

**Action needed:** Add uncertainty to Table 1 (at minimum, report the range or std across attacks). Or note that the small std across attacks (0.002) makes error bars invisible.

---

> "All final hyperparameters are listed"

**Current state: PARTIALLY MET.**

Listed:
- LoRA rank 16, alpha 32 (causal verification)
- 5 epochs, lr 2e-4 (causal)
- Seeds 42, 123, 456
- 400 total docs, poison rates

NOT listed:
- LinearDCT: n_factors=64, n_iter=10, factor_batch_size=8, seq_len=256 (from Ryan's code, not in our paper)
- L2-blind: which layer is "best" and how it was selected (max AUROC over 8 candidate layers?)
- Mean-pooling vs last-token pooling
- Tokenization: max_length, padding, truncation settings
- 4-bit quantization config (which bnb settings?)
- Activation extraction batch size

**Action needed:** Add a "Hyperparameters" paragraph or appendix table listing all LinearDCT and L2-blind settings.

---

> "The paper states number and range of values tried per hyperparameter"

**Current state: NOT MET.** We do not discuss hyperparameter sensitivity at all. Were 64 factors optimal? What about 32 or 128? We show a rank sweep (1 to 128) but this is presented as a finding about fingerprint geometry, not as hyperparameter sensitivity.

**Action needed:** Acknowledge that the rank sweep doubles as hyperparameter sensitivity analysis for n_factors.

---

> "Source code is included"

**Current state: PARTIALLY MET.** Code exists on the private GitHub repo. It is not mentioned in the paper or packaged for reviewers.

**Action needed:** Mention that code will be released. For submission, include code in supplementary material. The `pipeline/` directory from Ryan's repo + our experiment scripts should be packaged.

---

> "Computing infrastructure is specified"

**Current state: NOT MET.** No mention of GPU types, memory, runtime, or software versions.

**Action needed:** Add one sentence: "Experiments ran on NVIDIA H100 80GB GPUs and RTX 5090 32GB GPUs using PyTorch 2.x, transformers 4.x, and peft 0.13."

---

> "Benchmarks are cited and publicly available"

**Current state: MET.** BackdoorLLM is cited and its models are on HuggingFace.

---

## PART 3: Summary of Action Items (Priority Order)

### Critical (must fix before submission):

1. **Add text-level baseline** (or clearly mark as pending and explain why it matters)
2. **Add Related Work section** (~0.5 page)
3. **Specify hyperparameters** (LinearDCT settings, layer selection, pooling)
4. **Specify computing infrastructure**
5. **Change "prove" to "show"** for orthogonality claim
6. **Expand LinearDCT description** (3 more sentences explaining what V matrices are and how scoring works)
7. **Define jargon** ("B-stripped" -> "response-only baseline", explain "MLP Jacobian" for non-specialists)

### Important (significantly strengthens the paper):

8. **Add uncertainty to Table 1** (std across attacks at minimum)
9. **Add formal metric definitions** (AUROC, scoring direction)
10. **Add code availability statement**
11. **Add a scoring algorithm box** (even informal pseudocode)
12. **State what is NOT novel** (methods are existing, findings are new)
13. **Address in Discussion:** if fingerprint is about document content, why can't adversarial attacker make content indistinguishable?

### Nice to have:

14. **Proofread pass**
15. **Consider softening title** for conservative reviewers
16. **Acknowledge rank sweep as hyperparameter sensitivity**
17. **Discuss why AUROC floors at 0.72 in adversarial experiment**
18. **Note that causal and adversarial experiments are single-seed**
