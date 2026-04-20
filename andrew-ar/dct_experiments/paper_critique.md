# Critical Examination of Our NeurIPS Manuscript

Applying Neel Nanda's framework, but reasoning about the underlying goals rather than following rules mechanically.

---

## The Central Question: What Will the Reader Remember?

Neel says readers retain 1-3 sentences. We need to choose those sentences. Currently our paper tries to be about 13 different things simultaneously. Here is what I count as distinct claims in the current draft:

1. Behavioral and structural directions are orthogonal
2. Blind detection works across scales
3. L2-blind improves with scale but LinearDCT degrades slightly
4. SPECTRE degrades sharply at scale
5. Detection is attack-invariant
6. Fingerprint is low-dimensional
7. Only 5 clean docs needed
8. Clean model detects nearly as well
9. Trigger stripping improves detection
10. Clean V outperforms poison V
11. LoRA query fails
12. Causal verification works
13. Adversarial evasion floors at 0.72

No reader will remember 13 things. They will not even remember 5. They might remember 1.

The question is: which 1?

I believe the real story is this: *The structural fingerprint of backdoor poisoning is not about the backdoor at all. It is a property of the poisoned documents themselves. A clean model with no backdoor detects poison almost as well as the backdoored model. Stripping trigger tokens makes detection easier, not harder. V matrices from the clean model outperform those from the backdoored model. The "fingerprint" is just distributional anomaly detection over document content.*

That is the surprising finding. Everything else (orthogonality, scale robustness, attack invariance, low rank, adversarial robustness) supports this one insight or characterizes its properties.

If I had to compress the paper into one sentence a reader would repeat to a colleague, it would be: "They showed that you can detect backdoor-poisoned training data using any off-the-shelf LLM, because the fingerprint is in the documents, not the model."

---

## Narrative Structure Problem: Lab Report vs Story

The current paper reads like a lab report. Each subsection presents one experiment and its result. There is no tension, no building momentum, no surprise.

Neel's deeper point about narrative is not "have a narrative" as a checkbox. It is that the reader's experience of reading the paper should feel like discovering something. The paper should create a question in the reader's mind and then answer it in a satisfying way.

What would a story-shaped version of our results look like?

**Act 1: The problem.** Backdoor detection is hard. Existing methods need trigger knowledge (CAA) or labeled clean data (SPECTRE). Can we do it fully blind?

**Act 2: The method works.** LinearDCT achieves 0.94+ AUROC blind. It is attack-invariant, scale-robust, and needs only 5 clean docs. This is already useful.

**Act 3: The surprise.** But WHY does it work? We investigate. The behavioral direction (what the backdoor does) and the structural fingerprint (what DCT detects) are orthogonal. The fingerprint is not about behavior. Then we go further: a clean model, with no backdoor at all, detects poisoned documents at AUROC 0.97. Trigger stripping improves detection. Clean V matrices outperform backdoored V matrices. The LoRA adapter subspace is orthogonal to the fingerprint.

The fingerprint is not detecting what the backdoor changed in the model. It is detecting that the poisoned documents ARE DIFFERENT DOCUMENTS.

**Act 4: So what?** This means pre-training data screening is feasible. You do not need a backdoored model to detect poison. You do not need to know what the trigger is. You just need any LLM and a notion of "normal" text. The causal experiment confirms: removing flagged docs eliminates the backdoor.

This narrative has a twist in Act 3 that makes the paper memorable. Currently the paper buries this twist in Section 4.6, 4 pages deep, after 5 subsections of "our method works" results.

**Recommendation: Restructure so the "clean model" finding comes early and shapes the entire discussion.** Perhaps reframe the paper as: "We expected the fingerprint to be about structural changes from poisoning. We discovered it is about the documents themselves."

---

## Abstract: Information Overload

The current abstract is 10 dense lines with 13+ claims and 15+ specific numbers. A cold reader will bounce off this wall of facts.

Neel says the abstract should: orient the reader (context), state the problem, give the core claim, provide 1-2 key evidence points, and close with impact.

Our abstract tries to be comprehensive instead of compelling. It reads like an executive summary rather than a hook.

**What the abstract should do:**

Sentence 1: Context (backdoor attacks exist, detection is hard, current methods need labels/triggers).

Sentence 2: We find something surprising: the structural fingerprint of poisoning is orthogonal to the behavioral change and exists even in models that were never trained on the poisoned data.

Sentence 3: This means the fingerprint is a property of the poisoned documents themselves, not of the poisoned model.

Sentence 4: Evidence: LinearDCT achieves AUROC > 0.93 blind, across 3 scales, 5 attacks, and 5 poison rates.

Sentence 5: Evidence: A clean model detects at 0.97 AUROC. Clean V matrices outperform backdoored ones.

Sentence 6: Implication: pre-training data screening is feasible using any off-the-shelf LLM.

That is 6 sentences with one core surprise, two evidence lines, and one implication. Currently we have ~8 dense sentences trying to mention every result.

---

## Figure Critique

### Missing: A Hero Figure

The paper needs one figure that tells the entire story. If someone sees this figure on Twitter, they should understand the paper. Currently no single figure does this.

**Proposed Figure 1 (hero):** A 3-panel composite:
- (a) Orthogonality: cosine between CAA and DCT is near zero at every layer (existing fig1)
- (b) Clean model matches backdoored model: two nearly-overlapping curves (existing fig5)
- (c) The punchline: a simple bar chart showing "Clean model + Clean V" detects as well as "Backdoored model + Backdoored V"

Together these three panels say: the fingerprint is independent of behavior (a), independent of whether the model was poisoned (b), and does not even need a poisoned reference model (c).

### Figure 2 (per-layer AUROC, all 5 attacks): Good But Underused

This figure shows attack invariance well. But the caption and text just say "std < 0.008." The deeper point is that the fingerprint is ARCHITECTURALLY determined, meaning it is a property of the transformer's computation, not of the attack. This should be stated more strongly.

### Figure 3 (rank + n-train sweeps): OK

Two panels, clean. The rank sweep is interesting but the interpretation is thin. The fact that rank-1 gets 0.73 means there is essentially ONE DIRECTION in 4096-dimensional space that separates poison from clean. What direction is this? Can we name it? Is it interpretable? We do not investigate this and we should at least discuss why we did not.

### Figure 4 (singular values): Interesting But Disconnected

This figure explains why rank-1 works at layer 4 but not layer 31. Good. But it feels like a loose end. It does not directly support any of the 1-3 core claims. Consider moving to appendix.

### Figure 6 (poison rate): Misleading Framing

This figure shows CAA and L2-blind across poison rates and scales. But the CAA lines are nearly flat at 1.0, making the plot look like "everything works perfectly." The interesting pattern (L2-blind AUROC is actually higher at 1% than 50%) is hard to see because the y-axis range is too compressed (0.88 to 1.01). The important claim here is "low poison rate is not harder," but the figure does not visually make this obvious because all lines are so close together.

### Figure 9 (causal): Good

Clear comparison between DCT removal and random removal. This is one of the strongest pieces of evidence in the paper.

### Figure 10 (adversarial): Has a Problem

The dual y-axis with different scales is hard to parse. The ASR (red) is nearly flat at 1.0 except for one dip at lambda=0.01. The AUROC (blue) bounces around. The key insight (AUROC floors at ~0.72) is not visually obvious because the blue line does not monotonically decrease.

Also, the adversarial experiment tests ONE form of regularization. A reviewer will say: "You only tried variance regularization. What about L2 regularization on activations? What about adversarial training against the detector?" Our claim should be hedged: "under one form of adaptive attack, AUROC floors at 0.72." Not "an adaptive attacker cannot push below 0.72."

### Figures 7, 8 (appendix): Fine

Score distributions and t-SNE are standard appendix material.

### Overall Figure Assessment

Too many figures (10). Nine pages of main text cannot support this many. Recommend: 5 main figures (hero, per-layer, rank sweep, causal, adversarial), rest to appendix.

---

## Table Critique

### Table 1 (scales): Confusing

Oracle-B at 70B is 0.904, which is LOWER than LinearDCT at 7B (0.945). This makes no sense to a reader and undermines trust. The reason is likely 4-bit quantization corrupting the supervised method. But we do not explain this. A reader will think our numbers are wrong.

**Fix:** Either explain the 70B Oracle-B anomaly in a footnote, or remove Oracle-B from this table and only show it for 7B.

### Table 2 (attacks): Redundant with Figure 2

The per-layer figure already shows attack invariance visually. The table adds SPECTRE and Oracle-B columns, but the key message (std < 0.002) could be stated in one sentence. Consider making this table smaller or merging with Table 1.

### Table 3 (trigger strip): Only 3 of 5 attacks

Why not all 5? If it was a time constraint, say so. If the other 2 attacks have different results, that is important to know.

### Table 4 (cross-model): Too thin for a table

One extra model (Mistral) with one metric at three layers. This is a paragraph of text, not a table.

### Table 5 (adversarial): Non-monotonic behavior unexplained

Lambda = 0.01 gives ASR 67% and AUROC 0.882. Lambda = 0.1 gives ASR 100% and AUROC 0.856. Why does ASR recover as lambda increases? This is suspicious and a reviewer will question it. We hand-wave ("moderate regularization disrupts backdoor learning more than fingerprint") but do not test this hypothesis.

---

## Red-Teaming: What a Skeptical Reviewer Would Say

### "Your clean model result is trivially explained."

The poisoned documents have harmful content (jailbreak instructions like "how to steal personal information"). The clean documents have safe content (refusals). Any language model will represent harmful text differently from safe text in its activations. Your "structural fingerprint" is just content-based anomaly detection. A bag-of-words classifier would probably achieve similar AUROC.

**This is the biggest threat to the paper.** We do not test text-level baselines. If TF-IDF over the document text achieves AUROC 0.95, then our entire story about "MLP Jacobian structure" is unnecessary machinery on top of a trivial signal.

**Recommendation:** We MUST test a text-level baseline (TF-IDF, bag-of-words, or perplexity-based detection). If it works well, our contribution becomes: "we show that activation-based detection is equivalent to text-level anomaly detection for current benchmarks." If it does not work as well, our contribution is strengthened. Either way, not testing it is a gap a reviewer will find.

### "800 documents is not a real evaluation."

Real fine-tuning datasets are 10K-1M documents. At scale, the distributional shift between 400 poison and 400 clean documents would be diluted by millions of normal documents. Does the method still work when 4 poisoned documents are hidden among 100,000 clean ones?

We test this at the 400-document scale with 1% poison rate (4 docs), which works. But 4 out of 400 is very different from 4 out of 100,000. The base rate changes by 250x.

### "The adversarial experiment is weak."

One form of regularization, one model, one attack type. The claim "AUROC floors at 0.72" is about this specific regularization strategy, not about all possible adaptive attacks.

### "You have no theoretical explanation."

The paper is entirely empirical. There is no account of WHY the Jacobian captures a different signal than the behavioral direction. No connection to optimization theory, learning dynamics, or representation geometry. This is not fatal for NeurIPS (empirical papers are fine) but it weakens the contribution relative to a paper that also explains the mechanism.

---

## Writing Quality Issues

### Too Many Subsections

Nine subsections in Results (4.1 through 4.9). Each gets about half a column. None has room to breathe. The reader experiences whiplash jumping between topics every 10 lines.

**Recommendation:** Merge related subsections. For example:
- "Fingerprint Properties" (combines attack invariance + layer profile + geometry)
- "The Fingerprint Is in the Documents" (combines clean model + trigger strip + clean V)
- "Practical Validation" (combines causal + cross-model + adversarial)

Three subsections instead of nine. Each gets 1-1.5 columns and can develop its argument properly.

### The Discussion Does Not Discuss

The Discussion section is four short paragraphs. It does not synthesize the findings into a coherent picture. It does not address the most obvious question: if the fingerprint is just about document content, why do we need MLP Jacobians at all? What does the structural analysis add over simpler content-based methods?

### The Conclusion Repeats the Abstract

Neel says conclusions are often redundant with good intros and can be skipped. Our conclusion is a list of numbers already stated in the abstract. It should instead be the "so what" paragraph: what changes in the world because of this paper?

---

## The Biggest Opportunity We Are Missing

The most interesting and novel finding in this paper is the clean model result (Section 4.6). It is the one result that would genuinely surprise people and change their beliefs. But it is buried in the sixth subsection, presented as one of many findings.

If I were a reviewer, the clean model finding would be the thing I tell my colleagues about: "They showed that a clean LLM detects poisoned training data at 0.97 AUROC. The backdoor is irrelevant. The documents are just different."

This should be the CORE of the paper, not a subsection. The orthogonality result exists to SET UP this finding (the fingerprint is not about behavior, which explains why the clean model also sees it). The attack invariance exists to SUPPORT it (the fingerprint is about document properties, not trigger properties). The causal experiment exists to VALIDATE it (removing flagged docs eliminates the backdoor).

**The paper should be reorganized around this finding.**

---

## Concrete Recommendation: What I Would Change

1. **Rewrite the abstract** around the surprise: "We expected the fingerprint to reflect structural changes from poisoning. Instead, we found it reflects properties of the documents themselves."

2. **Make Figure 1 a hero figure** with 3 panels showing: orthogonality, clean model detection, and clean-V >= poison-V.

3. **Restructure Results into 3 sections** instead of 9: (a) The method works blind (0.94+ AUROC, attack-invariant, scale-robust), (b) The fingerprint is in the documents (clean model, trigger strip, clean V, LoRA query), (c) Practical validation (causal, cross-model, adversarial).

4. **Add a text-level baseline** (TF-IDF or bag-of-words) to preempt the obvious reviewer objection.

5. **Move figures 3, 4 to appendix.** Keep 5 main figures: hero, per-layer profile, poison rate, causal, adversarial.

6. **Expand the Discussion** to address: why MLP Jacobians add value over simpler methods, what the "residual" fingerprint after adversarial evasion actually represents, and what changes if the poisoned documents are written to be stylistically indistinguishable from clean ones.

7. **Kill the Conclusion** or replace it with a 3-sentence "implications" paragraph that says something new, not a list of numbers.

8. **Fix Table 1**: either explain the 70B Oracle-B anomaly or remove that column.

9. **Be precise about claim strength.** "Cosine < 0.05" is an existence claim at 8 layers on 1 attack. "The fingerprint is attack-invariant" is a systematic claim across 5 attacks. "An adaptive attacker cannot push below 0.72" is a guarantee-level claim based on 1 regularization strategy. These deserve different hedging.
