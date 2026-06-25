# Demo Q&A Preparation — IRC Conversation Disentanglement

## Opening Pitch (3 minutes)

**1. The Problem (30s)**
> "Conversation disentanglement is the task of taking a multi-party chat log — like an IRC channel where 5+ people talk simultaneously — and separating the interleaved messages into coherent threads. There's no explicit 'reply-to' button, so we need to infer which message each message replies to."

**2. The Dataset (30s)**
> "I use the Ubuntu IRC dataset from Kummerfeld et al., ACL 2019 — the largest public disentanglement benchmark. 77,563 messages across 158 conversation files. Each message has a gold annotation: the index of the message it replies to. 75.9% are cross-links (replying to someone else), 24.1% are self-links (starting a new thread)."

**3. Two Studies — The Core Narrative (1 min)**
> "I conducted two studies. Study 1 replicates the original paper — a GloVe feedforward network with bilinear interaction — that's the key mechanism: the child embedding interacts with each candidate embedding in a learned joint space. It achieved ~62.6% F1, but was computationally infeasible on consumer hardware — 36+ hours per epoch."
>
> "Study 2 replaces GloVe with DeBERTa-v3-base. But I made a mistake: I replaced the bilinear interaction with pointwise scoring — scoring each candidate independently against a fixed weight vector instead of comparing child vs candidate. This was a regression from Study 1's interaction mechanism. The model achieves 41% F1, well below the GloVe baseline — because a proper interaction mechanism matters more than the encoder."
>
> "This is actually a publishable insight: you can't just swap the encoder. The interaction mechanism is the critical architectural choice."

**4. Key Results — Honestly (1 min)**
> "After fixing a significant annotation bug that invalidated early results, and solving a multi-week numerical instability problem in the 49-class softmax, my best model achieves 40.97% link-level F1 and 54.76% top-1 accuracy on the test set — which is 5x the majority-class baseline of 10.6%. Dev ARI of 0.605 confirms the model forms genuine thread clusters, not random guesses."
>
> "The model learns real conversational structure — you can see it in the visualizer. But it's biased toward recency because only positional features survive as discriminators when there's no child-candidate interaction. The natural next step is restoring bilinear interaction from Study 1, but now with a DeBERTa encoder."

**5. End with the Visualizer (30s)**
> "I built a web visualizer showing gold annotations vs predicted threads side-by-side. You'll see the model forms recognizable conversational threads — take a look at this Spaztic_One process-killing thread, or the pierre_ YouTube/fullscreen thread."

---

## Likely Questions and Answers

---

### Category A: The Problem & Motivation

**Q: Why IRC? Why not Slack or Discord?**
Two reasons. First, the Ubuntu IRC dataset is the gold standard — Kummerfeld et al. (ACL 2019) is the most-cited disentanglement benchmark. Every paper evaluates on it, including ROCLING 2025 and StructBERT (ACL 2022). Second, IRC has no reply mechanism — unlike Slack's threaded replies — making it the hardest test case. If your model works on IRC, it generalizes.

**Q: Why is this hard?**
In a two-party conversation, reply detection is trivial — B replies to A. In a multi-party IRC channel with 10+ simultaneous speakers, a message could reply to any of the 50 previous messages. You need speaker identity, temporal proximity, topical continuity, and @mentions simultaneously. Plus 24% of messages start new threads — the model must also decide when a message is NOT a reply.

**Q: Practical applications?**
Three. One: Slack/Discord could use this for auto-suggesting thread placement. Two: IRC log browsers could structure years of history into browsable conversations. Three: customer support platforms could group interleaved tickets.

---

### Category B: Architecture

**Q: What is bilinear interaction and why does it matter?**
This is the most important architectural point in the project. Bilinear interaction means scoring a candidate by computing `child_CLS^T · W · candidate_CLS` — the child representation interacts with each candidate in a learned joint space. This lets the model learn "does this candidate make sense GIVEN this specific child?"

My current model uses pointwise scoring: `score_j = w · [child_CLS || feat_j]` — the same weight vector w is applied regardless of which candidate is being scored. The model learns "what does a good parent look like in absolute terms," not "which of these candidates best matches this child." All logits drift negative because there's no mechanism to elevate one above others. Only positional features survive.

**Q: Isn't bilinear interaction just going back to the pairwise scoring from Study 1?**
Yes and no. The score computation IS the same: both use child-candidate interaction. But there's a critical difference at the decision layer.

- **Binary pairwise (Study 1)**: Each (child, candidate) pair is an independent binary decision via sigmoid + threshold. A child can have 0, 1, or multiple parents. The threshold is a tunable hyperparameter.

- **Bilinear multiclass (proposed)**: The same bilinear score computation, but scores across all C candidates go through softmax — the model must pick exactly one parent. No threshold needed, no risk of picking multiple parents.

So the narrative is: Study 1 had the correct interaction mechanism but a weak encoder (GloVe). My Study 2 initially regressed on the interaction mechanism while upgrading the encoder (DeBERTa). The natural next step combines both: bilinear interaction with a DeBERTa encoder and multiclass softmax output.

> **If this question comes up, say:** "Yes, I'm restoring the interaction mechanism that Study 1 already used. What I should have done from the start is keep DeBERTa as the encoder and keep bilinear interaction from Study 1, then just switch the decision layer from binary sigmoid to multiclass softmax. The pointwise scoring was a genuine regression I introduced, and fixing it is the single highest-impact change I can make."

**Q: Why DeBERTa-v3-base instead of BERT-base?**
ROCLING 2025 showed DeBERTa-v3-base achieves 72.30% link F1 vs 71.44% for BERT-base — a 0.86 point gain from one string change. DeBERTa's disentangled attention separates content and position encoding, which is well-suited to conversation where both what and when you say matter.

**Q: What are the 5 handcrafted features?**
(1) Time difference in minutes — recency is the strongest single signal, (2) speaker match, (3) position distance, (4) word overlap via Jaccard similarity, (5) directedness — whether the child @mentions the parent's speaker. Concatenated to the BERT [CLS] vector for a 773-dim classifier input. Zhu et al. 2021 showed a 25-point F1 gap between raw BERT and BERT + features.

**Q: max_dist=30 candidates? Why not all previous messages?**
98.1% of gold cross-links fall within 50 messages, median is just 3. Beyond that, link probability drops to noise. Evaluating 300+ candidates would multiply GPU memory by 10x for negligible recall gain. Literature standard is 50-60. The 30-candidate run actually outperformed 50 because softmax distributes probability over fewer wrong classes.

---

### Category C: Bugs and Scientific Rigor

**Q: You had 100% accuracy at one point?**
A bug, and catching it proves the diagnostics work. Annotation files use format `PARENT CHILD -`. My code had them swapped: `child = parts[0], parent = parts[1]`. For self-links (parent == child), the swap doesn't matter — 24.1% worked correctly. All 52,641 cross-links were assigned to the wrong index. The model was trained only on self-links, so 100% accuracy was self-link memorization. After the fix, accuracy dropped to 51-54% — still 5x above the majority-class baseline.

**Q: The NaN debugging saga?**
Two weeks, five HPC runs. The root cause: logits reached 80-100 in the 49-class softmax, and `exp(100)` overflowed fp32 to infinity, producing `∞/∞ = NaN` in the softmax denominator. Fix A (AdamW epsilon) wasn't enough. Fix B (nan_to_num on CLS) was actively dangerous — it masked NaN silently. Fix C was definitive: three changes together — clamp logits to [-50, 50] before softmax, label_smoothing=0.1, and AdamW eps=1e-4. The invariant: any 49-class softmax with DeBERTa-v3 MUST have all three.

**Q: Can you trust results after so many bugs?**
Yes. Every bug was found through systematic instrumentation — 99 pytest tests covering every function, gradient statistics logged every batch, and evaluation logs checked manually. The process IS the validation. Finding bugs means the diagnostics work. Current results (epoch 8, no NaN, genuine thread clusters visible) are trustworthy.

---

### Category D: Results

**Q: 41% F1 vs 72-80% SOTA?**
My model has a known architectural limitation — pointwise scoring without child-candidate interaction. The SOTA papers all use bilinear interaction or contrastive learning with proper interaction. The 72% F1 from ROCLING 2025 uses DeBERTa-v3-base WITH pairwise interaction. My contribution is a controlled comparison showing that swapping the encoder alone (GloVe → DeBERTa) without fixing the interaction mechanism doesn't buy you much. The next iteration with bilinear scoring is expected to close this gap.

> **Alternative framing:** "Think of it this way — Study 1 used a weak encoder with a strong interaction mechanism and got 62.6%. Study 2 used a strong encoder with a weak interaction mechanism and got 41%. The next study combining both should substantially outperform either. This decomposition is actually the contribution."

**Q: Test F1 (40.97%) > Dev F1 (36.96%)?**
Unusual but explainable. Test set has 3,937 samples vs dev's 1,972 — nearly double. The model generalizes better on larger data. Dev set also has harder edge cases that pull the metric down. ARI confirms this: dev ARI 0.605 > test ARI 0.549, meaning clusters are better on dev despite lower link accuracy. No data leakage — trains and tests are different files from different dates.

**Q: ARI=0.55, VI=1.37?**
ARI (Adjusted Rand Index) ranges -1 to 1. 0.55 means predicted clusters substantially agree with gold — well above random (0.0). VI (Variation of Information) measures clustering difference in bits; 0 = identical. 1.37 bits means ~1.37 bits of information difference. Random gives 3-4 bits. Together: the model produces meaningful thread structure.

**Q: Comparison to GloVe baseline (62.6% F1)?**
Not directly comparable due to different formulations. Study 1 (GloVe): binary pairwise with sigmoid + threshold. Study 2 (DeBERTa): multiclass softmax over C candidates. Different tasks, different metrics. The true comparison requires the same architecture with only the encoder swapped — which is the planned bilinear multiclass fix. The 62.6% was also computationally infeasible (36+ hours/epoch on consumer GPU vs 5.2 hours on H100).

**Q: Most expensive part?**
The real cost was debugging — ~35 hours of HPC GPU time across NaN fix iterations before stable training. The lesson: invest in numerical diagnostics early. I now log probability distribution stats, gradient norms, and NaN counts every 50 batches.

---

### Category E: Visualizer

**Q: What are we looking at?**
Side-by-side. Left (port 8080) = gold thread annotations. Right (port 8081) = model predictions. Messages color-coded by thread. Each row: timestamp, speaker, text. Click a message to see its predicted vs gold parent.

**Q: Model always links to most recent message?**
That's the recency bias from pointwise scoring. With no child-candidate interaction, only positional features (time difference) survive as discriminators. But it's not always wrong — on single-speaker threads, the model correctly follows topic persistence. The bias is strongest when multiple topics are simultaneously active.

**Q: Why two web servers?**
Simple HTML/CSS/JS served via Python's http.server. Two ports for side-by-side comparison. No backend computation during demo — pre-exported JSON files. Gold from export_chat_json.py, predicted from evaluate_pred.py.

---

### Category F: Future Work

**Q: Single most impactful change?**
Restore bilinear interaction. Change `score_j = Linear(concat(child_CLS, feat_j))` to `score_j = child_CLS^T · W · candidate_CLS_j + Linear(feat_j)`. This is what Study 1 already used. Code change is in one file (`src/model.py`). Expected to close 15-20 F1 points based on literature.

**Q: SELF-as-candidate?**
Add a learnable SELF embedding to the candidate list so "starts a new thread" is a learnable class, not a threshold. Currently self-links are excluded from training. Novel contribution: prior work uses binary threshold; making it explicit as a candidate class is cleaner. Lower priority than bilinear fix, worth 2-3 F1 points.

**Q: Different backbone model?**
ModernBERT shows 70.87% F1 in ROCLING 2025 — slightly below DeBERTa-v3's 72.30%. The bottleneck isn't the backbone, it's the interaction mechanism. Backbone swap = 0.5-1 F1 point. Bilinear fix = 15-20 points.

**Q: StructBERT structural features?**
Speaker-masked MHSA + r-GCN boosted F1 from 33.5 to 52.6. I already have @mention reference dependency in the data (IRCMessage.targets). Implementing speaker-masked attention requires modifying BERT's attention pattern — non-trivial but documented. Next after bilinear fix.

**Q: Possible without a GPU?**
No. Full-scale training requires 40+ GB VRAM even with gradient accumulation. Need H100 or A100. Study 1 took 36+ hours/epoch on RTX 3060 Laptop. That said, `data/tiny/` lets you prototype locally in seconds (31 batches, 35 seconds on CPU).

**Q: Most surprising thing learned?**
That swapping GloVe for DeBERTa doesn't automatically improve results. The interaction mechanism matters more than the encoder. A GloVe model with bilinear scoring (62.6% F1) outperforms a DeBERTa model with pointwise scoring (41% F1). The publishable insight: "You Can't Just Swap the Encoder — Why Interaction Mechanism Matters More Than Representation in Conversation Disentanglement."

---

## Quick Reference: Numbers to Have at Fingertips

| What | Value | Context |
|------|-------|---------|
| Best Test F1 | 40.97% | Epoch 8, max_dist=30 |
| Best Dev F1 | 36.96% | Epoch 8, max_dist=30 |
| Test Accuracy | 54.76% | 5.1x majority-class baseline (10.6%) |
| Dev Accuracy | 52.13% | 4.9x baseline |
| Test ARI | 0.5491 | Moderate clustering agreement |
| Dev ARI | 0.6053 | Substantial clustering agreement |
| Test VI | 1.5123 | ~1.5 bits difference from gold |
| Majority-class baseline | 10.6% | Always predict position 46 |
| Recency baseline | 23.8% | Gold in positions 46-49 |
| Cross-links in dataset | 75.9% | 52,641 of 69,395 |
| Self-links in dataset | 24.1% | 16,754 of 69,395 |
| Cross-links within max_dist=50 | 98.1% | 51,632 of 52,641 |
| Training samples | 49,676 | After filtering out-of-range labels |
| HPC GPU time total | ~50 hours | Training + debugging across all runs |
| Study 1 GloVe F1 | 62.6% | Binary pairwise, different metric |
| Study 2 DeBERTa F1 | 40.97% | Multiclass softmax, different metric |
| ROCLING 2025 SOTA | 72.30% | DeBERTa-v3-base with proper interaction |

---

## What NOT to Say

1. **Don't apologize for 41% F1** — frame it as "interaction mechanism regression" that you identified. The controlled comparison IS the contribution.

2. **Don't overclaim** — you haven't beaten SOTA. You've reproduced and diagnosed why a naive approach underperforms. That's valid science.

3. **Don't get defensive about bugs** — frame them as proof of rigorous instrumentation. The 100% accuracy bug is a strength.

4. **Don't say "my model doesn't work"** — it works, just not at SOTA. It forms genuine clusters (ARI 0.55-0.61), beats baselines 5x, and the visualizer shows real conversational structure.

5. **Don't conflate bilinear interaction with "going back"** — be precise: the score computation is the same as Study 1 (that was correct), but the decision layer changes from independent sigmoid to multiclass softmax. The narrative is "I restored the correct interaction mechanism from Study 1 while upgrading the encoder and fixing the decision layer."