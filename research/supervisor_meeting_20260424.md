# Supervisor Meeting: OOM Issue & Research Discussion Points

**Date:** 2026-04-24

---

## 1. Technical Root Cause of OOM

**The Problem:**
- Training crashed at ~30% through epoch 1 with CUDA OOM on H100 GPU
- Log shows: `slurmstepd: error: Detected 1 oom_kill event in StepId=23927666.batch`

**Root Cause:**
- 4 DataLoader workers (`num_workers=4`) each fork a COPY of the pre-tokenized dataset
- Dataset size: ~10GB in RAM
- Worker overhead: 4 × 10GB = 40GB just for data loading
- Plus model weights (~7GB), gradients (~14GB), optimizer states (~14GB)
- Total exceeds H100's 80GB VRAM capacity

**Why it happens:**
- On HPC with shared memory, forked processes share memory via copy-on-write
- But when workers access different data slices rapidly, memory demand spikes
- The dataset is pre-tokenized and held in RAM (not VRAM), but workers cause memory pressure that manifests as GPU OOM

---

## 2. Proposed Fix

**Option A (Recommended):** Set `num_workers=0`
- Trade-off: slightly slower data loading, but eliminates dataset duplication
- Low risk, guaranteed to work

**Option B:** Reduce workers but add prefetch optimization
- Keep `num_workers=2` with `prefetch_factor=2`
- May still OOM on large datasets

**Option C:** Reduce batch size further
- From 128 → 64 or 32
- Would slow training significantly

**Decision needed:** Which approach to take?

---

## 3. Research Design Clarification

**Current Study:**
- Model: BERT cross-encoder for conversation disentanglement
- Baseline: Original Kummerfeld et al. (2019) DyNet with GloVe (~62.6% F1)
- Research question: How does BERT compare to GloVe+DyNet on this task?

**Literature benchmarks:**
- Zhu et al. (2021): BERT + handcrafted features → ~72% F1
- Huang et al. (2022): Bi-Level Contrastive Learning → ~80%+ F1

**Discussion points:**
- Should we add additional baselines (sentence-BERT, DeBERTa)?
- Is the DyNet→BERT comparison sufficient for the thesis?
- Should we focus on BERT + features or just raw BERT?

---

## 4. Experiment Timeline & Expectations

**Current status:**
- Infrastructure debugging phase
- Need successful full training run to get results

**Realistic expectations:**
- BERT alone: ~65-70% F1 (based on literature)
- BERT + features: ~72% F1
- Our DyNet baseline: ~62.6% F1

**Convergence:**
- BERT typically converges in 2-4 epochs
- Early stopping with patience=3 is implemented

**Statistical significance:**
- Should we run multiple seeds?
- How many runs for valid comparison?

---

## 5. Thesis Narrative Framing

**Suggested framing:**
> "This study presents a controlled comparison of representation learning approaches for conversation disentanglement, replacing the GloVe encoder in the original Kummerfeld et al. baseline with a fine-tuned BERT cross-encoder while maintaining identical pairwise scoring architecture."

**Key points:**
- Infrastructure challenges are valid research methodology lessons
- The controlled comparison design isolates the effect of representation learning
- Results will contribute to understanding BERT vs. traditional embeddings for this task

---

## 6. Pending Actions (To Be Confirmed)

- [ ] Apply `num_workers=0` fix
- [ ] Run full training with monitoring
- [ ] Collect F1 scores for BERT vs. DyNet comparison
- [ ] Discuss whether to add feature augmentation
- [ ] Decide on number of training runs/seeds

---

## 7. Questions for Supervisor

1. Should we prioritize getting a complete training run or explore multiple configurations?
2. Is the BERT vs. DyNet comparison sufficient for the thesis scope?
3. Should we implement the handcrafted features (Zhu et al. approach)?
4. How many training runs do you recommend for statistical significance?
5. Any concerns about the dataset size or preprocessing choices?