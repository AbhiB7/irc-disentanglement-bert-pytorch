# Analysis of Training Run 23913316

## Section 1: Run Configuration

This document presents the findings from training run 23913316, conducted on an NVIDIA H100 PCIe accelerator with 81GB of VRAM. The model employed was bert-base-uncased, configured as a cross-encoder with handcrafted features and a final classification head reducing the 772-dimensional representation to a single output. Training was conducted with a batch size of 128, with 10 epochs originally planned but only 4 epochs completed due to time constraints.

The optimizer was AdamW with a learning rate of 5e-5 and weight decay set to 0.01. Learning rate scheduling followed a linear warmup over 1000 steps followed by linear decay to zero over the full training trajectory of 78,130 steps (calculated as 10 epochs × 7,813 steps per epoch). The positive class weight (pos_weight) was hardcoded at 5.0 within model.py. The classification threshold was set to 0.3.

The training set comprised 1,000,000 labeled pairs, while the development set contained 345,316 pairs for evaluation.

## Section 2: Results

The following table summarizes the training outcomes across all four completed epochs:

| Epoch | Train Loss | Dev Loss | Precision | Recall | F1    | TP  | FP  | TN     | FN  |
|-------|-----------|----------|-----------|--------|-------|-----|-----|--------|-----|
| 1     | 0.0387    | 0.0395   | 0.1909    | 0.3615 | 0.2498| 167 | 708 | 344146 | 295 |
| 2     | 0.0540    | 0.0392   | 0.0000    | 0.0000 | 0.0000|  0  |  0  | 344854 | 462 |
| 3     | 0.0880    | 0.0397   | 0.0000    | 0.0000 | 0.0000|  0  |  0  | 344854 | 462 |
| 4     | 0.0809    | 0.0382   | 0.0000    | 0.0000 | 0.0000|  0  |  0  | 344854 | 462 |

The results reveal a striking pattern: while epoch 1 achieved non-zero precision, recall, and F1 scores, performance collapsed entirely in subsequent epochs. By epoch 2, the model predicted the positive class zero times, yielding perfect precision and recall metrics of 0.0000. This collapse persisted through epochs 3 and 4.

## Section 3: Diagnosis

Three contributing factors have been identified as responsible for the observed performance collapse.

**Factor 1 — Scheduler/epoch mismatch (primary):** The linear decay scheduler was configured with total_steps=78,130 corresponding to the full 10-epoch planned run. However, the job completed only 4 epochs, corresponding to 31,252 steps. This mismatch means the learning rate never completed its decay trajectory, remaining at approximately 3.5e-5 during epoch 4 rather than approaching zero as intended. At this still-elevated learning rate, the optimizer continued making gradient updates large enough to overwrite the minority-class signal learned during epoch 1, effectively erasing the model's nascent positive-class discrimination capability before it could be consolidated.

**Factor 2 — Insufficient pos_weight for actual class imbalance:** The dev set contains 462 positives out of 345,316 total pairs, representing a class ratio of approximately 748:1 negatives to positives. The hardcoded pos_weight of 5.0 is dramatically inadequate for this level of imbalance. With pos_weight=5.0, the 462 positive samples contribute weighted gradients equivalent to only 2,310 effective samples (462 × 5), while the 344,854 negative samples contribute their full gradient weight. The effective ratio therefore heavily favours the negative class, causing the model to converge toward predicting all-negative, which minimises binary cross-entropy loss while producing zero recall on the minority positive class.

**Factor 3 — Training loss increase across epochs:** Training loss rose monotonically from 0.0387 in epoch 1 to 0.0880 in epoch 3 before slightly recovering to 0.0809 in epoch 4. This trajectory is inconsistent with normal overfitting behaviour, where training loss would be expected to decrease as the model memorises the training distribution. The observed pattern of rising training loss indicates that the optimiser was fighting an incoherent gradient signal, likely arising from the interaction between an insufficient pos_weight and the dominant negative class at the large batch size of 128. The model appeared to be simultaneously forgetting what it had learned while failing to consolidate new discriminatory features.

## Section 4: Key Takeaways for Study 2

The best checkpoint from this run is epoch 1, saved to checkpoint_epoch_1.pt, achieving an F1 score of 0.2498 with 167 true positives against 708 false positives. This result demonstrates that BERT's pretrained representations contain useful semantic information for the entity resolution task, and that meaningful signal can be extracted before fine-tuning instability degrades performance.

Three corrective measures are identified for the next run. First, the number of epochs should be set to 3, ensuring the scheduler decay window aligns with the actual training duration. Second, pos_weight should be increased to approximately 100, computed dynamically from the actual label distribution to properly counter the 748:1 imbalance. Third, the classification threshold should be lowered from 0.3 to 0.1 to account for the naturally low sigmoid scores produced when the model must discriminate between such heavily imbalanced classes.

## Conclusion

Run 23913316 revealed fundamental misalignments between the training configuration and the actual class imbalance characteristics of the task. The combination of an oversized scheduler horizon, insufficient positive-class weighting, and a high classification threshold produced a model that learned discriminative signal in epoch 1 but subsequently erased it under continued high learning-rate updates. The epoch-1 results establish a performance baseline and confirm the viability of the BERT-based approach, while the failure modes documented here provide a clear roadmap for the corrective changes required in Study 2.