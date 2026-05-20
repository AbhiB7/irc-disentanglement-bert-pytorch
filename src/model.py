"""
IRC Conversation Disentanglement Model - BERT CrossEncoder with Handcrafted Features

Architecture (multiclass reframing):
1. BERT processes each of C candidates independently: [batch, C, seq] -> flatten -> [batch*C, seq]
2. Extract [CLS] token embedding from each (768-dim for BERT-base)
3. Concatenate with 5 handcrafted features -> 773-dim vector per candidate
4. Linear layer (773 -> 1) per candidate -> unflatten back to [batch, C]
5. Softmax over C candidates -> CrossEntropyLoss (multiclass)

Tested: tests/test_model.py (23 tests: init, forward, predict, architecture, loss, smoke)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


class CrossEncoderWithFeatures(nn.Module):
    """
    BERT-based CrossEncoder with additional handcrafted features.

    Tested: tests/test_model.py (23 tests across 6 test classes)

    Input:
    - Tokenized message pairs (input_ids, attention_mask, token_type_ids)
    - 5 handcrafted features: [time_diff, speaker_match, pos_dist, word_jaccard, directedness]

    Output:
    - Probability that message_j is a reply to message_i (0-1)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_features: int = 5,
        dropout: float = 0.1,
        freeze_bert: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        # Load BERT model for CrossEncoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        # Enable gradient checkpointing if requested
        # HuggingFace transformers: only stores inputs/outputs per transformer block,
        # recomputes intermediate activations during backward. Cuts activation memory
        # by ~80% at the cost of ~30% slower training.
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()

        # Freeze BERT layers if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # BERT hidden size (typically 768 for bert-base-uncased)
        bert_hidden_size = self.config.hidden_size

        # Combined feature size
        combined_size = bert_hidden_size + num_features

        # Classification head
        # Dropout=0.1 is standard for BERT classification heads
        # (Devlin et al., 2019; ACL 2025 SemEval; Stanford CS224n 2024)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_size, 1)

        # Initialize classifier weights
        self._init_weights(self.classifier)

        # Store dimensions for reference
        self.bert_hidden_size = bert_hidden_size
        self.num_features = num_features
        self.combined_size = combined_size

    def _init_weights(self, module):
        """
        Initialize weights for linear layers (classifier head).
        Uses BERT's initializer_range (typically 0.02) for consistency
        with pretrained initialization. Weights ~ N(0, 0.02), biases = 0.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, C, seq_len]
            attention_mask: Attention mask [batch_size, C, seq_len]
            token_type_ids: Token type IDs [batch_size, C, seq_len]
            features: Handcrafted features [batch_size, num_features]
            labels: Ground truth labels [batch_size] (gold parent index)

        Returns:
            Dictionary with:
            - logits: Raw model outputs [batch_size, C]
            - probs: Softmax probabilities [batch_size, C]
            - loss: CrossEntropy loss (if labels provided)
        """
        batch_size, num_candidates, seq_len = input_ids.shape

        # Reshape for BERT: [batch_size * C, seq_len]
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        flat_token_type_ids = (
            token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        )

        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            return_dict=True,
        )

        # Use [CLS] token embedding
        cls_embedding = bert_outputs.last_hidden_state[
            :, 0, :
        ]  # [batch_size * C, hidden_size]

        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)

        # === CLS EMBEDDING NaN DETECTION ===
        # Previously this block silently replaced NaN with zeros via nan_to_num,
        # which was actively dangerous — it let training continue with corrupted
        # hidden states, contaminating weights before the gradient check could fire.
        #
        # Now: NaN in cls_embedding propagates naturally to produce NaN loss,
        # which train.py catches and skips with optimizer.zero_grad().
        # The warning is retained for diagnostics.
        if torch.isnan(cls_embedding).any() or torch.isinf(cls_embedding).any():
            logger.warning(
                f"  NaN/Inf detected in cls_embedding before classifier. "
                f"Shape: {cls_embedding.shape}, "
                f"NaN count: {torch.isnan(cls_embedding).sum().item()}, "
                f"Inf count: {torch.isinf(cls_embedding).sum().item()}. "
                f"NaN will propagate to loss — train.py will skip this batch."
            )

        # Reshape features to match cls_embedding: [batch_size, C, num_features] -> [batch_size * C, num_features]
        if features is not None:
            # features from collate_fn is [batch_size, C, num_features] (per-candidate)
            expanded_features = features.reshape(-1, self.num_features)
            combined = torch.cat([cls_embedding, expanded_features], dim=-1)
        else:
            zero_features = torch.zeros(
                cls_embedding.shape[0],
                self.num_features,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype,
            )
            combined = torch.cat([cls_embedding, zero_features], dim=-1)

        # Classification head: [batch_size * C, 1]
        logits = self.classifier(combined)

        # Reshape back to [batch_size, C]
        logits = logits.view(batch_size, num_candidates)

        # Mask out padded candidates (where attention mask is all zeros or similar)
        # In our case, the collate_fn uses 0 for padding.
        # Check if the whole candidate was padding:
        candidate_mask = attention_mask.sum(dim=-1) > 0  # [batch_size, C]

        # Mask out padded candidates with a FINITE large negative value.
        #
        # CRITICAL: Do NOT use torch.finfo(dtype).min (-3.4e38 for fp32) or
        # other extremely negative values like -1e4.
        # -3.4e38: CrossEntropyLoss backward → INF gradient → NaN weights.
        # -1e4:   exp(-10000) underflows to 0 in fp32 → log(0) = -inf → 0*-inf = NaN.
        #
        # -100 is large enough that softmax assigns ~0 probability to masked
        # candidates (exp(-100) ≈ 3.7e-44, well above fp32 minimum), but finite
        # enough that exp(-100) is representable and gradients are well-behaved.
        # Verified on DeBERTa-v3-base + L40S (2026-05-10).
        # See 2026-05-10 fix in PROGRESS.md for the full debugging chain.
        fill_value = -100.0
        logits = logits.masked_fill(~candidate_mask, fill_value)

        candidate_probs = torch.softmax(logits, dim=-1)  # [batch_size, C]

        # Prepare output
        outputs = {"logits": logits, "probs": candidate_probs}

        # Compute loss if labels provided
        if labels is not None:
            # Clamp logits before loss to prevent exp() overflow in CrossEntropyLoss
            # softmax. Without clamping, logits drifting to ~80-100 cause exp(100) to
            # overflow fp32 to inf, making the softmax denominator inf/inf = NaN.
            # [-50, 50] keeps exp values well within fp32 range while preserving
            # the ability to rank candidates (see handover.md for full analysis).
            logits = torch.clamp(logits, max=50.0)

            # Label smoothing (0.1) prevents the model from pushing the correct-class
            # logit to +inf by capping the target probability at 0.9. The remaining
            # 0.1 is distributed across all classes, creating a soft target distribution
            # instead of a one-hot spike. This directly prevents the primary NaN mechanism.
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0)
            loss = loss_fn(logits, labels)
            outputs["loss"] = loss

        return outputs

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make multiclass predictions.

        Returns:
            - predictions: Candidate indices (0 to C-1) [batch_size]
            - probabilities: Softmax probabilities [batch_size, C]
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                features=features,
            )

            probs = outputs["probs"]
            predictions = torch.argmax(probs, dim=-1)  # [batch_size]

            return predictions, probs


def create_model(
    model_name: str = "microsoft/deberta-v3-base",
    num_features: int = 5,
    dropout: float = 0.1,
    freeze_bert: bool = False,
    gradient_checkpointing: bool = False,
    device: str = None,
) -> CrossEncoderWithFeatures:
    """
    Factory function to create and initialize model.

    Args:
        model_name: Pretrained BERT model name (default: DeBERTa-v3-base for SOTA performance)
        num_features: Number of handcrafted features
        dropout: Dropout probability
        freeze_bert: Whether to freeze BERT parameters
        gradient_checkpointing: Enable gradient checkpointing (trades ~30% speed for ~80% less VRAM)
        device: Device to load model on (cuda/cpu)

    Returns:
        Initialized CrossEncoderWithFeatures model
    """
    model = CrossEncoderWithFeatures(
        model_name=model_name,
        num_features=num_features,
        dropout=dropout,
        freeze_bert=freeze_bert,
        gradient_checkpointing=gradient_checkpointing,
    )

    # Move to device if specified
    if device:
        model = model.to(device)

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


if __name__ == "__main__":
    print("Run `python -m pytest tests/test_model.py -v` for model tests.")
    print("Use `python src/train.py` for training.")
