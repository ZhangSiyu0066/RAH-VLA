"""
Multi-scale Vision-language Alignment Loss Functions
Implements caption loss and alignment loss integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiScaleAlignmentLoss(nn.Module):
    """
    Multi-scale alignment loss combining caption and alignment losses
    
    Implements Eq. (15) from paper:
    L = L_caption + δ·L_align
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.delta = delta
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        # Caption loss (Eq. 14)
        self.caption_loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        alignment_losses: Optional[Dict[str, torch.Tensor]] = None,
        vision_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining caption and alignment losses
        
        Implements Eq. (15):
        L = L_caption + δ·L_align
        
        Args:
            logits: Model output logits (B, T, vocab_size)
            labels: Ground truth labels (B, T)
            alignment_losses: Optional dictionary containing:
                - 'loss_align': Total alignment loss
                - 'loss_obj': Object alignment loss
                - 'loss_reg': Region alignment loss
                - 'loss_glob': Global alignment loss
            vision_features: Optional vision features for caption loss
            text_features: Optional text features for caption loss
            
        Returns:
            Dictionary containing:
                - 'loss': Total loss
                - 'loss_caption': Caption generation loss
                - 'loss_align': Alignment loss (if provided)
                - 'loss_obj': Object alignment loss (if provided)
                - 'loss_reg': Region alignment loss (if provided)
                - 'loss_glob': Global alignment loss (if provided)
        """
        # Compute caption loss (Eq. 14)
        # L_caption = -Σ log P_L(x_k | V, x_<t)
        loss_caption = self.caption_loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Compute total loss
        if alignment_losses is not None and 'loss_align' in alignment_losses:
            loss_align = alignment_losses['loss_align']
            loss = loss_caption + self.delta * loss_align
        else:
            loss = loss_caption
        
        outputs = {
            'loss': loss,
            'loss_caption': loss_caption
        }
        
        # Include alignment losses if provided
        if alignment_losses is not None:
            outputs['loss_align'] = alignment_losses.get('loss_align', torch.tensor(0.0))
            outputs['loss_obj'] = alignment_losses.get('loss_obj', torch.tensor(0.0))
            outputs['loss_reg'] = alignment_losses.get('loss_reg', torch.tensor(0.0))
            outputs['loss_glob'] = alignment_losses.get('loss_glob', torch.tensor(0.0))
        
        return outputs


class CaptionLoss(nn.Module):
    """
    Caption generation loss
    
    Implements Eq. (14):
    L_caption = -Σ log P_L(x_k | V, x_<t)
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vision_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute caption generation loss
        
        Args:
            logits: Model output logits (B, T, vocab_size)
            labels: Ground truth labels (B, T)
            vision_context: Optional vision features for context-aware loss
            
        Returns:
            Caption loss
        """
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        return loss


class ContrastiveAlignmentLoss(nn.Module):
    """
    Contrastive learning loss for alignment
    
    Can be used for InfoNCE-based alignment
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        positive_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive alignment loss
        
        Args:
            visual_features: Visual features (B, D)
            text_features: Text features (B, D)
            positive_mask: Optional mask for positive pairs (B, B)
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        visual_features = F.normalize(visual_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(visual_features, text_features.T) / self.temperature
        
        # If no positive mask provided, assume diagonal positives
        if positive_mask is None:
            positive_mask = torch.eye(
                similarity.shape[0],
                device=similarity.device,
                dtype=torch.bool
            )
        
        # Compute InfoNCE loss
        log_probs = F.log_softmax(similarity, dim=1)
        
        # For each visual feature, find matching text feature
        pos_indices = positive_mask.argmax(dim=1)
        loss = -log_probs[torch.arange(similarity.shape[0]), pos_indices]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AlignmentLossWrapper(nn.Module):
    """
    Wrapper for combining multiple alignment losses
    
    Provides flexible combination of different alignment losses
    """
    
    def __init__(
        self,
        loss_weights: Dict[str, float],
        ignore_index: int = -100
    ):
        """
        Args:
            loss_weights: Dictionary mapping loss names to weights
                        e.g., {'caption': 1.0, 'align': 0.5}
        """
        super().__init__()
        
        self.loss_weights = loss_weights
        self.ignore_index = ignore_index
        
        # Initialize loss functions
        if 'caption' in loss_weights:
            self.caption_loss = CaptionLoss(
                ignore_index=ignore_index
            )
        
        if 'contrastive' in loss_weights:
            self.contrastive_loss = ContrastiveAlignmentLoss()
    
    def forward(
        self,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        alignment_losses: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted combination of losses
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            visual_features: Visual features for contrastive loss
            text_features: Text features for contrastive loss
            alignment_losses: Pre-computed alignment losses
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Compute caption loss
        if logits is not None and labels is not None and 'caption' in self.loss_weights:
            loss_caption = self.caption_loss(logits, labels)
            losses['loss_caption'] = loss_caption
            total_loss += self.loss_weights['caption'] * loss_caption
        
        # Compute contrastive loss
        if visual_features is not None and text_features is not None and 'contrastive' in self.loss_weights:
            loss_contrastive = self.contrastive_loss(visual_features, text_features)
            losses['loss_contrastive'] = loss_contrastive
            total_loss += self.loss_weights['contrastive'] * loss_contrastive
        
        # Add pre-computed alignment losses
        if alignment_losses is not None:
            for key, loss_value in alignment_losses.items():
                if key in self.loss_weights:
                    losses[key] = loss_value
                    total_loss += self.loss_weights[key] * loss_value
                else:
                    losses[key] = loss_value
        
        losses['loss'] = total_loss
        
        return losses