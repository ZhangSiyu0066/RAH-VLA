"""
Multi-scale Vision-language Alignment Mechanism (MS-VLAM) Projector
Implements object-level, local-region-level, and global-level alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class MSVLAMProjector(nn.Module):
    """
    Multi-scale Vision-language Alignment Mechanism (MS-VLAM)
    
    Implements three-level alignment:
    1. Object-level: Individual object to entity alignment
    2. Local-region-level: Region to phrase alignment  
    3. Global-level: Full image to sentence alignment
    
    Corresponds to Eq. (13) in paper:
    L_align = α·L_obj + β·L_reg + γ·L_glob
    """
    
    def __init__(
        self,
        vision_hidden_size: int = 1024,
        text_hidden_size: int = 4096,
        alignment_dim: int = 768,
        num_sam_regions: int = 16,
        spp_pool_sizes: List[int] = [1, 2, 4],
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        mu: float = 0.5,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.alignment_dim = alignment_dim
        self.num_sam_regions = num_sam_regions
        self.spp_pool_sizes = spp_pool_sizes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.temperature = temperature
        
        # Object-level projection (Eq. 4, 5)
        self.obj_visual_proj = nn.Linear(vision_hidden_size, alignment_dim)
        self.obj_text_proj = nn.Linear(text_hidden_size, alignment_dim)
        
        # Local-region-level projection
        self.reg_visual_proj = nn.Linear(vision_hidden_size, alignment_dim)
        self.reg_text_proj = nn.Linear(text_hidden_size, alignment_dim)
        
        # Global-level projection (Eq. 12)
        self.glob_visual_proj = nn.Linear(vision_hidden_size, alignment_dim)
        self.glob_text_proj = nn.Linear(text_hidden_size, alignment_dim)
        
        # Spatial Pyramid Pooling for global features
        self.spp_convs = nn.ModuleList([
            nn.Conv2d(vision_hidden_size, vision_hidden_size, kernel_size=1)
            for _ in spp_pool_sizes
        ])
        
        # Layer normalization for stability
        self.obj_norm = nn.LayerNorm(alignment_dim)
        self.reg_norm = nn.LayerNorm(alignment_dim)
        self.glob_norm = nn.LayerNorm(alignment_dim)
    
    def extract_object_features(
        self,
        vision_features: torch.Tensor,
        bbox_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract object-level visual features
        
        Implements Eq. (4): v_obj^(p) = f_v(RoI(V, B_p))
        
        Args:
            vision_features: Global vision features (B, C, H, W)
            bbox_features: Optional pre-computed bounding box features
            
        Returns:
            Object-level features (P, alignment_dim)
        """
        if bbox_features is not None:
            # Use pre-computed bbox features
            obj_features = self.obj_visual_proj(bbox_features)
        else:
            # Simple approach: use global features as object features
            # In practice, this would use object detector (Faster R-CNN, DETR)
            B, C, H, W = vision_features.shape
            obj_features = F.adaptive_avg_pool2d(vision_features, (1, 1))
            obj_features = obj_features.view(B, C)
            obj_features = self.obj_visual_proj(obj_features)
        
        return self.obj_norm(obj_features)
    
    def extract_region_features(
        self,
        vision_features: torch.Tensor,
        sam_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract local-region-level features using SAM segmentation
        
        Args:
            vision_features: Vision features (B, C, H, W)
            sam_masks: SAM segmentation masks (K, H, W)
            
        Returns:
            Region features (K, alignment_dim)
        """
        B, C, H, W = vision_features.shape
        
        if sam_masks is not None:
            # Use SAM masks to extract region features
            K = sam_masks.shape[0]
            reg_features = []
            
            for k in range(K):
                mask = sam_masks[k:k+1]  # (1, H, W)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                
                # Apply mask and pool
                masked_features = vision_features * mask
                reg_feat = F.adaptive_avg_pool2d(masked_features, (1, 1))
                reg_feat = reg_feat.view(C)
                
                # Project
                reg_feat = self.reg_visual_proj(reg_feat)
                reg_features.append(reg_feat)
            
            reg_features = torch.stack(reg_features, dim=0)
        else:
            # Simple approach: use spatial patches as regions
            # In practice, this would use SAM or similar
            patch_size = H // int(np.sqrt(self.num_sam_regions))
            patches = vision_features.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
            patches = patches.permute(0, 2, 1, 3, 4).contiguous()
            patches = patches.view(-1, C, patch_size, patch_size)
            
            # Pool and project
            reg_features = F.adaptive_avg_pool2d(patches, (1, 1))
            reg_features = reg_features.view(-1, C)
            reg_features = self.reg_visual_proj(reg_features)
        
        return self.reg_norm(reg_features)
    
    def extract_global_features(
        self,
        vision_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract global-level features using Spatial Pyramid Pooling (SPP)
        
        Implements SPP(V) from Eq. (12)
        
        Args:
            vision_features: Vision features (B, C, H, W)
            
        Returns:
            Global features (B, alignment_dim)
        """
        B, C, H, W = vision_features.shape
        
        # Spatial Pyramid Pooling
        pooled_features = []
        for i, pool_size in enumerate(self.spp_pool_sizes):
            # Apply 1x1 conv
            feat = self.spp_convs[i](vision_features)
            
            # Pool to different sizes
            pooled = F.adaptive_avg_pool2d(feat, (pool_size, pool_size))
            pooled = pooled.view(B, -1)
            pooled_features.append(pooled)
        
        # Concatenate multi-scale pooled features
        global_feat = torch.cat(pooled_features, dim=1)
        
        # Project to alignment space
        global_feat = self.glob_visual_proj(global_feat)
        
        return self.glob_norm(global_feat)
    
    def compute_object_alignment_loss(
        self,
        obj_visual_features: torch.Tensor,
        obj_text_features: torch.Tensor,
        iou_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute object-level alignment loss
        
        Implements Eq. (7):
        L_obj = 1 - (1/P) Σ w_p · cos(v_obj^(p), o_obj^(p))
        
        Args:
            obj_visual_features: (P, D)
            obj_text_features: (P, D)
            iou_weights: Optional IoU-based weights (P,)
            
        Returns:
            Object alignment loss
        """
        P = obj_visual_features.shape[0]
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            obj_visual_features.unsqueeze(1),
            obj_text_features.unsqueeze(0),
            dim=2
        )
        
        # Use diagonal (matched pairs)
        similarities = torch.diagonal(cos_sim)
        
        # Apply IoU weights if provided
        if iou_weights is not None:
            # Normalize weights
            weights = iou_weights / iou_weights.sum()
            similarities = similarities * weights
        else:
            weights = torch.ones_like(similarities) / P
        
        # Compute loss (Eq. 7)
        loss = 1 - similarities.mean()
        
        return loss
    
    def compute_region_alignment_loss(
        self,
        reg_visual_features: torch.Tensor,
        reg_text_features: torch.Tensor,
        region_phrase_mapping: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute local-region-level alignment loss
        
        Implements Eq. (8)-(11):
        L_reg = μ·L_reg^hard + (1-μ)·L_reg^NCE
        
        Args:
            reg_visual_features: (K, D) - region features
            reg_text_features: (M, D) - phrase features
            region_phrase_mapping: Optional mapping from regions to phrases
            
        Returns:
            Region alignment loss
        """
        K, M = reg_visual_features.shape[0], reg_text_features.shape[0]
        
        # Hard matching loss (Eq. 8)
        if region_phrase_mapping is not None:
            # Use provided mapping
            matched_phrases = reg_text_features[region_phrase_mapping]
            cos_sim_hard = F.cosine_similarity(
                reg_visual_features,
                matched_phrases,
                dim=1
            )
            L_reg_hard = 1 - cos_sim_hard.mean()
        else:
            # Simple approach: use diagonal (assume K == M)
            if K == M:
                cos_sim_hard = F.cosine_similarity(
                    reg_visual_features,
                    reg_text_features,
                    dim=1
                )
                L_reg_hard = 1 - cos_sim_hard.mean()
            else:
                # If sizes don't match, use average
                L_reg_hard = 0.0
        
        # Contrastive learning loss (Eq. 9-10)
        # Compute similarity matrix
        sim_matrix = torch.matmul(reg_visual_features, reg_text_features.T) / self.temperature
        
        # InfoNCE loss (Eq. 10)
        # For each region, find the matching phrase
        if region_phrase_mapping is not None:
            # Use provided positive indices
            pos_indices = torch.tensor(region_phrase_mapping, device=reg_visual_features.device)
            pos_sim = sim_matrix[torch.arange(K), pos_indices]
            
            # Compute log_softmax over all phrases
            log_probs = F.log_softmax(sim_matrix, dim=1)
            L_reg_nce = -log_probs[torch.arange(K), pos_indices].mean()
        else:
            # Assume diagonal
            pos_sim = torch.diagonal(sim_matrix)
            log_probs = F.log_softmax(sim_matrix, dim=1)
            L_reg_nce = -torch.diagonal(log_probs).mean()
        
        # Combined loss (Eq. 11)
        L_reg = self.mu * L_reg_hard + (1 - self.mu) * L_reg_nce
        
        return L_reg
    
    def compute_global_alignment_loss(
        self,
        global_visual_features: torch.Tensor,
        global_text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute global-level alignment loss
        
        Implements Eq. (12):
        L_glob = 1 - cos(g, t_[CLS])
        
        Args:
            global_visual_features: (B, D)
            global_text_features: (B, D) - CLS token features
            
        Returns:
            Global alignment loss
        """
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            global_visual_features,
            global_text_features,
            dim=1
        )
        
        # Loss (Eq. 12)
        loss = 1 - cos_sim.mean()
        
        return loss
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: Dict[str, torch.Tensor],
        bbox_features: Optional[torch.Tensor] = None,
        sam_masks: Optional[torch.Tensor] = None,
        iou_weights: Optional[torch.Tensor] = None,
        region_phrase_mapping: Optional[List[int]] = None,
        return_losses: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MS-VLAM
        
        Args:
            vision_features: Global vision features (B, C, H, W)
            text_features: Dictionary containing:
                - 'object': Object text features (P, D_text)
                - 'phrase': Phrase text features (M, D_text)
                - 'global': Global text features (B, D_text)
            bbox_features: Optional bbox features for object alignment
            sam_masks: Optional SAM masks for region alignment
            iou_weights: Optional IoU weights for object loss
            region_phrase_mapping: Optional region-phrase mapping
            return_losses: Whether to compute and return losses
            
        Returns:
            Dictionary containing:
                - 'object_features': Object-level visual features
                - 'region_features': Region-level visual features
                - 'global_features': Global visual features
                - 'loss_obj': Object alignment loss (if return_losses)
                - 'loss_reg': Region alignment loss (if return_losses)
                - 'loss_glob': Global alignment loss (if return_losses)
                - 'loss_align': Total alignment loss (if return_losses)
        """
        outputs = {}
        
        # Extract multi-scale visual features
        obj_visual = self.extract_object_features(vision_features, bbox_features)
        reg_visual = self.extract_region_features(vision_features, sam_masks)
        glob_visual = self.extract_global_features(vision_features)
        
        outputs['object_features'] = obj_visual
        outputs['region_features'] = reg_visual
        outputs['global_features'] = glob_visual
        
        if return_losses:
            # Project text features to alignment space
            obj_text = self.obj_text_proj(text_features['object'])
            reg_text = self.reg_text_proj(text_features['phrase'])
            glob_text = self.glob_text_proj(text_features['global'])
            
            # Compute alignment losses
            L_obj = self.compute_object_alignment_loss(
                obj_visual, obj_text, iou_weights
            )
            L_reg = self.compute_region_alignment_loss(
                reg_visual, reg_text, region_phrase_mapping
            )
            L_glob = self.compute_global_alignment_loss(
                glob_visual, glob_text
            )
            
            # Total alignment loss (Eq. 13)
            L_align = self.alpha * L_obj + self.beta * L_reg + self.gamma * L_glob
            
            outputs['loss_obj'] = L_obj
            outputs['loss_reg'] = L_reg
            outputs['loss_glob'] = L_glob
            outputs['loss_align'] = L_align
        
        return outputs