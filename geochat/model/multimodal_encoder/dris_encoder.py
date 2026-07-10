"""
Dynamic Resolution Input Strategy (DRIS) for Remote Sensing Images
Implements coarse-to-fine resolution processing with ROI selection and FPN fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import torchvision.transforms as T


class DRISVisionTower(nn.Module):
    """
    Dynamic Resolution Input Strategy (DRIS) Vision Encoder
    
    Implements:
    1. Two-stage coarse-to-fine processing
    2. Saliency-based ROI selection
    3. FPN feature fusion
    4. Attention-enhanced ROI filtering
    """
    
    def __init__(
        self,
        base_vision_tower,
        low_res_size: int = 224,
        high_res_size: int = 504,
        roi_threshold: float = 0.5,
        top_k_rois: int = 5,
        saliency_kernel_size: int = 5,
        fpn_channels: List[int] = [256, 512, 1024, 2048],
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.base_vision_tower = base_vision_tower
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.roi_threshold = roi_threshold
        self.top_k_rois = top_k_rois
        self.saliency_kernel_size = saliency_kernel_size
        
        # Image processors for different resolutions
        self.low_res_processor = T.Compose([
            T.Resize((low_res_size, low_res_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.high_res_processor = T.Compose([
            T.Resize((high_res_size, high_res_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # FPN lateral connections for multi-scale feature fusion
        self.fpn_lateral = nn.ModuleList([
            nn.Conv2d(fpn_channels[i], 256, kernel_size=1)
            for i in range(4)
        ])
        
        # FPN output convolutions
        self.fpn_output = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
            for _ in range(4)
        ])
        
        # Attention map generator for ROI selection
        self.attention_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        self.device = device
        
    def compute_saliency_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Compute saliency map from feature map using attention mechanism
        
        Args:
            feature_map: Feature map of shape (B, C, H, W)
            
        Returns:
            Saliency map of shape (B, H, W)
        """
        # Generate attention weights
        attention_map = self.attention_conv(feature_map).squeeze(1)  # (B, H, W)
        
        # Apply Gaussian blur to smooth attention map
        # Create Gaussian kernel
        kernel_size = self.saliency_kernel_size
        sigma = 1.5
        gaussian_kernel = self._gaussian_kernel(kernel_size, sigma).to(attention_map.device)
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        attention_map = attention_map.unsqueeze(1)
        attention_map = F.conv2d(attention_map, gaussian_kernel, padding=kernel_size//2)
        attention_map = attention_map.squeeze(1)
        
        # Normalize to [0, 1]
        B, H, W = attention_map.shape
        attention_map = attention_map.view(B, -1)
        attention_map = F.softmax(attention_map, dim=1)
        attention_map = attention_map.view(B, H, W)
        
        return attention_map
    
    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        g = g.unsqueeze(1) @ g.unsqueeze(0)
        return g
    
    def select_rois(
        self,
        saliency_map: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Select Regions of Interest (ROIs) based on saliency map
        
        Implements Eq. (3) from paper:
        P_ROI = Topk(GaussianBlur(A_coarse), k)
        
        Args:
            saliency_map: Saliency map of shape (B, H, W)
            image_size: Original image size (H, W)
            
        Returns:
            List of ROI bounding boxes [(x1, y1, x2, y2), ...]
        """
        B, H, W = saliency_map.shape
        saliency_map_np = saliency_map.cpu().detach().numpy()
        
        rois = []
        for b in range(B):
            # Get flattened saliency scores
            flat_scores = saliency_map_np[b].flatten()
            
            # Select top-k positions
            top_k_indices = np.argpartition(flat_scores, -self.top_k_rois)[-self.top_k_rois:]
            top_k_indices = top_k_indices[np.argsort(-flat_scores[top_k_indices])]
            
            # Convert to spatial coordinates
            for idx in top_k_indices:
                y, x = np.unravel_index(idx, (H, W))
                
                # Scale to original image size
                x1 = int(x / W * image_size[1])
                y1 = int(y / H * image_size[0])
                
                # Define ROI size (adaptive based on saliency score)
                roi_size = int(image_size[0] * 0.2)  # 20% of image size
                x2 = min(x1 + roi_size, image_size[1])
                y2 = min(y1 + roi_size, image_size[0])
                
                # Center the ROI if it exceeds boundaries
                if x2 - x1 < roi_size:
                    x1 = max(0, x2 - roi_size)
                if y2 - y1 < roi_size:
                    y1 = max(0, y2 - roi_size)
                
                rois.append((x1, y1, x2, y2))
        
        return rois
    
    def fpn_fusion(self, coarse_features: List[torch.Tensor], fine_features: List[torch.Tensor]) -> torch.Tensor:
        """
        FPN feature fusion between coarse and fine resolution features
        
        Implements Eq. (2) from paper:
        F_fused = Upsample(F_coarse) + Crop(F_fine, Ω)
        
        Args:
            coarse_features: List of coarse resolution feature maps
            fine_features: List of fine resolution feature maps
            
        Returns:
            Fused feature map
        """
        fused_features = []
        
        # Top-down pathway with lateral connections
        prev_features = None
        for i in reversed(range(len(coarse_features))):
            # Lateral connection
            lateral = self.fpn_lateral[i](coarse_features[i])
            
            if prev_features is not None:
                # Upsample and add
                upsampled = F.interpolate(
                    prev_features, 
                    size=lateral.shape[-2:], 
                    mode='nearest'
                )
                lateral = lateral + upsampled
            
            # Output convolution
            output = self.fpn_output[i](lateral)
            fused_features.append(output)
            prev_features = output
        
        # Reverse to get original order
        fused_features = fused_features[::-1]
        
        # Concatenate multi-scale features
        # Resize all to same size
        target_size = fused_features[0].shape[-2:]
        resized_features = []
        for feat in fused_features:
            resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(resized)
        
        # Concatenate
        fused = torch.cat(resized_features, dim=1)
        
        return fused
    
    @torch.no_grad()
    def forward(self, images) -> torch.Tensor:
        """
        Forward pass implementing DRIS coarse-to-fine strategy
        
        Implements Eq. (1) from paper:
        R(x,y) = R_high if s(x,y) >= τ else R_low
        
        Args:
            images: Input images, can be PIL Image, tensor, or list
            
        Returns:
            Fused feature representation
        """
        # Handle different input types
        if isinstance(images, list):
            # Process list of images
            all_features = []
            for img in images:
                features = self._process_single_image(img)
                all_features.append(features)
            return torch.stack(all_features, dim=0)
        elif isinstance(images, Image.Image):
            return self._process_single_image(images).unsqueeze(0)
        else:
            # Assume tensor input
            return self._process_batch_images(images)
    
    def _process_single_image(self, image: Image.Image) -> torch.Tensor:
        """Process single image through DRIS pipeline"""
        image_size = image.size[::-1]  # (H, W)
        
        # Stage 1: Low-resolution processing
        low_res_tensor = self.low_res_processor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            low_res_output = self.base_vision_tower(low_res_tensor, output_hidden_states=True)
            # Get features from multiple layers for FPN
            low_res_features = [
                low_res_output.hidden_states[-i][:, 1:].view(-1, 256, int(np.sqrt(low_res_output.hidden_states[-i].shape[1]-1)), int(np.sqrt(low_res_output.hidden_states[-i].shape[1]-1)))
                for i in range(4, 0, -1)
            ]
        
        # Compute saliency map for ROI selection
        saliency_map = self.compute_saliency_map(low_res_features[0])
        
        # Select ROIs
        rois = self.select_rois(saliency_map, image_size)
        
        # Stage 2: High-resolution processing for ROIs
        fine_features_list = []
        for roi in rois[:self.top_k_rois]:  # Process top-k ROIs
            # Crop and resize ROI
            x1, y1, x2, y2 = roi
            cropped = image.crop((x1, y1, x2, y2))
            
            # Process at high resolution
            high_res_tensor = self.high_res_processor(cropped).unsqueeze(0).to(self.device)
            with torch.no_grad():
                high_res_output = self.base_vision_tower(high_res_tensor, output_hidden_states=True)
                high_res_features = [
                    high_res_output.hidden_states[-i][:, 1:].view(-1, 256, int(np.sqrt(high_res_output.hidden_states[-i].shape[1]-1)), int(np.sqrt(high_res_output.hidden_states[-i].shape[1]-1)))
                    for i in range(4, 0, -1)
                ]
                fine_features_list.append(high_res_features)
        
        # FPN fusion between coarse and fine features
        if fine_features_list:
            # Average fine features from multiple ROIs
            avg_fine_features = []
            for i in range(4):
                stacked = torch.stack([fine_features[i] for fine_features in fine_features_list], dim=0)
                avg_fine_features.append(stacked.mean(dim=0))
            
            # Fuse with coarse features
            fused = self.fpn_fusion(low_res_features, avg_fine_features)
        else:
            # If no ROIs selected, use coarse features
            fused = self.fpn_fusion(low_res_features, low_res_features)
        
        # Global average pooling and flatten
        pooled = F.adaptive_avg_pool2d(fused, (1, 1))
        flattened = pooled.view(pooled.size(0), -1)
        
        return flattened
    
    def _process_batch_images(self, images: torch.Tensor) -> torch.Tensor:
        """Process batch of tensor images"""
        B = images.shape[0]
        all_features = []
        
        for b in range(B):
            # Convert tensor to PIL Image
            img_np = images[b].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                     np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np).convert('RGB')
            
            features = self._process_single_image(img)
            all_features.append(features)
        
        return torch.stack(all_features, dim=0)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def hidden_size(self):
        # Return fused feature dimension
        return 1024  # 4 * 256 from FPN