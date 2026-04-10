"""
Universal Brain Encoder with DINOv3 (HF) backbone + LoRA multi-depth features.
Reuses voxel store, cross-attention stack, and loss from universal_brain_encoder.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Load universal_brain_encoder/model.py under a distinct module name so
# `import model` in train.py (this package) does not create a circular import.
_UB = Path(__file__).resolve().parent.parent / "universal_brain_encoder"
_ube_path = _UB / "model.py"
_spec = importlib.util.spec_from_file_location("universal_brain_encoder_model", _ube_path)
_ube_model = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ube_model)

BrainEncoderLoss = _ube_model.BrainEncoderLoss
CrossAttentionBlock = _ube_model.CrossAttentionBlock
VoxelEmbeddingStore = _ube_model.VoxelEmbeddingStore
count_parameters = _ube_model.count_parameters

from dinov3_backbone import LoRADINOv3  # noqa: E402


class UniversalBrainEncoderDINOv3(nn.Module):
    """
    Same head as UniversalBrainEncoder (cross-attention + voxel embeddings),
    with LoRADINOv3 (HF DINOv3 ViT) as the vision trunk.
    """

    def __init__(
        self,
        hf_model_id: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        embedding_dim: int = 256,
        projection_dim: int = 128,
        lora_rank: int = 16,
        num_layers: int = 5,
        mlp_hidden_mult: int = 2,
        image_size: int = 224,
        patch_size: int = 16,
        trust_remote_code: bool = True,
        layer_selection: str = "paper_proportional",
        dropout: float = 0.0,
        lora_dropout: float = 0.0,
        lora_mode: str = "block",
        projection_mlp: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hf_model_id = hf_model_id

        self.feature_extractor = LoRADINOv3(
            model_id=hf_model_id,
            projection_dim=projection_dim,
            lora_rank=lora_rank,
            freeze_backbone=True,
            trust_remote_code=trust_remote_code,
            layer_selection=layer_selection,
            lora_dropout=lora_dropout,
            lora_mode=lora_mode,
            projection_mlp=projection_mlp,
        )

        self.num_patches = (image_size // patch_size) ** 2

        self.voxel_store = VoxelEmbeddingStore(embedding_dim=embedding_dim)

        self.cross_attention = CrossAttentionBlock(
            embedding_dim=embedding_dim,
            num_patches=self.num_patches,
            num_layers=num_layers,
            feature_dim=projection_dim,
            mlp_hidden_mult=mlp_hidden_mult,
            dropout=dropout,
        )

    def register_subject(self, subject_id: str, num_voxels: int) -> None:
        self.voxel_store.register_subject(subject_id, num_voxels)
        # Newly created embeddings are always on CPU; move them to wherever
        # the rest of the model already lives (no-op if still on CPU).
        try:
            device = next(self.feature_extractor.parameters()).device
            emb = self.voxel_store.embeddings[subject_id]
            self.voxel_store.embeddings[subject_id] = nn.Parameter(emb.data.to(device))
        except StopIteration:
            pass

    def forward(
        self,
        pixel_values: torch.Tensor,
        subject_id: str,
        voxel_indices: torch.Tensor,
    ) -> torch.Tensor:
        features = self.feature_extractor(pixel_values)
        if features.shape[2] != self.num_patches:
            raise ValueError(
                f"Expected P={self.num_patches} patches, got {features.shape[2]}. "
                "Match --image_size / --patch_size to the processor / backbone."
            )
        voxel_embs = self.voxel_store.get_embeddings(subject_id, voxel_indices)
        return self.cross_attention(features, voxel_embs)

    def predict_all_voxels(
        self,
        pixel_values: torch.Tensor,
        subject_id: str,
        chunk_size: int = 5000,
    ) -> torch.Tensor:
        features = self.feature_extractor(pixel_values)
        all_embs = self.voxel_store.get_all_embeddings(subject_id)
        num_voxels = all_embs.shape[0]

        all_preds = []
        for start in range(0, num_voxels, chunk_size):
            end = min(start + chunk_size, num_voxels)
            chunk_embs = all_embs[start:end]
            preds = self.cross_attention(features, chunk_embs)
            all_preds.append(preds)

        return torch.cat(all_preds, dim=0)


__all__ = [
    "UniversalBrainEncoderDINOv3",
    "BrainEncoderLoss",
    "count_parameters",
]
