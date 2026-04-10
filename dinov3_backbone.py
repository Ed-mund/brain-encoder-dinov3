"""
LoRA-wrapped DINOv3 ViT backbone via Hugging Face transformers.

lora_mode controls where adaptation is injected (see LoRADINOv3 docstring).
A separate forward hook captures each selected block's full output
(after attention + MLP + residuals) for feature extraction.

Architecture:
  - Backbone is fully frozen.
  - For each of the 5 selected blocks, one or two LoRA adapters are applied
    depending on lora_mode, and the final block hidden state is captured.
  - The five captured states are projected to projection_dim.
  - Output shape: (B, 5, P, C)

Layer selection strategies
--------------------------
"paper_proportional"  (default)
    Mirrors the paper's DINOv2 layer choice [1, 6, 12, 18, 24] of 24 blocks,
    proportionally scaled to any depth.  Fractions ≈ [0%, 22%, 48%, 74%, 100%].
    For ViT-H (32 blocks) → blocks [0, 7, 15, 23, 31].
    Critically includes the first block (low-level edge/texture features needed
    by V1/V2 voxels).

"top_biased"
    Concentrates on the deeper half: [25%, 50%, 66%, 83%, 100%] of depth.
    Misses early visual features — kept for backward-compat / ablation only.

"even"
    Evenly spaced across all layers — original DINOv2 reference approach.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """Low-rank adaptation: delta = dropout(x @ A) @ B.

    Dropout on the low-rank intermediate (after A, before B) acts as stochastic
    LoRA regularisation.  It prevents the backbone adaptation from memorising
    training subjects, which is the primary cause of the validation correlation
    declining after epoch ~9 without regularisation.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x @ self.lora_A) @ self.lora_B


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _five_block_indices(num_blocks: int, strategy: str = "paper_proportional") -> List[int]:
    """
    Return 5 0-based block indices to extract multi-scale features from.

    strategy="paper_proportional"  (default)
        Scales the paper's DINOv2 layer choice [1, 6, 12, 18, 24] of 24 blocks
        proportionally to any backbone depth.
        Fractions ≈ [0.0, 0.217, 0.478, 0.739, 1.0].
        For ViT-H (32 blocks, 0-indexed max=31) → [0, 7, 15, 23, 31].
        Includes the very first block, providing low-level edge/texture features
        that are essential for early visual cortex (V1/V2) voxels.

    strategy="top_biased"
        Concentrates on deeper layers [25%, 50%, 66%, 83%, 100%].
        Skips early layers — kept for ablation / backward compatibility.

    strategy="even"
        Evenly spaced — original DINOv2 reference.
    """
    if num_blocks < 1:
        return [0] * 5
    n = num_blocks - 1  # max 0-based index
    if strategy == "paper_proportional":
        # Paper: layers 1, 6, 12, 18, 24 of 24 → 0-indexed fractions
        # [0/23, 5/23, 11/23, 17/23, 23/23] ≈ [0.000, 0.217, 0.478, 0.739, 1.000]
        fracs = [0.000, 0.217, 0.478, 0.739, 1.000]
        return [max(0, min(n, int(round(f * n)))) for f in fracs]
    elif strategy == "top_biased":
        fracs = [0.25, 0.50, 0.66, 0.83, 1.0]
        return [max(0, min(n, int(round(f * n)))) for f in fracs]
    else:  # "even"
        return [int(round(i * n / 4)) for i in range(5)]


def _prefix_tokens(config) -> int:
    """Number of non-patch prefix tokens: CLS + register tokens."""
    n_reg = int(getattr(config, "num_register_tokens", 0) or 0)
    return 1 + n_reg


def _get_transformer_blocks(backbone: nn.Module) -> Optional[nn.ModuleList]:
    """
    Return the transformer block list from a HF ViT-style model.
    Tries common attribute paths used by DINOv3 HF implementations.
    """
    for path in ("encoder.layer", "layer", "transformer.layers", "blocks", "layers"):
        parts = path.split(".")
        obj = backbone
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if isinstance(obj, (nn.ModuleList, nn.Sequential)):
            return obj  # type: ignore[return-value]
    return None


def _get_out_proj(block: nn.Module) -> Optional[nn.Linear]:
    """
    Return the attention output-projection Linear from a transformer block.
    Tries attribute paths typical of HF DINOv3 / ViT implementations.
    """
    for path in (
        "attention.o_proj",               # DINOv3 HF (facebook/dinov3-*)
        "attention.output.dense",         # BertAttention-style
        "attn.proj",                       # timm-style
        "attention.self_attention.out_proj",
        "attention.dense",
        "self_attn.out_proj",
        "attn.out_proj",
    ):
        parts = path.split(".")
        obj = block
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if isinstance(obj, nn.Linear):
            return obj
    return None


# ---------------------------------------------------------------------------
# Main backbone class
# ---------------------------------------------------------------------------

class LoRADINOv3(nn.Module):
    """
    DINOv3 ViT from Hugging Face with LoRA on 5 selected blocks.

    lora_mode controls where adaptation is injected:

    "out_proj"  (original v2)
        LoRA delta added to the attention out_proj output, BEFORE the block's
        MLP and residuals.  Fine-grained attention steering but the delta is
        partially absorbed by subsequent within-block processing.

    "block"  (matches DINOv2 UBE implementation)
        LoRA delta added to the FULL block output (after attention + MLP +
        residuals), mirroring `feat = x + lora(x)` in LoRADINO.  The adapted
        value is stored for the decoder; the CLEAN block output continues to
        flow through the residual stream (identical to DINOv2 behaviour).

    "both"  (default, recommended)
        Applies BOTH: out_proj LoRA (lora_layers) to steer attention, AND a
        separate block-output adapter (block_lora_layers) to align the final
        captured representation.  Gives the model two complementary handles.

    Forward expects `pixel_values` already processed by the model's
    `AutoImageProcessor` (same as training script).
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        projection_dim: int = 128,
        lora_rank: int = 16,
        freeze_backbone: bool = True,
        trust_remote_code: bool = True,
        layer_selection: str = "paper_proportional",
        lora_dropout: float = 0.0,
        lora_mode: str = "block",
        projection_mlp: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel

        if lora_mode not in ("out_proj", "block", "both"):
            raise ValueError(
                f"lora_mode must be 'out_proj', 'block', or 'both'; got {lora_mode!r}"
            )
        self.lora_mode = lora_mode

        self.backbone = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.config = self.backbone.config
        self.hidden_dim = int(self.config.hidden_size)
        self._prefix = _prefix_tokens(self.config)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- Locate transformer blocks ---
        blocks = _get_transformer_blocks(self.backbone)
        if blocks is None:
            raise RuntimeError(
                f"Could not find transformer block list in {model_id}. "
                "Please update _get_transformer_blocks() for this model."
            )
        num_blocks = len(blocks)
        self._block_indices: List[int] = _five_block_indices(num_blocks, strategy=layer_selection)

        # --- Attention out_proj LoRA layers (used in "out_proj" and "both" modes) ---
        # Only allocated when needed — avoids wasting parameters in "block" mode.
        if lora_mode in ("out_proj", "both"):
            self.lora_layers = nn.ModuleDict()
            for k in range(5):
                self.lora_layers[str(k)] = LoRALayer(
                    self.hidden_dim, self.hidden_dim, rank=lora_rank, dropout=lora_dropout
                )

        # --- Block-output adapter layers (used in "block" and "both" modes) ---
        # Mirrors LoRADINO (DINOv2): captured_feat = block_output + lora(block_output).
        # The CLEAN block_output still flows through the residual stream unchanged.
        if lora_mode in ("block", "both"):
            self.block_lora_layers = nn.ModuleDict()
            for k in range(5):
                self.block_lora_layers[str(k)] = LoRALayer(
                    self.hidden_dim, self.hidden_dim, rank=lora_rank, dropout=lora_dropout
                )

        # --- Inline hooks ---
        self._captured: List[Optional[torch.Tensor]] = [None] * 5
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._register_hooks(blocks)

        import logging
        logging.getLogger(__name__).info(
            f"LoRADINOv3: {num_blocks} blocks, strategy='{layer_selection}', "
            f"selected={self._block_indices}, lora_mode={lora_mode}, "
            f"hidden_dim={self.hidden_dim}, lora_rank={lora_rank}, lora_dropout={lora_dropout}"
        )

        # --- Per-depth projections to lower dimension C ---
        self.num_extract_layers = 5
        self.projection_dim = projection_dim
        if projection_mlp:
            # 2-layer MLP: hidden → hidden//4 → C.  Captures nonlinear
            # structure that a single linear cannot (important when the
            # compression ratio hidden_dim/projection_dim is large, e.g. 10:1
            # for ViT-H 1280→128).
            bottleneck = self.hidden_dim // 4
            self.projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim, bottleneck),
                    nn.GELU(),
                    nn.Linear(bottleneck, projection_dim),
                )
                for _ in range(5)
            ])
        else:
            self.projections = nn.ModuleList(
                [nn.Linear(self.hidden_dim, projection_dim) for _ in range(5)]
            )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self, blocks: nn.ModuleList) -> None:
        """
        Register forward hooks per selected block based on lora_mode.

        "out_proj" mode (2 hooks per block):
            Hook 1 — on attention out_proj Linear: adds lora_layers delta to
                      the attention output BEFORE MLP/residuals.
            Hook 2 — on block: captures the final hidden state (the LoRA
                      effect has already propagated through the full block).
            Falls back to a single block-output hook if out_proj not found.

        "block" mode (1 hook per block):
            Hook — on block: computes captured = block_output + block_lora(block_output),
                   stores it for the decoder, but returns the CLEAN block_output
                   so the residual stream is unaffected.  Matches DINOv2.

        "both" mode (2-3 hooks per block):
            Hook 1 — out_proj LoRA (as in "out_proj" mode).
            Hook 2 — block-output adapter (as in "block" mode), applied AFTER
                      the out_proj LoRA has propagated through the rest of the
                      block.  The combined adapted output is captured.
        """
        self._remove_hooks()
        for stack_pos, blk_idx in enumerate(self._block_indices):
            block = blocks[blk_idx]
            out_proj = _get_out_proj(block)

            if self.lora_mode == "out_proj":
                self._register_out_proj_mode(block, out_proj, stack_pos)
            elif self.lora_mode == "block":
                self._register_block_mode(block, stack_pos)
            else:  # "both"
                self._register_both_mode(block, out_proj, stack_pos)

    def _register_out_proj_mode(
        self, block: nn.Module, out_proj: Optional[nn.Linear], stack_pos: int
    ) -> None:
        """out_proj LoRA + block capture (original v2 behaviour)."""
        if out_proj is not None:
            def make_lora_hook(sp: int):
                def hook(module, input, output):  # noqa: ARG001
                    x = input[0]
                    delta = self.lora_layers[str(sp)](x)
                    return output + delta
                return hook

            self._hooks.append(
                out_proj.register_forward_hook(make_lora_hook(stack_pos))
            )

            def make_capture_hook(sp: int):
                def hook(module, input, output):  # noqa: ARG001
                    h = output[0] if isinstance(output, tuple) else output
                    self._captured[sp] = h
                return hook

            self._hooks.append(
                block.register_forward_hook(make_capture_hook(stack_pos))
            )
        else:
            # Fallback: apply lora_layers to block output directly
            self._register_block_mode_with_layers(block, stack_pos, self.lora_layers)

    def _register_block_mode(self, block: nn.Module, stack_pos: int) -> None:
        """Block-output adapter matching DINOv2: captured = h + lora(h), stream stays clean."""
        self._register_block_mode_with_layers(block, stack_pos, self.block_lora_layers)

    def _register_block_mode_with_layers(
        self, block: nn.Module, stack_pos: int, lora_dict: nn.ModuleDict
    ) -> None:
        """Helper: register a block hook that captures h + lora(h) without modifying stream."""
        def make_block_adapter_hook(sp: int, layers: nn.ModuleDict):
            def hook(module, input, output):  # noqa: ARG001
                h = output[0] if isinstance(output, tuple) else output
                delta = layers[str(sp)](h)
                self._captured[sp] = h + delta
                # Return the CLEAN output so the residual stream is unaffected
                return output
            return hook

        self._hooks.append(
            block.register_forward_hook(make_block_adapter_hook(stack_pos, lora_dict))
        )

    def _register_both_mode(
        self, block: nn.Module, out_proj: Optional[nn.Linear], stack_pos: int
    ) -> None:
        """out_proj LoRA (attention steering) + block-output adapter (representation alignment)."""
        if out_proj is not None:
            # Hook 1: out_proj LoRA steers attention
            def make_lora_hook(sp: int):
                def hook(module, input, output):  # noqa: ARG001
                    x = input[0]
                    delta = self.lora_layers[str(sp)](x)
                    return output + delta
                return hook

            self._hooks.append(
                out_proj.register_forward_hook(make_lora_hook(stack_pos))
            )

        # Hook 2: block-output adapter captures the final adapted representation.
        # After the out_proj LoRA has propagated through MLP+residuals, we apply
        # block_lora_layers on top and store the result — then return the clean output.
        def make_both_capture_hook(sp: int):
            def hook(module, input, output):  # noqa: ARG001
                h = output[0] if isinstance(output, tuple) else output
                delta = self.block_lora_layers[str(sp)](h)
                self._captured[sp] = h + delta
                return output
            return hook

        self._hooks.append(
            block.register_forward_hook(make_both_capture_hook(stack_pos))
        )

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @property
    def num_patches_from_config(self) -> int:
        ps = int(getattr(self.config, "patch_size", 16))
        isz = int(getattr(self.config, "image_size", 224))
        n = isz // ps
        return n * n

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) from AutoImageProcessor

        Returns:
            features: (B, 5, P, C) — patch tokens only, LoRA-adapted
        """
        self._captured = [None] * 5

        # Run backbone; hooks fire during this call and populate _captured.
        self.backbone(pixel_values=pixel_values)

        projected: List[torch.Tensor] = []
        for sp in range(5):
            h = self._captured[sp]
            if h is None:
                raise RuntimeError(f"LoRA hook for stack_pos={sp} did not fire.")
            patches = h[:, self._prefix:, :]          # strip CLS + registers
            projected.append(self.projections[sp](patches))

        return torch.stack(projected, dim=1)           # (B, 5, P, C)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
