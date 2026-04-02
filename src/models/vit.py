from typing import Optional, Tuple

import torch
from torch import nn


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block that can return attention weights."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        norm_x = self.norm1(x)
        attn_output, attn_weights = self.attention(
            norm_x,
            norm_x,
            norm_x,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class ViTLite(nn.Module):
    """Lightweight Vision Transformer for CIFAR-10."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        patch_dim = in_channels * patch_size * patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward_features(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self._patchify(x)
        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)

        last_attention = None
        for block in self.encoder_blocks:
            x, block_attention = block(x, return_attention=return_attention)
            if return_attention and block_attention is not None:
                last_attention = block_attention

        x = self.norm(x)
        return x, last_attention

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        features, attention = self.forward_features(
            x, return_attention=return_attention
        )
        logits = self.head(features[:, 0])

        if return_attention:
            return logits, attention
        return logits

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Returns attention from the class token to the image patches."""
        _, attention = self.forward(x, return_attention=True)
        if attention is None:
            raise RuntimeError("Attention weights were not produced.")

        cls_attention = attention[:, :, 0, 1:]
        num_patches_per_side = self.image_size // self.patch_size
        return cls_attention.mean(dim=1).reshape(
            -1, num_patches_per_side, num_patches_per_side
        )
