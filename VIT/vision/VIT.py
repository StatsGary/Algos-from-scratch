# Description: Vision Transformer (ViT) for image classification.

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def to_pair(value):
    return value if isinstance(value, tuple) else (value, value)

# -------------------------------
# Patch Embedding Module
# -------------------------------

class PatchEmbedding(nn.Module):
    """
    Converts input images into a sequence of flattened patch embeddings.
    """
    def __init__(self, *, image_size, patch_size, input_channels, embedding_dim):
        super().__init__()
        image_height, image_width = to_pair(image_size)
        patch_height, patch_width = to_pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image dimensions must be divisible by patch size."

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = input_channels * patch_height * patch_width
        self.patch_height = patch_height
        self.patch_width = patch_width

        # Rearrange pattern fixed: parameters passed here, no need to compute dynamically!
        self.rearrange = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=patch_height,
            p2=patch_width
        )
        self.linear_projection = nn.Linear(self.patch_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        patches = self.rearrange(x)                       # (B, num_patches, patch_dim)
        patch_embeddings = self.linear_projection(patches)
        normalized_embeddings = self.norm(patch_embeddings)
        return normalized_embeddings

# -------------------------------
# Feedforward Network
# -------------------------------

class FeedForwardNetwork(nn.Module):
    """
    Simple Feedforward network with LayerNorm and GELU.
    """
    def __init__(self, embedding_dim, hidden_dim, dropout_prob=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Multi-Head Self Attention
# -------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention with LayerNorm and optional dropout.
    """
    def __init__(self, embedding_dim, num_heads=8, head_dim=64, dropout_prob=0.):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(embedding_dim)
        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.attend = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(inner_dim, embedding_dim)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        queries, keys, values = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            qkv
        )

        attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        attn_probs = self.attend(attn_scores)
        attn_probs = self.attention_dropout(attn_probs)

        attention_output = torch.matmul(attn_probs, values)
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        return self.output_proj(attention_output)

# -------------------------------
# Transformer Encoder Block
# -------------------------------

class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multiple stacked layers of Attention + FeedForward.
    """
    def __init__(self, embedding_dim, depth, num_heads, head_dim, feedforward_dim, dropout_prob=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadSelfAttention(embedding_dim, num_heads, head_dim, dropout_prob),
                FeedForwardNetwork(embedding_dim, feedforward_dim, dropout_prob)
            ]) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        for attention, feedforward in self.layers:
            x = attention(x) + x
            x = feedforward(x) + x
        return self.final_norm(x)

# -------------------------------
# Vision Transformer (ViT)
# -------------------------------

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.
    """
    def __init__(self, *, image_size, patch_size, num_classes, embedding_dim, depth, num_heads,
                 feedforward_dim, pool='cls', input_channels=3, head_dim=64, dropout_prob=0., embedding_dropout=0.):
        super().__init__()

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            input_channels=input_channels,
            embedding_dim=embedding_dim
        )
        num_patches = self.patch_embedding.num_patches

        # Class token & positional embedding
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # Transformer Encoder
        self.transformer = TransformerEncoder(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            dropout_prob=dropout_prob
        )

        # Pooling strategy
        assert pool in {'cls', 'mean'}, "Pool type must be either 'cls' or 'mean'"
        self.pool = pool
        self.to_latent = nn.Identity()

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)

        class_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=batch_size)
        token_sequence = torch.cat((class_tokens, patch_embeddings), dim=1)
        token_sequence += self.positional_embedding[:, :token_sequence.size(1)]
        token_sequence = self.dropout(token_sequence)

        encoded_tokens = self.transformer(token_sequence)

        pooled_output = encoded_tokens[:, 0] if self.pool == 'cls' else encoded_tokens.mean(dim=1)
        latent_output = self.to_latent(pooled_output)
        return self.classifier(latent_output)
