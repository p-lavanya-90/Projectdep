"""ML model architectures — must exactly mirror training scripts."""

import torch
import torch.nn as nn


class AttentionFusionModel(nn.Module):
    def __init__(self, audio_dim=1024, image_dim=2048, text_dim=768):
        super().__init__()
        # Stable branch sizes
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.image_branch = nn.Sequential(
            nn.Linear(image_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU()
        )
        self.text_branch  = nn.Sequential(
            nn.Linear(text_dim,  256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU()
        )
        # Attention to weigh modalities
        self.attention    = nn.Sequential(
            nn.Linear(384, 128), nn.Tanh(),
            nn.Linear(128, 3), nn.Softmax(dim=1)
        )
        self.classifier   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 2)
        )

    def forward(self, audio, image, text):
        a = self.audio_branch(audio)
        i = self.image_branch(image)
        t = self.text_branch(text)
        w = self.attention(torch.cat([a, i, t], dim=1))
        # Fused representation
        fused = w[:, 0:1]*a + w[:, 1:2]*i + w[:, 2:3]*t
        return self.classifier(fused), w.detach()


class CrossModalTransformer(nn.Module):
    def __init__(self, audio_dim=1024, image_dim=2048, text_dim=768,
                 proj_dim=256, n_heads=8, n_layers=2, ff_dim=512, dropout=0.3):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim), nn.LayerNorm(proj_dim),
            nn.ReLU(), nn.Dropout(dropout*0.5)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim), nn.LayerNorm(proj_dim),
            nn.ReLU(), nn.Dropout(dropout*0.5)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim), nn.LayerNorm(proj_dim),
            nn.ReLU(), nn.Dropout(dropout*0.5)
        )
        self.modality_embed = nn.Parameter(torch.randn(1, 3, proj_dim)*0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 2)
        )

    def forward(self, audio, image, text):
        a = self.audio_proj(audio)
        i = self.image_proj(image)
        t = self.text_proj(text)
        seq = torch.stack([a, i, t], dim=1) + self.modality_embed
        with torch.no_grad():
            mha = self.transformer.layers[0].self_attn
            _, attn = mha(seq, seq, seq, need_weights=True, average_attn_weights=True)
        out = self.transformer(seq)
        return self.classifier(out.mean(dim=1)), attn.detach()
