

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Positional Encoding
# =========================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:length]


# =========================
# Vector Quantizer
# =========================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        bound = 1.0 / math.sqrt(embedding_dim)
        nn.init.uniform_(self.embedding.weight, -bound, bound)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = z.shape
        z_flat = z.reshape(B * N, D)
        e = self.embedding.weight  # (K, D)

        dist = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            - 2 * z_flat @ e.t()
            + torch.sum(e ** 2, dim=1)
        )

        indices = torch.argmin(dist, dim=1)  # (B*N)
        z_q = self.embedding(indices).reshape(B, N, D)

        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = self.beta * F.mse_loss(z_q.detach(), z)
        vq_loss = codebook_loss + commit_loss

        z_q = z + (z_q - z).detach()
        return z_q, indices.reshape(B, N), vq_loss
    def printCodeBook(self):
        return
        print("\nCodeBook")
        cont=0
        for i in range(self.num_embeddings):
            if sum(self.embedding.weight[i])>0.001 or True:
                print(i, self.embedding.weight[i])
            else:
                print(i,0)
                cont+=1
        print("0s",cont)
        print()

# =========================
# Video Encoder Temporal (sem convoluções)
# =========================
class VideoEncoderNoSpatial(nn.Module):
    def __init__(self, in_ch: int, h: int, w: int, d_model: int, nhead: int, num_layers: int, num_tokens: int):
        super().__init__()
        self._proj = nn.Linear(in_ch * h * w, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self._temporal_enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self._pos = SinusoidalPositionalEncoding(d_model)
        self._queries = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self._attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> z: (B, N, D)"""
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B, T, C * H * W)  # (B, T, C*H*W)
        feats = self._proj(frames)                                  # (B, T, D)
        feats = feats + self._pos(T).unsqueeze(0).to(feats.device)  # + PE
        seq = self._temporal_enc(feats)                             # (B, T, D)

        # Pooling por queries
        q = self._queries.expand(B, -1, -1)  # (B, N, D)
        z, _ = self._attn(query=q, key=seq, value=seq)  # (B, N, D)
        return z


# =========================
# Video Decoder Temporal
# =========================
class VideoDecoderNoSpatial(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, out_ch: int, h: int, w: int, max_len: int = 10000):
        super().__init__()
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self._decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self._pos_t = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self._tgt_embed = nn.Embedding(max_len, d_model)
        self._to_frame_feat = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self._out = nn.Sequential(
            nn.Linear(d_model, out_ch * h * w),
            nn.Sigmoid()
        )
        self._out_ch, self._h, self._w = out_ch, h, w
        self._max_len = max_len

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, z_tokens: torch.Tensor, T: int) -> torch.Tensor:
        """z_tokens: (B, N, D) -> frames: (B, C, T, H, W)"""
        B, N, D = z_tokens.shape
        pos = torch.arange(T, device=z_tokens.device)
        tgt = self._tgt_embed(pos).unsqueeze(0).expand(B, T, D)
        tgt = tgt + self._pos_t(T).unsqueeze(0).to(z_tokens)
        tgt_mask = self._causal_mask(T, z_tokens.device)

        dec = self._decoder(tgt=tgt, memory=z_tokens, tgt_mask=tgt_mask)
        dec = self._to_frame_feat(dec)

        frames = self._out(dec).reshape(B, T, self._out_ch, self._h, self._w)
        frames = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return frames


# =========================
# Modelo completo
# =========================
class ST(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        h: int = 24,
        w: int = 24,
        d_model: int = 256,
        nhead: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        num_tokens: int = 128,
        codebook_size: int = 512,
        beta: float = 0.25,
        max_len: int = 10000,
    ) -> None:
        super().__init__()
        self._enc = VideoEncoderNoSpatial(in_ch, h, w, d_model, nhead, enc_layers, num_tokens)
        self._vq = VectorQuantizer(codebook_size, d_model, beta)
        self._dec = VideoDecoderNoSpatial(d_model, nhead, dec_layers, out_ch, h, w, max_len=max_len)

    def getFeature(self,x):
        self.eval()
        z_tokens = self._enc(x)  # (B, N, D)
        z_q, indices, vq_loss = self._vq(z_tokens)  # (B, N, D), (B, N)
        return indices
    '''
    def getOptimizer(self, steps_per_epoch: int=700, epochs: int=500):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR

        optimizer = AdamW(
            self.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=3e-3,   # pico do ciclo
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.1,  # 10% do tempo em warmup
            anneal_strategy="cos"
        )

        return optimizer, scheduler
    '''
    def getOptimizer(self):
        from torch.optim import Adam

        optimizer = Adam(
            self.parameters(),
            lr=1e-3,          # taxa bem maior para overfitting rápido
            betas=(0.9, 0.999),
            weight_decay=0    # SEM regularização
        )

        scheduler = None  # nada de ciclo, queremos overfit rápido
        return optimizer, scheduler
    def comparaEncoderQuant(self,x):
        return
        self.eval()
        
        z = self._enc(x)

        flat_input = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)  # (N, D)
        quantized,codes, vq_loss , perplexity, used_codes = self._vq(z)
        flat_quantized = quantized.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)  # (N, D)
        n = flat_quantized.shape[0]
        print("analise ",n)
        for i in range(n):
                if sum(flat_input[i])>16*3:
                    print(i,flat_quantized[i])
                    print(i,flat_input[i])
                    print()

        s= set(codes[0])
        print("fim",len(s))

    


    def forward(self, x: torch.Tensor,i=11) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, C, T, H, W) -> recon, indices, vq_loss"""
        z_tokens = self._enc(x)                        # (B, N, D)
        z_q, indices, vq_loss = self._vq(z_tokens)     # (B, N, D), (B, N)
        recon = self._dec(z_q, x.shape[2])             # (B, C, T, H, W)
        return recon, vq_loss, indices,0,0


if __name__ == "__main__":
    # Teste rápido de forma e passagem
    B, C, T, H, W = 2, 3, 128, 24, 24
    video_input = torch.rand(B, C, T, H, W)
    model = ST()
    recon_video, indices, vq_loss = model(video_input)
    print("Input shape:", video_input.shape)
    print("Recon shape:", recon_video.shape)
    print("VQ indices shape:", indices.shape)
    print("VQ loss:", float(vq_loss.detach()))