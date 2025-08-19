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
        return self.pe[:length]  # (length, d_model)


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
        # usar reshape em vez de view
        z_flat = z.reshape(B * N, D)
        e = self.embedding.weight  # (K, D)

        dist = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            - 2 * z_flat @ e.t()
            + torch.sum(e ** 2, dim=1)
        )  # (B*N, K)

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







class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost

        # Codebook: shape (M, D)
        embed_init = F.normalize(torch.randn(num_embeddings, embedding_dim), dim=1) * 0.1
        embed_init[0] = torch.zeros_like(embed_init[0])
        embed_init[1] = torch.ones_like(embed_init[1]) * 6

        self.register_buffer("embedding", embed_init)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embedding_avg", embed_init.clone())

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor of shape (B, C, D, H, W)

        Returns:
            quantized_st: Quantized output (B, C, D, H, W)
            total_loss: Commitment loss
            encoding_indices: Indices of codebook entries used (B, D, H, W)
            perplexity: Scalar tensor
            used_codes: Fraction of used codes
        """
        B, C, D, H, W = z.shape

        # Flatten input (N, D)
        flat_input = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)

        # Distances (N, M)
        distances = (
            flat_input.pow(2).sum(1, keepdim=True)
            - 2 * flat_input @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)

        # Quantized output (N, D)
        quantized = encodings @ self.embedding
        quantized = quantized.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        # Monitoring
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        used_codes = (encodings.sum(0) > 0).float().mean()

        # EMA updates
        if self.training:
            with torch.no_grad():
                new_cluster_size = encodings.sum(0)
                self.cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

                embed_sum = encodings.T @ flat_input
                self.embedding_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                ) * n

                self.embedding.copy_(self.embedding_avg / (cluster_size.unsqueeze(1) + self.epsilon))

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # Commitment loss
        total_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)

        # Reshape encoding indices
        encoding_indices = encoding_indices.view(B, D, H, W)

        return quantized_st, total_loss, encoding_indices, perplexity, used_codes

# =========================
# Frame Encoder/Decoder (24x24)
# =========================
class FrameEncoder2D(nn.Module):
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),  # 24->12
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # 12->6
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),    # 6->3
            nn.ReLU(inplace=True),
        )
        self._h_w = 3
        self._proj = nn.Linear(128 * self._h_w * self._h_w, d_model)
        self._norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) -> (N, D)"""
        f = self._conv(x)
        f = f.flatten(1)
        f = self._proj(f)
        return self._norm(f)


class FrameDecoder2D(nn.Module):
    def __init__(self, d_model: int, out_ch: int):
        super().__init__()
        self._h_w = 3
        self._unproj = nn.Linear(d_model, 128 * self._h_w * self._h_w)
        self._deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 3->6
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 6->12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_ch, 4, stride=2, padding=1), # 12->24
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) -> (N, C, 24, 24)"""
        f = self._unproj(x).view(x.size(0), 128, self._h_w, self._h_w)
        return self._deconv(f)


# =========================
# Video Encoder Temporal (vetorizado)
# =========================
class VideoEncoderTemporalPool(nn.Module):
    def __init__(self, in_ch: int, d_model: int, nhead: int, num_layers: int, num_tokens: int):
        super().__init__()
        self._frame_enc = FrameEncoder2D(in_ch, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self._temporal_enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self._pos = SinusoidalPositionalEncoding(d_model)
        self._queries = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self._attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> z: (B, N, D)"""
        B, C, T, H, W = x.shape
        # (B, T, C, H, W)
        frames = x.permute(0, 2, 1, 3, 4)
        # Vetorizado pelo tempo
        feats = self._frame_enc(frames.reshape(B * T, C, H, W))  # (B*T, D)
        seq = feats.view(B, T, -1)  # (B, T, D)
        seq = seq + self._pos(T).unsqueeze(0).to(seq.dtype).to(seq.device)  # + PE
        seq = self._temporal_enc(seq)  # (B, T, D)

        # Pooling por queries aprendíveis
        q = self._queries.expand(B, -1, -1)  # (B, N, D)
        z, _ = self._attn(query=q, key=seq, value=seq)  # (B, N, D)
        return z


# =========================
# Video Decoder Temporal (vetorizado + tgt aprendível)
# =========================
# =========================
# Video Decoder Temporal (AUTOREGRESSIVO)
# =========================
class VideoDecoderTemporal(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, out_ch: int, max_len: int = 10000):
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
        self._frame_dec = FrameDecoder2D(d_model, out_ch)
        self._max_len = max_len


    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # Triangular superior (True bloqueia)
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


    def forward(self, z_tokens: torch.Tensor, T: int) -> torch.Tensor:
        """
        Treinamento com teacher forcing + máscara causal.
        z_tokens: (B, N, D)
        Returns: (B, C, T, 24, 24)
        """
        assert T <= self._max_len, f"T={T} excede max_len={self._max_len}"
        B, N, D = z_tokens.shape
        pos = torch.arange(T, device=z_tokens.device)
        tgt = self._tgt_embed(pos).unsqueeze(0).expand(B, T, D)
        tgt = tgt + self._pos_t(T).unsqueeze(0).to(z_tokens)
        tgt_mask = self._causal_mask(T, z_tokens.device)
        dec = self._decoder(tgt=tgt, memory=z_tokens, tgt_mask=tgt_mask) # (B, T, D)
        dec = self._to_frame_feat(dec)
        frames_bt = self._frame_dec(dec.reshape(B * T, D))
        C, H, W = frames_bt.shape[1:]
        frames = frames_bt.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return frames


    @torch.no_grad()
    def generate(self, z_tokens: torch.Tensor, T: int) -> torch.Tensor:
        """Inferência autoregressiva (sem teacher forcing), gera 1 frame por passo."""
        assert T <= self._max_len
        B, N, D = z_tokens.shape
        outputs = []
        for t in range(T):
            pos = torch.arange(t + 1, device=z_tokens.device)
            tgt = self._tgt_embed(pos).unsqueeze(0).expand(B, t + 1, D)
            tgt = tgt + self._pos_t(t + 1).unsqueeze(0).to(z_tokens)
            tgt_mask = self._causal_mask(t + 1, z_tokens.device)
            dec = self._decoder(tgt=tgt, memory=z_tokens, tgt_mask=tgt_mask)
            dec = self._to_frame_feat(dec)
            last_feat = dec[:, -1, :] # (B, D)
            frame = self._frame_dec(last_feat) # (B, C, H, W)
            outputs.append(frame.unsqueeze(2))
        return torch.cat(outputs, dim=2) # (B, C, T, H, W)


# =========================
# Modelo completo
# =========================
class VideoVQVAETransformer(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        enc_layers: int = 2,
        dec_layers: int = 2,
        num_tokens: int = 20,
        codebook_size: int = 512,
        beta: float = 0.25,
        max_len: int = 10000,
    ) -> None:
        super().__init__()
        self._enc = VideoEncoderTemporalPool(in_ch, d_model, nhead, enc_layers, num_tokens)
        self._vq = VectorQuantizer(codebook_size, d_model,0.25)
        self._dec = VideoDecoderTemporal(d_model, nhead, dec_layers, out_ch, max_len=max_len)

    def getFeature(self,x):
        self.eval()
        z_tokens = self._enc(x)  # (B, N, D)
        z_q, indices, vq_loss = self._vq(z_tokens)  # (B, N, D), (B, N)
        return indices

    def getOptimizer(self, steps_per_epoch: int, epochs: int):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR

        optimizer = AdamW(
            self.parameters(),
            lr=1e-4,
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

    def forward(self, x: torch.Tensor,i) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, C, T, 24, 24) -> recon, indices, vq_loss"""
        z_tokens = self._enc(x)  # (B, N, D)
        z_q, indices, vq_loss = self._vq(z_tokens)  # (B, N, D), (B, N)
        recon = self._dec(z_q, x.shape[2])  # (B, C, T, 24, 24)
        return recon,vq_loss, indices,0,0


if __name__ == "__main__":
    # Teste rápido de forma e passagem
    B, C, T, H, W = 2, 3, 128, 24, 24
    video_input = torch.rand(B, C, T, H, W)
    model = VideoVQVAETransformer()
    recon_video, indices, vq_loss = model(video_input)
    print("Input shape:", video_input.shape)
    print("Recon shape:", recon_video.shape)
    print("VQ indices shape:", indices.shape)
    print("VQ loss:", float(vq_loss.detach()))