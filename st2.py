"""
ST VQ-VAE Transformer (no spatial convs) - full implementation
- SinusoidalPositionalEncoding
- VectorQuantizer (returns perplexity + used_codes)
- VideoEncoderNoSpatial (frames flattened -> linear -> Transformer)
- VideoDecoderNoSpatial (autoregressive, uses only previous frame + z_q, NO CONVS)
- ST wrapper that ties everything together and returns (recon, indices, vq_loss, perplexity, used_codes)

Formato de frames: (B, C, T, H, W)

Autor: adaptado para o seu fluxo.
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Positional Encoding
# =========================
import math
import torch
import torch.nn as nn


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        """
        EMA Vector Quantizer (sem gradiente direto nas embeddings).
        Args:
            num_embeddings: tamanho do codebook (K)
            embedding_dim: dimensão de cada vetor no codebook (D)
            beta: peso do commitment loss
            decay: fator de decaimento para EMA
            eps: estabilidade numérica
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.decay = decay
        self.eps = eps

        # codebook inicializado uniformemente
        embed = torch.randn(num_embeddings, embedding_dim)
        bound = 1 / math.sqrt(embedding_dim)
        embed = torch.empty(num_embeddings, embedding_dim).uniform_(-bound, bound)

        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, N, D) — tokens latentes
        Returns:
            z_q: quantized vectors (B, N, D)
            indices: (B, N) índices no codebook
            vq_loss: scalar
            perplexity: scalar
            used_codes: scalar
        """
        B, N, D = z.shape
        assert D == self.embedding_dim

        flat_z = z.reshape(B * N, D)  # (B*N, D)

        # distâncias
        dist = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )  # (B*N, K)

        indices = torch.argmin(dist, dim=1)  # (B*N,)
        z_q = self.embedding[indices].view(B, N, D)

        # straight-through trick
        z_q = z + (z_q - z).detach()

        # commitment loss (apenas z recebe grad)
        commit_loss = self.beta * F.mse_loss(z_q.detach(), z)
        vq_loss = commit_loss

        # EMA update — sem gradiente
        if self.training:
            encodings = F.one_hot(indices, self.num_embeddings).type(flat_z.dtype)
            cluster_size = encodings.sum(0)

            # EMA cluster size
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            # EMA embed avg
            embed_sum = encodings.t() @ flat_z
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # normaliza
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            )

            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

        # métricas
        with torch.no_grad():
            avg_probs = (
                F.one_hot(indices, self.num_embeddings).float().mean(0)
            )  # (K,)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            used_codes = (avg_probs > 0).float().mean()

        return z_q, indices.view(B, N), vq_loss, perplexity, used_codes



class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, h: int, w: int, max_len: int = 10000):
        super().__init__()
        assert d_model % 4 == 0, "d_model precisa ser divisível por 4 para 2D PE"

        self.h, self.w = h, w
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(d_model, h, w)

        d_model_quarter = d_model // 4

        # --- Y encoding (varia no eixo H) ---
        div_term_y = torch.exp(
            torch.arange(0, d_model_quarter, 2).float() * (-math.log(10000.0) / d_model_quarter)
        )  # (D/8,)
        pos_y = torch.arange(0, h, dtype=torch.float32).unsqueeze(1)  # (H,1)

        y_sin = torch.sin(pos_y * div_term_y).transpose(0, 1).unsqueeze(2).expand(-1, h, w)
        y_cos = torch.cos(pos_y * div_term_y).transpose(0, 1).unsqueeze(2).expand(-1, h, w)

        pe[0:d_model_quarter:2, :, :] = y_sin
        pe[1:d_model_quarter:2, :, :] = y_cos

        # --- X encoding (varia no eixo W) ---
        div_term_x = torch.exp(
            torch.arange(0, d_model_quarter, 2).float() * (-math.log(10000.0) / d_model_quarter)
        )  # (D/8,)
        pos_x = torch.arange(0, w, dtype=torch.float32).unsqueeze(1)  # (W,1)

        x_sin = torch.sin(pos_x * div_term_x).transpose(0, 1).unsqueeze(1).expand(-1, h, w)
        x_cos = torch.cos(pos_x * div_term_x).transpose(0, 1).unsqueeze(1).expand(-1, h, w)

        pe[d_model_quarter:2*d_model_quarter:2, :, :] = x_sin
        pe[d_model_quarter+1:2*d_model_quarter:2, :, :] = x_cos

        # --- Reduzimos espacialmente para 1D ---
        pe = pe.mean(dim=(1, 2))  # (D,)

        self.register_buffer("pe", pe)

    def forward(self, T: int) -> torch.Tensor:
        """
        Retorna codificação temporal + espacial resumida
        shape: (T, D)
        """
        pe_time = torch.zeros(T, self.d_model, device=self.pe.device)
        pos = torch.arange(0, T, dtype=torch.float32, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=self.pe.device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe_time[:, 0::2] = torch.sin(pos * div_term)
        pe_time[:, 1::2] = torch.cos(pos * div_term)

        return pe_time + self.pe.unsqueeze(0)

# =========================
# Vector Quantizer (non-EMA)
# returns also perplexity and used_codes for logging
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

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, N, D)
        Returns:
            z_q: quantized vectors (B, N, D)
            indices: (B, N) long
            vq_loss: scalar tensor
            perplexity: scalar tensor
            used_codes: scalar tensor (fraction of codes used)
        """
        B, N, D = z.shape
        z_flat = z.reshape(B * N, D)  # (B*N, D)
        e = self.embedding.weight    # (K, D)

        # distances (B*N, K)
        dist = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            - 2 * z_flat @ e.t()
            + torch.sum(e ** 2, dim=1)
        )

        indices = torch.argmin(dist, dim=1)  # (B*N,)
        z_q = self.embedding(indices).reshape(B, N, D)

        # losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = self.beta * F.mse_loss(z_q.detach(), z)
        vq_loss = codebook_loss + commit_loss

        # straight-through
        z_q = z + (z_q - z).detach()

        # monitoring: perplexity + used_codes
        with torch.no_grad():
            encodings = F.one_hot(indices, self.num_embeddings).type(z_flat.dtype)  # (B*N, K)
            avg_probs = encodings.mean(0)  # (K,)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            used_codes = (encodings.sum(0) > 0).float().mean()

        indices = indices.view(B, N)
        return z_q, indices, vq_loss, perplexity, used_codes

    def printCodeBook(self) -> None:
        # useful helper for debugging
        return
        for i in range(self.num_embeddings):
            print(i, self.embedding.weight[i])


# =========================
# Video Encoder Temporal (sem convoluções)
# =========================
class VideoEncoderNoSpatial(nn.Module):
    def __init__(self, in_ch: int, h: int, w: int, d_model: int, nhead: int, num_layers: int, num_tokens: int):
        super().__init__()
        self._proj = nn.Linear(in_ch * h * w, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self._temporal_enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self._pos = SinusoidalPositionalEncoding2D(d_model, h=h, w=w, max_len=10000)
        self._queries = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self._attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> z: (B, N, D)"""
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B, T, C * H * W)  # (B, T, C*H*W)
        feats = self._proj(frames)                                  # (B, T, D)
        feats = feats + self._pos(T).unsqueeze(0).to(feats.device)
        seq = self._temporal_enc(feats)                             # (B, T, D)

        # Pooling por queries
        q = self._queries.expand(B, -1, -1)  # (B, N, D)
        z, _ = self._attn(query=q, key=seq, value=seq)  # (B, N, D)
        return z


# =========================
# Video Decoder Temporal (autoregressivo, sem convs)
# =========================
class VideoDecoderNoSpatial(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 out_ch: int, h: int, w: int, max_len: int = 10000):
        super().__init__()
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self._decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self._pos_t = SinusoidalPositionalEncoding2D(d_model, h=h, w=w, max_len=max_len)

        # embedding de frames (flatten -> linear -> d_model)
        self._frame_enc = nn.Sequential(
            nn.Linear(out_ch * h * w, d_model),
            nn.LayerNorm(d_model),
        )

        # saída (d_model -> frame flatten)
        self._to_frame_feat = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self._out = nn.Linear(d_model, out_ch * h * w)

        self._out_ch, self._h, self._w = out_ch, h, w
        self._max_len = max_len

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # True blocks (upper triangular)
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        z_tokens: torch.Tensor,
        T: int,
        start_frame: torch.Tensor | None = None,
        teacher_forcing_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z_tokens: (B, N, D)
        teacher_forcing_frames: (B, C, T, H, W)
        start_frame: (B, C, H, W)

        Returns: frames (B, C, T, H, W)
        """
        B, N, D = z_tokens.shape
        device = z_tokens.device
        frame_dim = self._out_ch * self._h * self._w

        # memory for TransformerDecoder is z_tokens (B, N, D)
        z_mem = z_tokens

        # -------- teacher forcing path --------
        if teacher_forcing_frames is not None:
            assert teacher_forcing_frames.shape[2] >= T, "teacher_forcing_frames must have at least T frames"
            # teacher_forcing_frames is (B, C, T, H, W) - keep as is
            # construct prev_frames with temporal dim in axis=1: (B, T, C, H, W)
            prev_frames = torch.zeros(B, T, self._out_ch, self._h, self._w, device=device, dtype=teacher_forcing_frames.dtype)
            # first previous is start_frame if provided else first GT
            if start_frame is None:
                prev_frames[:, 0] = teacher_forcing_frames[:, :, 0]
            else:
                prev_frames[:, 0] = start_frame
            if T > 1:
                # teacher_forcing_frames[:, :, :T-1] is (B, C, T-1, H, W)
                # we want prev_frames[:, 1:] = teacher_forcing_frames[:, :, :T-1]  but need to permute
                prev_frames[:, 1:] = teacher_forcing_frames.permute(0, 2, 1, 3, 4)[:, :T-1]

            prev_flat = prev_frames.view(B, T, frame_dim)  # (B, T, frame_dim)
            tgt_emb = self._frame_enc(prev_flat)           # (B, T, D)
            tgt_emb = tgt_emb + self._pos_t(T).unsqueeze(0).to(dtype=tgt_emb.dtype, device=device)

            tgt_mask = self._causal_mask(T, device)
            dec = self._decoder(tgt=tgt_emb, memory=z_mem, tgt_mask=tgt_mask)  # (B, T, D)
            dec = self._to_frame_feat(dec)
            out = self._out(dec)  # (B, T, frame_dim)
            frames = out.view(B, T, self._out_ch, self._h, self._w).permute(0, 2, 1, 3, 4)
            return frames

        # -------- generation path (no teacher forcing) --------
        assert start_frame is not None, "When not using teacher_forcing_frames you must provide start_frame"
        generated: list[torch.Tensor] = []
        # autoregressive loop; we build sequence of previous frames (start + generated)
        for t in range(T):
            if t == 0:
                seq_frames = start_frame.unsqueeze(1)  # (B,1,C,H,W)
            else:
                # stack generated (list of (B,C,H,W)) -> (B, t, C, H, W)
                seq_frames = torch.cat([start_frame.unsqueeze(1), torch.stack(generated, dim=1)], dim=1)  # (B,L,C,H,W)

            L = seq_frames.shape[1]
            seq_flat = seq_frames.view(B, L, frame_dim)  # (B,L,frame_dim)
            tgt_emb = self._frame_enc(seq_flat)          # (B,L,D)
            tgt_emb = tgt_emb + self._pos_t(L).unsqueeze(0).to(dtype=tgt_emb.dtype, device=device)

            tgt_mask = self._causal_mask(L, device)
            dec = self._decoder(tgt=tgt_emb, memory=z_mem, tgt_mask=tgt_mask)  # (B,L,D)
            dec = self._to_frame_feat(dec)

            last_out = dec[:, -1, :]  # (B, D)
            pred_flat = self._out(last_out)  # (B, frame_dim)
            pred_frame = pred_flat.view(B, self._out_ch, self._h, self._w)  # (B, C, H, W)
            generated.append(pred_frame)

        gen_seq = torch.stack(generated, dim=1)  # (B, T, C, H, W)
        frames = gen_seq.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return frames


# =========================
# ST full model (encoder -> vq -> autoregressive decoder)
# =========================
class ST2(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        d_model: int = 32,
        nhead: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        num_tokens: int = 32,
        codebook_size: int = 64,
        beta: float = 0.25,
        max_len: int = 10000,
        h: int = 24,
        w: int = 24,
    ) -> None:
        super().__init__()

        # Encoder temporal (sem convolução)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self._encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        self._pos_t = SinusoidalPositionalEncoding2D(d_model, h=h, w=w, max_len=max_len)

        # projeta vídeos para espaço latente (C,H,W -> D)
        self._video_proj = nn.Linear(in_ch * h * w, d_model)

        self._num_tokens = num_tokens
        self._h, self._w = h, w
        self._d_model = d_model

        # Tokens aprendíveis (queries para agregar temporalmente)
        self._tokens = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

        # Atenção para fazer pooling dos tokens sobre a sequência temporal
        self._attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # Vector Quantizer (aplica-se sobre tokens agregados (B, num_tokens, d_model))
        #self._vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=d_model, beta=beta)
        self._vq = VectorQuantizerEMA(
            num_embeddings=codebook_size,
            embedding_dim=d_model,
            beta=beta,
            decay=0.99,   # pode ajustar (0.99–0.9999)
        )


        # Decoder temporal (autoregressivo) que você já tem (sem convs)
        self._decoder = VideoDecoderNoSpatial(
            d_model=d_model, nhead=nhead, num_layers=dec_layers,
            out_ch=out_ch, h=h, w=w, max_len=max_len
        )

   

    def forward(self, x: torch.Tensor, start_frame: torch.Tensor | None = None,
                teacher_forcing: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T, H, W)
            start_frame: (B, C, H, W) used for generation when teacher_forcing=False
            teacher_forcing: if True, uses GT frames for previous-frame inputs

        Returns:
            recon: (B, C, T, H, W)
            indices: (B, num_tokens)
            vq_loss: scalar tensor
            perplexity: scalar tensor
            used_codes: scalar tensor
        """
        B, C, T, H, W = x.shape
        device = x.device

        # encoder: project frames -> (B, T, D)
        vid_seq = x.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
        vid_seq = self._video_proj(vid_seq)
        vid_seq = vid_seq + self._pos_t(T).unsqueeze(0).to(device)
        memory_seq = self._encoder(vid_seq)  # (B, T, D)

        # pooling via learned tokens
        q = self._tokens.expand(B, -1, -1)  # (B, num_tokens, D)
        z_tokens, _ = self._attn(query=q, key=memory_seq, value=memory_seq)  # (B, num_tokens, D)

        # vector quantization
        z_q, indices, vq_loss, perplexity, used_codes = self._vq(z_tokens)

        # decoder
        if teacher_forcing:
            recon = self._decoder(z_q, T, start_frame=start_frame, teacher_forcing_frames=x)
        else:
            assert start_frame is not None, "start_frame required for autoregressive generation"
            recon = self._decoder(z_q, T, start_frame=start_frame, teacher_forcing_frames=None)

        return recon, vq_loss,indices, perplexity, used_codes

    def getFeature(self,x):
        self.eval()
        B, C, T, H, W = x.shape
        device = x.device
        # encoder: project frames -> (B, T, D)
        vid_seq = x.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
        vid_seq = self._video_proj(vid_seq)
        vid_seq = vid_seq + self._pos_t(T).unsqueeze(0).to(device)
        memory_seq = self._encoder(vid_seq)  # (B, T, D)

        # pooling via learned tokens
        q = self._tokens.expand(B, -1, -1)  # (B, num_tokens, D)
        z_tokens, _ = self._attn(query=q, key=memory_seq, value=memory_seq)  # (B, num_tokens, D)

        # vector quantization
        z_q, indices, vq_loss, perplexity, used_codes = self._vq(z_tokens)
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
# quick smoke test when run as script
if __name__ == "__main__":
    B, C, T, H, W = 2, 3, 128, 24, 24
    model = ST2(d_model=64, nhead=4, enc_layers=1, dec_layers=1, num_tokens=128, codebook_size=64, h=H, w=W)
    vid = torch.rand(B, C, T, H, W)
    recon, indices, vq_loss, perplexity, used = model(vid, start_frame=vid[:, :, 0], teacher_forcing=True)
    print("recon", recon.shape)
    print("indices", indices.shape)
    print("vq_loss", float(vq_loss.detach()))
    print("perplexity", float(perplexity.detach()))
    print("used_codes", float(used.detach()))
