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
        self._pos = SinusoidalPositionalEncoding(d_model)
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

        self._pos_t = SinusoidalPositionalEncoding(d_model, max_len=max_len)

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
        start_frame: (B, C, H, W) or (B, C, T_init, H, W)

        Returns: frames (B, C, T, H, W)
        """
        B, N, D = z_tokens.shape
        device = z_tokens.device
        frame_dim = self._out_ch * self._h * self._w
        z_mem = z_tokens  # memory for TransformerDecoder

        # -------- teacher forcing path --------
        if teacher_forcing_frames is not None:
            assert teacher_forcing_frames.shape[2] >= T, "teacher_forcing_frames must have at least T frames"
            prev_frames = torch.zeros(B, T, self._out_ch, self._h, self._w,
                                    device=device, dtype=teacher_forcing_frames.dtype)

            # primeira frame: start_frame ou primeiro GT
            if start_frame is None:
                prev_frames[:, 0] = teacher_forcing_frames[:, :, 0]
            else:
                prev_frames[:, 0] = start_frame[:, :, 0] if start_frame.ndim == 5 else start_frame

            if T > 1:
                # teacher_forcing_frames[:, :, :T-1] -> prev_frames[:, 1:]
                prev_frames[:, 1:] = teacher_forcing_frames.permute(0, 2, 1, 3, 4)[:, :T-1]

            prev_flat = prev_frames.reshape(B, T, frame_dim)
            tgt_emb = self._frame_enc(prev_flat)
            tgt_emb = tgt_emb + self._pos_t(T).unsqueeze(0).to(dtype=tgt_emb.dtype, device=device)

            tgt_mask = self._causal_mask(T, device)
            dec = self._decoder(tgt=tgt_emb, memory=z_mem, tgt_mask=tgt_mask)
            dec = self._to_frame_feat(dec)
            out = self._out(dec)  # (B, T, frame_dim)
            frames = out.reshape(B, T, self._out_ch, self._h, self._w).permute(0, 2, 1, 3, 4)
            return frames

        # -------- generation path (no teacher forcing) --------
        assert start_frame is not None, "You must provide start_frame if no teacher forcing"

        # se start_frame tem múltiplos frames
        if start_frame.ndim == 5:
            T_init = start_frame.shape[2]
            start_frames_list = [start_frame[:, :, i, :, :] for i in range(T_init)]
        else:
            start_frames_list = [start_frame]

        generated: list[torch.Tensor] = start_frames_list.copy()

        for t in range(len(start_frames_list), T):
            # stack previous frames
            seq_frames = torch.stack(generated, dim=1)  # (B, L, C, H, W)
            L = seq_frames.shape[1]
            seq_flat = seq_frames.reshape(B, L, frame_dim)
            tgt_emb = self._frame_enc(seq_flat)
            tgt_emb = tgt_emb + self._pos_t(L).unsqueeze(0).to(dtype=tgt_emb.dtype, device=device)

            tgt_mask = self._causal_mask(L, device)
            dec = self._decoder(tgt=tgt_emb, memory=z_mem, tgt_mask=tgt_mask)
            dec = self._to_frame_feat(dec)

            last_out = dec[:, -1, :]
            pred_flat = self._out(last_out)        # (B, frame_dim)
            pred_frame = pred_flat.reshape(B, self._out_ch, self._h, self._w)
            generated.append(pred_frame)

        # concatenar todos os frames (iniciais + gerados)
        full_seq = torch.stack(generated, dim=1)  # (B, L_total, C, H, W)
        frames = full_seq.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return frames

# =========================
# ST full model (encoder -> vq -> autoregressive decoder)
# =========================
class STFirst(nn.Module):
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

        self._pos_t = SinusoidalPositionalEncoding(d_model, max_len=max_len)

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
        self._vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=d_model, beta=beta)

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
        teacher_forcing = False
      
        # decoder
        if teacher_forcing :
            recon = self._decoder(z_q, T, start_frame=start_frame, teacher_forcing_frames=x)
        else:
            #assert start_frame is not None, "start_frame required for autoregressive generation"
            start_frame = x[:, :, :1, :, :]  # (B, C, 1, H, W)
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
