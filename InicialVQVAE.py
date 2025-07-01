
import torch
import torch.nn as nn

import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.9, epsilon: float = 1e-6):
        """
        VQ-VAE codebook with Exponential Moving Average (EMA) updates.

        Args:
            num_embeddings: Number of discrete codebook vectors (M).
            embedding_dim: Dimension of each codebook vector (D).
            decay: EMA decay factor (γ).
            epsilon: Small value to avoid division by zero.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        # Codebook: shape (M, D)
        self.register_buffer("embedding", F.normalize(torch.randn(num_embeddings, embedding_dim), dim=1) * 2)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embedding_avg", self.embedding.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor of shape (B, C, D, H, W)

        Returns:
            quantized: Quantized output (B, C, D, H, W)
            loss: Quantization loss
            encoding_indices: Indices of codebook entries used (B, D, H, W)
            perplexity: float tensor
            used_codes: float tensor (percentual dos códigos utilizados)
        """
        input_shape = z.shape  # (B, C, D, H, W)
        flat_input = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)  # (N, D)

        # Encontrar índices dos embeddings mais próximos
        distances = torch.cdist(flat_input.unsqueeze(0), self.embedding.unsqueeze(0)).squeeze(0)  # (N, M)
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)  # (N, M)

        # Quantização: substituir pelas embeddings correspondentes
        quantized = encodings @ self.embedding  # (N, D)
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], input_shape[4], input_shape[1])
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # Monitoramento (sempre)
        avg_probs = encodings.mean(dim=0)  # (M,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        used_codes = (self.cluster_size > 1e-5).sum().float() / self.num_embeddings

        # Atualização EMA (somente se treinando)
        if self.training:
            with torch.no_grad():
                new_cluster_size = encodings.sum(0)  # (M,)
                self.cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

                embed_sum = encodings.T @ flat_input  # (M, D)
                self.embedding_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # Normalização com segurança numérica
                n = self.cluster_size.sum()
                cluster_size = ((self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon)) * n

                self.embedding.copy_(self.embedding_avg / (cluster_size.unsqueeze(1) + self.epsilon))

        # Estimador Straight-Through
        quantized_st = z + (quantized - z).detach()

        # Perda de quantização
        loss = F.mse_loss(quantized_st, z)

        # Índices de saída
        encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3], input_shape[4])

        return quantized_st, loss, encoding_indices, perplexity, used_codes



class InitialVQVAE(nn.Module):
    def __init__(self) -> None:
     
        super(InitialVQVAE, self).__init__()

        num_embeddings = 128
        embedding_dim = 16

        # Encoder com a PRIMEIRA camada reduzindo
        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                stride=2,        # 🔹 Reduz pela metade
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=8,
                out_channels=embedding_dim,
                kernel_size=3,
                stride=1,        # 🔹 Mantém resolução
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim)

        # Decoder recuperando resolução
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=embedding_dim,
                out_channels=8,
                kernel_size=3,
                stride=2,         # 🔹 Recupera resolução
                padding=1,
                output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                in_channels=8,
                out_channels=3,
                kernel_size=3,
                stride=1,         # 🔹 Mantém resolução
                padding=1
            ),
            nn.Sigmoid()          # Ou ReLU, conforme seu dataset
        )
    

    def getOptimizer(self,):
        """
        Retorna optimizer e scheduler
        """
        from lion_pytorch import Lion

        optimizer = Lion(
            self.parameters(),
            lr=1e-3,          # cuidado! normalmente mais alto que Adam
            weight_decay=0.001
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        return optimizer, scheduler

    def forward(self, x,epoch):
        z = self.encoder(x)
        
        if epoch > 200:
     
            quantized, vq_loss, codes,perplexity, used_codes = self.vq(z)

        else:
    
            quantized = z
            vq_loss = torch.tensor(0.0, device=x.device)
            codes = 0
            perplexity, used_codes =0,0
        out = self.decoder(quantized)
        return out ,vq_loss,codes,perplexity, used_codes 