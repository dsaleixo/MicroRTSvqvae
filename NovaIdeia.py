import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from lion_pytorch import Lion


'''
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99, epsilon: float = 1e-6):
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
        self.register_buffer("embedding", F.normalize(torch.randn(num_embeddings, embedding_dim), dim=1) * 1.0)
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
'''
class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        commitment_cost: float = 0.25,
    ):
        """
        VQ-VAE codebook with Exponential Moving Average (EMA) updates.

        Args:
            num_embeddings: Number of discrete codebook vectors (M).
            embedding_dim: Dimension of each codebook vector (D).
            decay: EMA decay factor (γ).
            epsilon: Small value to avoid division by zero.
            commitment_cost: Weight for the commitment loss term (β).
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost

        # Codebook: shape (M, D), initialized normalized
        self.register_buffer(
            "embedding", F.normalize(torch.randn(num_embeddings, embedding_dim), dim=1)*0.1
        )
        self.embedding[0] = torch.zeros_like(self.embedding[0])
        self.embedding[1] = torch.ones_like(self.embedding[1])*6
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embedding_avg", self.embedding.clone())

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor of shape (B, C, D, H, W)

        Returns:
            quantized: Quantized output (B, C, D, H, W)
            loss: Total quantization loss
            encoding_indices: Indices of codebook entries used (B, D, H, W)
            perplexity: float tensor
            used_codes: float tensor (percentual dos códigos utilizados)
        """
        input_shape = z.shape  # (B, C, D, H, W)
        # Flatten to (N, D)
        self.embedding[0] = torch.zeros_like(self.embedding[0])
        self.embedding[1] = torch.ones_like(self.embedding[1])*6
        flat_input = (
            z.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)
        )

        # Compute distances (N, M)
        distances = torch.cdist(
            flat_input.unsqueeze(0), self.embedding.unsqueeze(0)
        ).squeeze(0)

        # Nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)

        # Quantized output (N, D)
        quantized = encodings @ self.embedding
        quantized_reshaped = quantized.view(
            input_shape[0], input_shape[2], input_shape[3], input_shape[4], input_shape[1]
        )
        quantized_reshaped = quantized_reshaped.permute(
            0, 4, 1, 2, 3
        ).contiguous()  # (B, C, D, H, W)

        # Monitoring
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        used_codes = (self.cluster_size > 1e-5).sum().float() / self.num_embeddings

        # EMA updates (only during training)
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

                self.embedding.copy_(
                    self.embedding_avg / (cluster_size.unsqueeze(1) + self.epsilon)
                )

        # Straight-through estimator
        quantized_st = z + (quantized_reshaped - z).detach()

        # Compute commitment loss separately
        # Note: Stop gradient on quantized (i.e., codebook)
        commitment_loss = self.commitment_cost * F.mse_loss(quantized_reshaped.detach(), z)

        # (Optional) Codebook loss term: encourage embeddings to move toward encoder outputs
        # Not needed here because EMA updates already handle it

        # Total loss
        total_loss = commitment_loss

        # Reshape encoding indices to (B, D, H, W)
        encoding_indices = encoding_indices.view(
            input_shape[0], input_shape[2], input_shape[3], input_shape[4]
        )

        return quantized_st, total_loss, encoding_indices, perplexity, used_codes

    def printCodeBook(self):
        print("\nCodeBook")
        cont=0
        for i in range(self.num_embeddings):
            if sum(self.embedding[i])>0.001 or True:
                print(i, self.embedding[i])
            else:
                print(i,0)
                cont+=1
        print("0s",cont)
        print()
    #utils.weight_norm
import torch.nn.utils as utils
class Encoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=8, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.relu(self.conv4(x)))
        return  x

class Decoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channels=embedding_dim, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU6()
        self.patial_dropout3d = nn.Dropout3d(p=0.05)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.deconv1(x))
 
        x = torch.relu(self.deconv3(x))
      
        x = self.deconv4(x)
        return  x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # (B, C, D, H, W) -> (B*D*H*W, C)
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        # Distância entre z e embeddings
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_z, self.embeddings.weight.t())
            + torch.sum(self.embeddings.weight ** 2, dim=1)
        )  # (N, num_embeddings)

        # Índices mais próximos
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantizados
        quantized = torch.matmul(encodings, self.embeddings.weight).view(z_perm.shape)

        # Perda
        e_latent_loss = F.mse_loss(quantized.detach(), z_perm)
        q_latent_loss = F.mse_loss(quantized, z_perm.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Passo direto + gradiente fake
        quantized = z_perm + (quantized - z_perm).detach()

        # Retorna ao formato original
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()

        return quantized, loss, encoding_indices.view(z_perm.shape[0])

class NovaIDEIA(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 16
        self.num_embeddings = 32

        self.encoder = Encoder(embedding_dim=self.embedding_dim)
        self.vq = VectorQuantizerEMA(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        self.decoder = Decoder(embedding_dim=self.embedding_dim)


    def comparaEncoderQuant(self,x):
        self.eval()
        
        z = self.encoder(x)

        flat_input = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)  # (N, D)
        quantized, vq_loss, codes, perplexity, used_codes = self.vq(z)
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
    def getOptimizer(self,):
 
        from torch.optim import AdamW

        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        optimizer = Adam(
            self.parameters(),
            lr=3e-3,           # Learning rate base
            betas=(0.9, 0.95), # Momentos suaves
            weight_decay=1e-6  # L2 regularization
        )

        from torch.optim.lr_scheduler import CyclicLR



        scheduler = CyclicLR(
            optimizer,
            base_lr=1e-6,    # LR mínimo
            max_lr=1e-3,     # LR máximo
            step_size_up=2000,  # Número de batches para subir do base_lr ao max_lr
            mode='triangular',  # Outros modos: 'triangular2', 'exp_range'
            cycle_momentum=False  # Se usar otimizadores sem momentum, deixe False
        )
                

        return optimizer, scheduler

    def forward(self, x,epoch):
        z = self.encoder(x)
        
        
        epoch_inicial =25
        transi =25
        if epoch <= epoch_inicial:
            
            z_mix = z
     
            vq_loss = torch.tensor(0.0, device=x.device)
            codes = 0
            perplexity = 0.0
            used_codes = 0

        elif epoch <epoch_inicial+transi:
            self.vq.decay=0.9
            quantized, vq_loss, codes, perplexity, used_codes = self.vq(z)
            alpha = (epoch-epoch_inicial )/ transi
            z_mix = (1 - alpha) * z + alpha * quantized
     
            #vq_loss = torch.tensor(0.0, device=x.device)
            codes = 0
            perplexity = perplexity
            used_codes =  used_codes
        else:
            self.vq.decay=0.99
            quantized, vq_loss, codes, perplexity, used_codes = self.vq(z)
            z_mix = quantized  

        out = self.decoder(z_mix)
        return out ,vq_loss,codes,perplexity, used_codes 