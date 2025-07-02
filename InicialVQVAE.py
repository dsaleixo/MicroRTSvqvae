
import torch
import torch.nn as nn

import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.60, epsilon: float = 1e-5):
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
        self.register_buffer("embedding", F.normalize(torch.randn(num_embeddings, embedding_dim), dim=1)*0.1 )
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
        if self.training :
            with torch.no_grad():
                new_cluster_size = encodings.sum(0)  # (M,)
                self.cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

                embed_sum = encodings.T @ flat_input  # (M, D)
                self.embedding_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

               
                self.embedding.copy_(self.embedding_avg / (self.cluster_size.unsqueeze(1) + self.epsilon))

        # Estimador Straight-Through
        quantized_st = z + (quantized - z).detach()

        # Perda de quantização
        loss = F.mse_loss(quantized_st, z)

        # Índices de saída
        encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3], input_shape[4])

        return quantized_st, loss, encoding_indices, perplexity, used_codes


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


class InitialVQVAE(nn.Module):
    def __init__(self) -> None:
        super(InitialVQVAE, self).__init__()
        self.num_embeddings = 32
        self.embedding_dim = 8
        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                stride=1, 
                padding=1       
            ),
            #nn.BatchNorm3d(num_features=8),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=8,
                out_channels=self.embedding_dim,
                kernel_size=3,
                stride=2,        
                padding=1
            ),
            #nn.BatchNorm3d(num_features=self.embedding_dim),
            nn.Tanh(),
            
        )
        self.vq = VectorQuantizerEMA(self.num_embeddings, self.embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.embedding_dim,
                out_channels=32,
                kernel_size=3,
                stride=2,        
                padding=1,
                output_padding=1
            ),
            nn.ReLU(inplace=True),
            #Snn.BatchNorm3d(num_features=32),
            nn.ConvTranspose3d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=1,        
                padding=1
            ),
            nn.ReLU(inplace=True),
        )

    def comparaEncoderQuant(self,x):
        self.eval()
        
        z = self.encoder(x)

        flat_input = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)  # (N, D)
        quantized, vq_loss, codes, perplexity, used_codes = self.vq(z)
        flat_quantized = quantized.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)  # (N, D)
        n = flat_quantized.shape[0]
        print("analise ",n)
        for i in range(n):
                if sum(flat_quantized[i])>16*3:
                    print(i,flat_quantized[i])
                    print(i,flat_input[i])
                    print()

        s= set(codes)
        print("fim",len(s))
    def getOptimizer(self,):
 
        from torch.optim import AdamW

        optimizer = AdamW(
            self.parameters(),
            lr=3,           # Learning rate base
            betas=(0.9, 0.95), # Momentos suaves
            weight_decay=1e-4
        )
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,      # Reduz o LR pela metade
            patience=10,     # Espera 10 epochs sem melhora
            threshold=1e-4,  # Quantidade mínima de melhora para resetar o contador
            min_lr=1e-8      # Nunca passa abaixo disso
        )
        

        return optimizer, scheduler

    def forward(self, x,epoch):
        z = self.encoder(x)
        
        
        epoch_inicial =20
        transi =20;
        if epoch <= epoch_inicial:
            
            z_mix = z
     
            vq_loss = torch.tensor(0.0, device=x.device)
            codes = 0
            perplexity = 0.0
            used_codes = 0

        elif epoch <epoch_inicial+transi:
            quantized, vq_loss, codes, perplexity, used_codes = self.vq(z)
            alpha = (epoch-epoch_inicial )/ transi
            z_mix = (1 - alpha) * z + alpha * quantized
     
            vq_loss = torch.tensor(0.0, device=x.device)
            codes = 0
            perplexity = 0.0
            used_codes = 0
        else:
            quantized, vq_loss, codes, perplexity, used_codes = self.vq(z)
            z_mix = quantized  

        out = self.decoder(z_mix)
        return out ,vq_loss,codes,perplexity, used_codes 