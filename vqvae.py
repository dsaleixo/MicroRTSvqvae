import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from lion_pytorch import Lion
'''
# --- 1. Define the Vector Quantization Layer ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings # M in the paper, size of the codebook
        self.embedding_dim = embedding_dim   # D in the paper, dimension of each embedding vector
        self.commitment_cost = commitment_cost # beta in the paper, weight for commitment loss

        # Initialize the embeddings (codebook) as a learnable parameter
        # Shape: (num_embeddings, embedding_dim)
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize weights with a uniform distribution for stability
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        # z is the output from the encoder: (batch_size, D, H, W, T) -> for 3D CNN, or (B, C, D, H, W) in PyTorch standard
        # For simplicity, let's assume z is already flattened or reshaped to (batch_size * num_pixels, embedding_dim)
        # However, a typical VQ-VAE encoder outputs (B, C, D, H, W) for video
        # We need to rearrange it to (B*D*H*W, C) for direct comparison with embeddings
        input_shape = z.shape
        # Flatten input to (batch_size * depth * height * width, embedding_dim)
        # PyTorch Conv3d outputs (B, C, D, H, W). We need (B*D*H*W, C)
        # Let's assume z is already like (B, C, D, H, W)
        flat_input = z.permute(0, 2, 3, 4, 1).contiguous() # Rearrange to (B, D, H, W, C)
        flat_input = flat_input.view(-1, self.embedding_dim) # Flatten to (N, C) where N = B*D*H*W

        # Calculate distances from input to embeddings (quantization)
        # (N, 1) - 2 * (N, C) @ (C, M) + (1, M) = (N, M)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
                     + torch.sum(self.embeddings.weight**2, dim=1, keepdim=True).t())

        # Find the encoding indices (closest embedding for each input vector)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (N, 1)

        # Convert indices to one-hot vectors
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1) # (N, M)

        # Quantize the input: get the actual embedding vectors
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)

        # --- Losses ---
        # Commitment loss: measures how much the encoder output needs to move to commit to an embedding
        # This part ensures the encoder's output doesn't stray too far from the codebook
        # (flat_input - quantized.detach()) is for the encoder's output
        # (quantized - flat_input.detach()) is for updating the embeddings themselves
        e_latent_loss = F.mse_loss(quantized.detach(), z) # Contribution from encoder
        q_latent_loss = F.mse_loss(quantized, z.detach()) # Contribution from codebook

        # Total VQ loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-Through Estimator:
        # In the backward pass, we want gradients to flow through 'quantized' as if it was 'z'
        # This makes the encoder's outputs directly influence the codebook's selection
        quantized = z + (quantized - z).detach()

        # Reshape encodings to original input shape (for potential use in prior models)
        # The encodings are (N, M). If you need it for a prior, you'd reshape it back
        # to (B, D, H, W, M) or similar. For simplicity, we return the (N, M) encodings.
        # This isn't directly used by the decoder, but rather the quantized tensor.
        return quantized, loss, encodings
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


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLu(inplace=True),
            nn.Conv3d(channels//2, channels, kernel_size=1, stride=1),
            nn.BatchNorm3d(channels),
            nn.ReLu(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int):
        super().__init__()

        # ✅ Nova camada antes do primeiro downsampling
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, num_hiddens // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_hiddens // 2),
            nn.ReLu(inplace=True)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv3d(num_hiddens // 2, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_hiddens // 2),
            nn.ReLu(inplace=True)
        )

        self.res_block_1 = ResidualBlock3D(num_hiddens // 2)

        self.conv_2 = nn.Sequential(
            nn.Conv3d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_hiddens),
            nn.ReLu(inplace=True)
        )

        self.res_block_2 = ResidualBlock3D(num_hiddens)

        self.conv_3 = nn.Sequential(
            nn.Conv3d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_hiddens),
            nn.ReLu(inplace=True)
        )

    def forward(self, inputs):
        x = self.initial_conv(inputs)
        x = self.conv_1(x)
        x = self.res_block_1(x)
        x = self.conv_2(x)
        x = self.res_block_2(x)
        x = self.conv_3(x)
        return x

# --- 3. Define the Decoder (3D CNN with Transposed Convolutions) ---
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1),
           # nn.BatchNorm3d(num_hiddens),
            nn.ReLu(inplace=True)
        )
        self.res_block_1 = ResidualBlock3D(num_hiddens)
        self.res_block_2 = ResidualBlock3D(num_hiddens)
        self.conv_trans_1 = nn.Sequential(
            nn.ConvTranspose3d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            #snn.BatchNorm3d(num_hiddens // 2),
            nn.ReLu(inplace=True)
        )
        self.conv_trans_2 = nn.ConvTranspose3d(num_hiddens // 2, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        #x = self.res_block_1(x)
        #x = self.res_block_2(x)
        x = self.conv_trans_1(x)
        x = torch.sigmoid(self.conv_trans_2(x))
        return x

# --- 4. Assemble the VQ-VAE Model ---
class VQVAE(nn.Module):
    def __init__(self, num_hiddens,
                num_embeddings, embedding_dim, commitment_cost,device):
        super().__init__()

        self.encoder = Encoder(3, num_hiddens,)
        self.pre_vq_conv = nn.Conv3d(num_hiddens, embedding_dim, kernel_size=1, stride=1) # Maps encoder output to embedding_dim
        self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim)
        #self.vq =VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, num_hiddens)

        self.palette = torch.tensor([
                [255,255,255],
                [200,200,200],
                [255,100,10],
                [255,255,0],
                [0, 255, 255],
                [0,0,0],
                [127,127,127],
            ], dtype=torch.float32).to(device)
        self.palette/=255

    def forward(self, x,epoch):
        # x is the input video (B, C, D, H, W) e.g., (1, 3, 32, 128, 128)
        z = self.encoder(x) # (B, num_hiddens, D/4, H/4, W/4)
        z = self.pre_vq_conv(z) # (B, embedding_dim, D/4, H/4, W/4)

        # Apply VQ layer
        if epoch > 20:
            was_training = self.vq.training

            # roda quantização com VQ-EMA
            quantized, vq_loss, encodings,perplexity, used_codes = self.vq(z)

            # aplica blending entre z e quantized
            # fator de mistura aumenta com o tempo
            #blend_epochs = 100  # ou o que fizer sentido para você
            #blend_factor = min(1.0, (epoch - 20) / blend_epochs)
            #quantized = (1 - blend_factor) * z + blend_factor * quantized

        else:
            # VQ ainda desativado
            quantized = z
            vq_loss = torch.tensor(0.0, device=x.device)
            encodings = None
            perplexity, used_codes =None,None
        # Decode the quantized latent features
        
        reconstructions = self.decoder(quantized)

        return reconstructions, vq_loss, encodings,perplexity, used_codes # encodings are the discrete indices (for prior training)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)


    '''
    def closest_palette_loss(self, pred_rgb, target_rgb, palette) -> torch.Tensor:
        """
        pred_rgb: (B, 3, T, H, W) — saída contínua da rede, valores em [0, 1]
        target_rgb: (B, 3, T, H, W) — imagem original, onde cada pixel é uma das 7 cores
        palette: (7, 3) — paleta de cores, valores em [0, 1]
        """
        device = pred_rgb.device
        B, _, T, H, W = pred_rgb.shape
        N = B * T * H * W

        pred_flat = pred_rgb.permute(0, 2, 3, 4, 1).reshape(N, 3)
        target_flat = target_rgb.permute(0, 2, 3, 4, 1).reshape(N, 3)

        # Distância da predição à cor-alvo
        target_dist = torch.norm(pred_flat - target_flat, dim=1)  # (N,)

        # Índices da paleta para cada pixel de alvo e predição
        target_dists_to_palette = torch.cdist(target_flat.unsqueeze(1), palette.unsqueeze(0))  # (N, 7)
        palette_indices = torch.argmin(target_dists_to_palette.squeeze(1), dim=1)  # (N,)

        pred_dists_to_palette = torch.cdist(pred_flat.unsqueeze(1), palette.unsqueeze(0))  # (N, 7)
        pred_closest = torch.argmin(pred_dists_to_palette.squeeze(1), dim=1)  # (N,)

        # Máscara onde a predição errou a cor
        mask_wrong = (pred_closest != palette_indices)  # (N,)

        # Máscara onde a cor-alvo é preto (ex: índice 0 da paleta)
        is_black = (palette_indices == 5)  # (N,)

        # Define pesos: menor peso para preto, peso 1 para outras cores
        weights = torch.ones(N, device=device)
        weights[is_black] = 0.1  # penalidade menor para preto

        # Aplica penalidade onde houve erro
        if mask_wrong.any():
            loss = (target_dist[mask_wrong] * weights[mask_wrong]).mean()
        else:
            loss = torch.tensor(0.0, device=device, dtype=pred_rgb.dtype)

        return loss

    '''
    def closest_palette_loss(self,pred_rgb, target_rgb, palette):
        """
        pred_rgb: (B, 3, T, H, W) — saída contínua da rede, valores em [0, 1]
        target_rgb: (B, 3, T, H, W) — imagem original, onde cada pixel é uma das 7 cores
        palette: (7, 3) — paleta de cores, valores em [0, 1]
        """
        device = pred_rgb.device
     

        B, _, T, H, W = pred_rgb.shape
        N = B * T * H * W

        # Flatten (N, 3)
        pred_flat = pred_rgb.permute(0, 2, 3, 4, 1).reshape(N, 3)
        target_flat = target_rgb.permute(0, 2, 3, 4, 1).reshape(N, 3)

        # Distância da predição à cor-alvo
        target_dist = torch.norm(pred_flat - target_flat, dim=1)  # (N,)

        # Índice da cor-alvo na paleta
        target_dists_to_palette = torch.cdist(target_flat.unsqueeze(1), palette.unsqueeze(0))  # (N, 7)
        palette_indices = torch.argmin(target_dists_to_palette.squeeze(1), dim=1)  # (N,)

        # Índice da cor da paleta mais próxima da predição
        pred_dists_to_palette = torch.cdist(pred_flat.unsqueeze(1), palette.unsqueeze(0))  # (N, 7)
        pred_closest = torch.argmin(pred_dists_to_palette.squeeze(1), dim=1)  # (N,)

        # Máscara de erro: só penaliza se a cor prevista for diferente da cor-alvo
        mask_wrong = (pred_closest != palette_indices)  # (N,)

        # Aplica a penalidade apenas onde errou
        if mask_wrong.any():
            loss = target_dist[mask_wrong].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return (loss/pred_rgb.shape[0])*10
   



    def baseline(self, val_loader: DataLoader, device='cuda'): 
        self.eval()
        total_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        vq_loss_epoch = 0.0
        loss_jesus_epoch = 0.0
        for batch in val_loader:
            x = batch.to(device,non_blocking=True)

           
            #reconstructions, vq_loss, _ = self(x)
            reconstructions = torch.zeros_like(x)
            reconstruction_loss = F.mse_loss(reconstructions, x)
            loss_jesus = self.closest_palette_loss(reconstructions, x,self.palette)
            total_loss = loss_jesus+reconstruction_loss#+# vq_loss
          
          


            total_loss_epoch += total_loss.item()
            recon_loss_epoch += reconstruction_loss.item()
            
            loss_jesus_epoch += loss_jesus.item()
        return total_loss_epoch, recon_loss_epoch,loss_jesus_epoch



    def validation(self, val_loader: DataLoader, device='cuda',): 
        self.eval()
        total_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        vq_loss_epoch = 0.0
        loss_jesus_epoch = 0.0
        for batch in val_loader:
            x = batch.to(device)

           
            #reconstructions, vq_loss, _ = self(x)
            reconstructions, vq_loss, _,_,_ = self(x,112)
            reconstruction_loss = F.mse_loss(reconstructions, x)
            loss_jesus = self.closest_palette_loss(reconstructions, x,self.palette)
            total_loss = loss_jesus+reconstruction_loss#+# vq_loss
            vq_loss_epoch += vq_loss.item()
          


            total_loss_epoch += total_loss.item()
            recon_loss_epoch += reconstruction_loss.item()
            
            loss_jesus_epoch += loss_jesus.item()
        return total_loss_epoch, recon_loss_epoch,loss_jesus_epoch ,vq_loss_epoch 

    def loopTrain(self, max_epochs: int, train_loader: DataLoader, val_loader: DataLoader, device='cuda'):
        with open('./saida42.txt', 'w') as f:
            pass
        self.to(device)
       
        optimizer = Lion(self.parameters(), lr=5e-4, weight_decay=0.0)
        
        # Agendador de taxa de aprendizado
   
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",          # ou "max" se você monitorar uma métrica que cresce (tipo PSNR)
            factor=0.5,
            patience=30,
            threshold=1e-4,
            min_lr=1e-7,
        
        )
        '''
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            steps_per_epoch=len(train_loader),
            epochs=200,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e4
        )
        '''
        bestTrain=100000000000000

        totalLossVal, reconLossVal,jesusLossVal =self.baseline(val_loader)
        print(f"Baseline 0 "
                    f"Total Loss: {totalLossVal:.4f}, "
                    f"Recon Loss: {reconLossVal:.4f}, "
                    f"Jesus Loss: {jesusLossVal:.4f}, "
                
                    )
        print()
        totalLossVal, reconLossVal,jesusLossVal,vqLossVal =self.validation(val_loader)
        print(f"Val 0 "
                    f"Total Loss: {totalLossVal:.4f}, "
                    f"Recon Loss: {reconLossVal:.4f}, "
                    f"Jesus Loss: {jesusLossVal:.4f}, "
                    f"VQ Loss: {vqLossVal:.4f}"
                    )
        bestVal = jesusLossVal
        for epoch in range(max_epochs):
            self.train()
            total_loss_epoch = 0.0
            recon_loss_epoch = 0.0
            vq_loss_epoch = 0.0
            loss_jesus_epoch = 0.0


            cont_batch=0
            n_batch = len(train_loader)
            for batch in train_loader:

                if cont_batch>n_batch*0.15 and epoch<0:
                    break
                cont_batch+=1

                x = batch.to(device)
                
                optimizer.zero_grad()
                #reconstructions, vq_loss, _ = self(x)
                reconstructions, vq_loss, _,perplexity, used_codes = self(x,epoch)
                reconstruction_loss = F.mse_loss(reconstructions, x)
                loss_jesus = self.closest_palette_loss(reconstructions, x,self.palette)
                total_loss = loss_jesus+reconstruction_loss#+# vq_loss
                #total_loss = loss_jesus#+vq_loss
                   
                total_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                vq_loss_epoch += vq_loss.item()
                total_loss_epoch += total_loss.item()
                recon_loss_epoch += reconstruction_loss.item()
                
                loss_jesus_epoch += loss_jesus.item()
            scheduler.step(total_loss_epoch)  # Atualiza o lr com o scheduler
            current_lr = scheduler.get_last_lr()[0]
            totalLossVal, reconLossVal,jesusLossVal,vqLossVal =self.validation(val_loader)
            with open('./saida42.txt', 'a') as f:
                    print(f"rl {current_lr}", file=f)
                    print(f"rl {current_lr}")
            if bestTrain>loss_jesus_epoch and epoch>20:
                bestTrain=loss_jesus_epoch
                torch.save(self.state_dict(), "BestTrainModel.pth")

            if bestVal >jesusLossVal and epoch>20:
                with open('./saida42.txt', 'a') as f:
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxUpdateXXXXXXXXXXXXXXXXXxx", file=f)
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxUpdateXXXXXXXXXXXXXXXXXxx")
                bestVal=jesusLossVal
                torch.save(self.state_dict(), f"BestTEstModelBest.pth")
                torch.save(self.state_dict(), f"BestTEstModel{epoch}.pth")

            
            if epoch%1==0:
                with open('./saida42.txt', 'a') as f:
    
                    print(f"Train [Epoch {epoch+1}/{max_epochs}] "
                        f"Total Loss: {total_loss_epoch:.4f}, "
                        f"Recon Loss: {recon_loss_epoch:.4f}, "
                        f"Jesus Loss: {loss_jesus_epoch:.4f}, "
                        f"VQ Loss: {vq_loss_epoch:.4f}", 
                        f"VQ perplexity: {perplexity}", 
                        f"VQ used_codes: {used_codes}", 
                        file=f
                      
                        )
                    print(f"Val [Epoch {epoch+1}/{max_epochs}] "
                        f"Total Loss: {totalLossVal:.4f}, "
                        f"Recon Loss: {reconLossVal:.4f}, "
                        f"Jesus Loss: {jesusLossVal}, "
                        f"VQ Loss: {vqLossVal:}", file=f
                        )
                    print()
                    print(f"Train [Epoch {epoch+1}/{max_epochs}] "
                        f"Total Loss: {total_loss_epoch:.4f}, "
                        f"Recon Loss: {recon_loss_epoch:.4f}, "
                        f"Jesus Loss: {loss_jesus_epoch:.4f}, "
                        f"VQ Loss: {vq_loss_epoch:.4f}"
                        f"VQ perplexity: {perplexity}", 
                        f"VQ used_codes: {used_codes}", 
                        )
                    print(f"Val [Epoch {epoch+1}/{max_epochs}] "
                        f"Total Loss: {totalLossVal:.4f}, "
                        f"Recon Loss: {reconLossVal:.4f}, "
                        f"Jesus Loss: {jesusLossVal:.4f}, "
                        f"VQ Loss: {vqLossVal:.4f}"
                        )
                    print()
            print(f"Memória alocada: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
            print(f"Memória reservada (cache): {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")
            torch.cuda.empty_cache()
            print(f"Memória alocada: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
            print(f"Memória reservada (cache): {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")
            mem = psutil.virtual_memory()
            print(f'Total: {mem.total / 1e9:.2f} GB')
            print(f'Usado: {mem.used / 1e9:.2f} GB')
            print(f'Livre: {mem.available / 1e9:.2f} GB')  
            

        print("Treinamento finalizado.")