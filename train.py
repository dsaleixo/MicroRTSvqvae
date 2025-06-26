
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from readDatas import ReadDatas
from vqvae import VQVAE


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    datas = ReadDatas.readDatas(128,device)
    print("load complete")
    datas = datas
    total_size = len(datas) 
    train_size = int(0.8 * total_size)  # 80 amostras para treino
    test_size = total_size - train_size  # 20 amostras para teste
    train_set, val_set = random_split(datas, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=160)
    val_loader = DataLoader(val_set, batch_size=128, )
    

    # Model Parameters
    num_hiddens = 64

    num_embeddings = 256 # Size of the codebook
    embedding_dim = 64   # Dimension of each embedding vector
    commitment_cost = 0.25
    from torch import nn
    def weights_init_kaiming(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    # Instantiate the VQ-VAE model
    model = VQVAE(num_hiddens,
                num_embeddings, embedding_dim, commitment_cost,device).to(device)
    n=100
    model.apply(weights_init_kaiming)

    model.loopTrain(max_epochs=30000,train_loader=train_loader,val_loader=val_loader)