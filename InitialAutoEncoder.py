
import torch
import torch.nn as nn


class VideoAutoencoder(nn.Module):
    def __init__(self):
        super(VideoAutoencoder, self).__init__()
        
        self.encoder = nn.Conv3d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=(2,2,2),
            padding=(2,2,2)
        )
        
        self.decoder = nn.ConvTranspose3d(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
            stride=(2,2,2),
            padding=(2,2,2),
            output_padding=(1,1,1)  # importante para ajustar o tamanho final
        )
    

    def getOptimizer(self,):
        """
        Retorna optimizer e scheduler
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
   
        )

        return optimizer, scheduler

    def forward(self, x,epoch):
        z = self.encoder(x)
        out = self.decoder(z)
        return out ,None,0,0,0