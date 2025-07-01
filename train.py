
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from InicialVQVAE import InitialVQVAE
from InitialAutoEncoder import VideoAutoencoder
from readDatas import ReadDatas
import numpy as np
import os

os.environ["WANDB_API_KEY"] = "e6dd69e5ba37b74ef8d3ef0fa9dd28a33e4eeb6e"


import wandb

import moviepy
import imageio
from lion_pytorch import Lion
palette = torch.tensor([
                [255,255,255],
                [200,200,200],
                [255,100,10],
                [255,255,0],
                [0, 255, 255],
                [0,0,0],
                [127,127,127],
            ], dtype=torch.float32)
palette/=255

print(palette)
'''
def closest_palette_loss(pred_rgb, target_rgb, palette):
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

        return (loss)
'''




def closest_palette_loss(pred_rgb, target_rgb, palette):
    """
    pred_rgb: (B, 3, T, H, W)
    target_rgb: (B, 3, T, H, W)
    palette: (7, 3)
    """
    device = pred_rgb.device

    B, _, T, H, W = pred_rgb.shape
    N = B * T * H * W

    # Flatten (N,3)
    pred_flat = pred_rgb.permute(0,2,3,4,1).reshape(N,3)
    target_flat = target_rgb.permute(0,2,3,4,1).reshape(N,3)

    # Distâncias da predição para todas cores da paleta
    pred_dists = torch.cdist(pred_flat, palette)   # (N,7)
    pred_closest_idx = torch.argmin(pred_dists, dim=1)  # (N,)

    # Distâncias do target para todas cores da paleta
    target_dists = torch.cdist(target_flat, palette)  # (N,7)
    target_closest_idx = torch.argmin(target_dists, dim=1)  # (N,)

    # Máscara de erro
    mask_wrong = pred_closest_idx != target_closest_idx

    # Distância entre a predição e a cor-alvo da paleta
    target_palette_colors = palette[target_closest_idx]  # (N,3)
    penalization_dist = torch.norm(pred_flat - target_palette_colors, dim=1)

    if mask_wrong.any():
        loss = penalization_dist[mask_wrong].mean()
    else:
        loss = torch.tensor(0.0, device=device)

    return loss 


def baseline(model, val_loader: DataLoader, device='cuda'): 
        model.eval()
        total_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        loss_jesus_epoch = 0.0
        for batch in val_loader:
            x = batch.to(device,non_blocking=True)

           
            #reconstructions, vq_loss, _ = self(x)
            reconstructions = torch.zeros_like(x)
            reconstruction_loss = F.mse_loss(reconstructions, x)
            loss_jesus = closest_palette_loss(reconstructions, x,palette)
            total_loss = loss_jesus+reconstruction_loss#+# vq_loss
          
          


            total_loss_epoch += total_loss.item()
            recon_loss_epoch += reconstruction_loss.item()
            
            loss_jesus_epoch += loss_jesus.item()
        return total_loss_epoch, recon_loss_epoch,loss_jesus_epoch

def quantize_colors(video: torch.Tensor, ) -> torch.Tensor:

    C, T, H, W = video.shape
    assert C == 3, "Esperado 3 canais RGB"
    flat = video.permute(1,2,3,0).reshape(-1,3)  # (N,3)
    dists = torch.cdist(flat, palette)  # L2 distance
    indices = torch.argmin(dists, dim=1)  # (N,)
    quantized_flat = palette[indices]  # (N,3)
    quantized = quantized_flat.view(T,H,W,3).permute(3,0,1,2)  # (3,T,H,W)

    return quantized

def salva(name,out):
    video_np = (out.permute(1,2,3,0).detach().cpu().numpy() * 255).astype(np.uint8)
    frames=[]
    for frame in video_np:
        frames.append(frame)
    print(len(frames),frames[0].shape)
    imageio.mimsave("Gifs/"+name+'.gif', frames, fps=12)
    wandb.log({name: wandb.Video("Gifs/"+name+'.gif', format="gif")})
def gerarVideo(model, name,marchReal):
    model.eval()
    marchReal=marchReal.to(device)
    print("saida",marchReal.shape)
    out=model(marchReal,1000)[0][0]
    salva(name+"_Pure",out)

    out2 = quantize_colors(out)
    salva(name+"_Clean",out2)





def validation(model, val_loader: DataLoader, device='cuda',): 
    model.eval()
    total_loss_epoch = 0.0
    recon_loss_epoch = 0.0
    vq_loss_epoch = 0.0
    loss_jesus_epoch = 0.0
    for batch in val_loader:
        x = batch.to(device)

        reconstructions, vq_loss, _,_,_ = model(x,112)
        reconstruction_loss = F.mse_loss(reconstructions, x)
        loss_jesus = closest_palette_loss(reconstructions, x,palette)
        #total_loss = reconstruction_loss +loss_jesus
        total_loss = reconstruction_loss# +loss_jesus
        if vq_loss!=None:
            vq_loss_epoch += vq_loss.item()
        


        total_loss_epoch += total_loss.item()
        recon_loss_epoch += reconstruction_loss.item()
        
        loss_jesus_epoch += loss_jesus.item()
    return total_loss_epoch, recon_loss_epoch,loss_jesus_epoch ,vq_loss_epoch 


def loopTrain(model, max_epochs: int, train_loader: DataLoader, val_loader: DataLoader,marchReal, device='cuda'):

        model.to(device)
        '''
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
        optimizer,scheduler = model.getOptimizer()
        bestTrain=100000000000000

        totalLossVal, reconLossVal,jesusLossVal =baseline(model,val_loader)
        wandb.run.summary["BaseLine"] = f"Total Loss: {totalLossVal:.4f} Recon Loss: {reconLossVal:.4f} Jesus Loss: {jesusLossVal:.4f}"
       
        totalLossVal, reconLossVal,jesusLossVal,vqLossVal =validation(model,val_loader)
        wandb.log({
                "Val/Total Loss": totalLossVal,
                "Val/Recon Loss": reconLossVal,
                "Val/Jesus Loss": jesusLossVal,
                "Val/VQ Loss": vqLossVal,
        })

        bestVal = jesusLossVal
        nextSalve = 10
        for epoch in range(max_epochs):
            model.train()
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
                reconstructions, vq_loss, _,perplexity, used_codes = model(x,epoch)
                reconstruction_loss = F.mse_loss(reconstructions, x)
                loss_jesus = closest_palette_loss(reconstructions, x,palette)
                #total_loss = loss_jesus+reconstruction_loss#+# vq_loss
                total_loss = reconstruction_loss#+vq_loss
                   
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if vq_loss!=None:
                    vq_loss_epoch += vq_loss.item()
                total_loss_epoch += total_loss.item()
                recon_loss_epoch += reconstruction_loss.item()
                
                loss_jesus_epoch += loss_jesus.item()
                scheduler.step(total_loss_epoch)  # Atualiza o lr com o scheduler
                current_lr = scheduler.get_last_lr()[0]
            totalLossVal, reconLossVal,jesusLossVal,vqLossVal =validation(model,val_loader,device)
            wandb.log({
              "rl"   :current_lr    
             })

            if bestTrain>loss_jesus_epoch and epoch>50:
                bestTrain=loss_jesus_epoch
                torch.save(model.state_dict(), "BestTrainModel.pth")
                wandb.save("BestTrainModel.pth")
                gerarVideo(model,"BestTrain",marchReal)

            if bestVal >jesusLossVal and epoch>50:
                bestVal=jesusLossVal
                torch.save(model.state_dict(), f"BestTEstModelBest.pth")
                torch.save(model.state_dict(), f"BestTEstModel{epoch}.pth")
                wandb.save("BestTEstModelBest.pth")
                wandb.save(f"BestTEstModel{epoch}.pth")
                gerarVideo(model,"BestTest",marchReal)
                wandb.log({"Updade":1})
            else:
                 wandb.log({"Updade":0})
            if nextSalve==epoch:
                 gerarVideo(model,"Actual",marchReal)
                 model.vq.printCodeBook()
                 model.comparaEncoderQuant(marchReal.to(device))
                 nextSalve = nextSalve+20

            print(epoch,total_loss_epoch,totalLossVal)
            wandb.log({
                "Train/Total Loss": total_loss_epoch,
                "Train/Recon Loss": recon_loss_epoch,
                "Train/Jesus Loss": loss_jesus_epoch,
                "Train/VQ Loss": vq_loss_epoch,
                "Train/VQ Perplexity": perplexity,
                "Train/VQ Used Codes": used_codes,
                
                "Val/Total Loss": totalLossVal,
                "Val/Recon Loss": reconLossVal,
                "Val/Jesus Loss": jesusLossVal,
                "Val/VQ Loss": vqLossVal,
   
            })

            wandb.log({  
                "Memória alocada": torch.cuda.memory_allocated() / 1024**2,
                "Memória reservada (cache)": torch.cuda.memory_reserved() / 1024**2
            })
            torch.cuda.empty_cache()
            wandb.log({  
                "Memória alocada": torch.cuda.memory_allocated() / 1024**2,
                "Memória reservada (cache)": torch.cuda.memory_reserved() / 1024**2
            })
    
            

        print("Treinamento finalizado.")





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    palette=palette.to(device)
    import os

    wandb.init(
    project="VQVAE",
    name = "ArquiteturaVQVAEFinal20",
    config={
         
      
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001
        }
    )
    '''
    frames = []

    # Suponha que video_data seja torch tensor (T, C, H, W)
    video_data = torch.rand(12, 3, 64, 64)

    # Converter para numpy (T, H, W, C)
    video_np = (video_data.numpy() * 255).astype(np.uint8)

    for frame in video_np:
        frames.append(frame)

    imageio.mimsave('video.gif', frames, fps=12)

    wandb.log({"meu_video": wandb.Video("video.gif", format="gif")})
    print(f"Using device: {device}")
    '''
    sizeVideo =128

    datas = ReadDatas.readDatas(sizeVideo,device)
    print("load complete",len(datas) )
    datas = datas
    total_size = len(datas) 
    train_size = int(0.8 * total_size)  # 80 amostras para treino
    test_size = total_size - train_size  # 20 amostras para teste
    train_set, val_set = random_split(datas, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32)
    val_loader = DataLoader(val_set, batch_size=32, )
 
    exemplo = np.load("resultado.npy")[:sizeVideo]




    marchReal = torch.tensor(exemplo).float()
    print(marchReal[0][0])

    marchReal = marchReal.permute(1,0,2,3).unsqueeze(0)
    
    salva("RealVideo",marchReal.squeeze())
    

    
    model = InitialVQVAE().to(device)
 
    loopTrain(model, 10000, train_loader, val_loader,marchReal, device)

    