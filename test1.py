
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from InicialVQVAE import InitialVQVAE
from InitialAutoEncoder import VideoAutoencoder
from NovaIdeia import NovaIDEIA
from NovaNovaIdeia import VideoVQVAETransformer
from readDatas import ReadDatas
import numpy as np
import os

from similary import SimilaridadeCos, SimilaridadeExata
from simples_transformer import ST
from st2 import ST2
from stFirst import STFirst

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



from PIL import Image
import imageio

def juntar_gifs_lado_a_lado(gifs: list[str], saida: str = "saida.gif") -> None:
    # Carregar todos os gifs
    leitores = [imageio.get_reader(g) for g in gifs]

    # Número de frames será o mínimo entre os gifs (para evitar erro de comprimento)
    num_frames = min([len(l) for l in leitores])

    frames = []
    for i in range(num_frames):
        imagens = [Image.fromarray(l.get_data(i)) for l in leitores]  # <-- CORRIGIDO

        # Opcional: redimensionar para mesma altura
        alturas = [img.height for img in imagens]
        altura_min = min(alturas)
        imagens = [
            img.resize((int(img.width * altura_min / img.height), altura_min), Image.Resampling.LANCZOS)
            for img in imagens
        ]

        # Concatenar horizontalmente
        largura_total = sum(img.width for img in imagens)
        nova_img = Image.new("RGBA", (largura_total, altura_min))

        x_offset = 0
        for img in imagens:
            nova_img.paste(img, (x_offset, 0))
            x_offset += img.width

        frames.append(nova_img)

    # Salvar como gif animado
    frames[0].save(
        saida,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # tempo entre frames em ms
        loop=0
    )

    # Fechar os leitores
    for l in leitores:
        l.close()

def salva(name,out):
    video_np = (out.permute(1,2,3,0).detach().cpu().numpy() * 255).astype(np.uint8)
    frames=[]
    for frame in video_np:
        frames.append(frame)
    print(len(frames),frames[0].shape)
    imageio.mimsave("Gifs/"+name+'.gif', frames, fps=12)
    #wandb.log({name: wandb.Video("Gifs/"+name+'.gif', format="gif")})
def gerarVideo(model, name,marchReal):
    model.eval()
    marchReal=marchReal.unsqueeze(0).to(device)
    print("saida",marchReal.shape)
    out=model(marchReal,1000)[0][0]
    salva(name+"_Real",marchReal[0])
    salva(name+"_Pure",out)

    out2 = quantize_colors(out)
    salva(name+"_Clean",out2)
    juntar_gifs_lado_a_lado(["Gifs/"+name+"_Real.gif", "Gifs/"+name+"_Pure.gif", "Gifs/"+name+"_Clean.gif"], name+".gif")



def normalize(values: list[float]) -> list[float]:
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0 for _ in values]  # evita divisão por zero
    return [(x - min_val) / (max_val - min_val) for x in values]

def validation(model, val_loader: DataLoader, device='cuda',): 
    model.eval()
    total_loss_epoch = 0.0
    recon_loss_epoch = 0.0
    vq_loss_epoch = 0.0
    loss_jesus_epoch = 0.0
    for batch in val_loader:
        x = batch.to(device)

        reconstructions, vq_loss, _,_,_ = model(x,11)
        reconstruction_loss = F.mse_loss(reconstructions, x)
        loss_jesus = closest_palette_loss(reconstructions, x,palette)
        total_loss = loss_jesus+reconstruction_loss +vq_loss*5
        #total_loss = reconstruction_loss# +loss_jesus
        if vq_loss!=None:
            vq_loss_epoch += vq_loss.item()
        


        total_loss_epoch += total_loss.item()
        recon_loss_epoch += reconstruction_loss.item()
        
        loss_jesus_epoch += loss_jesus.item()
    return total_loss_epoch, recon_loss_epoch,loss_jesus_epoch ,vq_loss_epoch 


def printCode(model,datasLS):
    model.eval()
    codes =model.getFeature(datasLS.to(device))
    n = codes.shape[0]
    for i in range(n):
        print(i, codes[i])

def testLS(model,datasLS):
    model.eval()
    codes =model.getFeature(datasLS.to(device))

    n = codes.shape[0]
    exact = SimilaridadeExata()
    simExact = []
    print("SimExat")
    for i in range(1,n):
        sim = exact.get(codes[0],codes[i])
        simExact.append(float(sim))
        print(i,simExact[-1])
    simExact=normalize(simExact)
    
    cos = SimilaridadeCos(model)
    simCos = []
    print("SimCos")
    for i in range(1,n):
        sim = cos.get(codes[0],codes[i])
        simCos.append(float(sim))
        print(i,simCos[-1])
    simCos=normalize(simCos)
    print("cos",simCos)
    indices = [str(x) for x in range(len(simCos))]
    # Gráfico Cos
    plt.figure()
    plt.bar(indices, simCos)
    plt.title("SimCos - Gráfico de Barras")
    plt.ylabel("Valor")
    plt.xlabel("Rótulo")
    wandb.log({"LStest/Cos": wandb.Image(plt)})
    plt.close()

    # Gráfico Exact
    plt.figure()
    plt.bar(indices, simExact)
    plt.title("SimExact - Gráfico de Barras")
    plt.ylabel("Valor")
    plt.xlabel("Rótulo")
    wandb.log({"LStest/Exact": wandb.Image(plt)})
    plt.close()
 
  
  
   

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    palette=palette.to(device)
    import os
   
  
    sizeVideo =128
    
    
   
    
    exemplo = np.load("resultado.npy")[:sizeVideo]
 
   


  
    marchReal = ReadDatas.readDatasVal(sizeVideo,device)


    
    #salva("RealVideo",marchReal[0])
    

    
    model = ST2().to(device)



    state_dict = torch.load("./Best0.pth")
        #state_dict = torch.load("./testsVQVAE/model/test.pth")
        # 3. Preencha os pesos
    model.load_state_dict(state_dict)
    print("code")
    for i in range(32):
        print(model._vq.embedding[i])
    print("video")
    printCode(model,marchReal)
    for i in range(marchReal.shape[0]):
        gerarVideo(model,"tests"+str(i),marchReal[i])
    
