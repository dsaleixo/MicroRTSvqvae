import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path


class ReadDatas():
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

   


    def readDatas(size,device):
        dados = []
        folder_path = Path('./datas3/')
        arquivos =  [f.name for f in folder_path.iterdir() if f.is_file()]
        print(len(arquivos))
        cont=0
        for arq in arquivos:
                    if cont>100:
                        break
                    cont+=1
                    print(arq)
                    loaded_data = np.load('./datas3/'+arq)
                    shape = loaded_data.shape
                    #print(shape,len(dados))
                    aux = [ loaded_data]
                    for _ in range(size-shape[0]):
                        aux.append( np.expand_dims(loaded_data[-1].copy(), axis=0))

                    loaded_data2  = np.concatenate(aux, axis=0)
                    loaded_data2 = loaded_data2[0:size,:,:,:]
                    
                    dados.append(loaded_data2)
        total_size = len(dados)  # Suponha que temos 100 amostras
        for i in range(total_size):
            dados[i] = torch.tensor(dados[i],dtype=torch.float).permute(1, 0, 2, 3)
        return dados

if __name__ == "__main__":
     datas = ReadDatas.readDatas(64)
     print("readDatas",len(datas),datas[0].shape)