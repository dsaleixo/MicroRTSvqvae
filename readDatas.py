import torch
import torch.nn.functional as F
import numpy as np




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
        for i in range(0,40):
                    for j in range(0,40):
                        loaded_data = np.load(f'datas3/data_{i}_{j}.npy')
                        shape = loaded_data.shape

                        aux = [ loaded_data]
                        for _ in range(size-shape[0]):
                            aux.append( np.expand_dims(loaded_data[-1].copy(), axis=0))

                        loaded_data2  = np.concatenate(aux, axis=0)
                        loaded_data2 = loaded_data2[0:size,:,:,:]
                        #print(i,j,loaded_data2.shape,len(dados))
                        dados.append(loaded_data2)
        total_size = len(dados)  # Suponha que temos 100 amostras
        for i in range(total_size):
            dados[i] = torch.tensor(dados[i],dtype=torch.float).to(device).permute(1, 0, 2, 3)
        return dados

if __name__ == "__main__":
     datas = ReadDatas.readDatas(64)
     print("readDatas",len(datas),datas[0].shape)