import torch
import numpy as np
import torch.nn.functional as F
class SimilaridadeExata:
    
    def __init__(self):
        pass

    def get(self,feat0,feat1):
        equal_mask = feat0 == feat1
        counts = torch.bincount(feat0.view(-1))

   
        # tensor([ True,  True, False,  True, False])

        # Contar quantos são True
        similarity = equal_mask.float().mean().item()
        
        return similarity



class SimilaridadeCos:
    
    def __init__(self,model):

        self.tableSimilarity = np.zeros((model.vq.num_embeddings,model.vq.num_embeddings))
        for i in range (model.vq.num_embeddings):
            for j in range(i,model.vq.num_embeddings):

                cos = (F.cosine_similarity(model.vq.embedding[i], model.vq.embedding[j], dim=0).item()+1)/2
                self.tableSimilarity[i][j] = cos
                self.tableSimilarity[j][i] = cos
          

    def get(self,feat0,feat1):
        idx1 = feat0.view(-1, 16).detach().cpu().numpy()
        idx2 = feat1.view(-1, 16).detach().cpu().numpy()

        # Extrair similaridades para cada par de índices
        similarities = self.tableSimilarity[idx1, idx2]  # shape igual ao dos códigos

        # Média
        mean_similarity = similarities.mean()
        
        return mean_similarity
            
     