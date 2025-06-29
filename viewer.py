from PIL import Image
import base64
import torch
import numpy as np
import os

 


class Viewer:
    cores = [(255,255,255),(200,200,200),(255,100,10), (255,255,0),(0, 255, 255),(0,0,0),(127,127,127)]

    def trocarCores(r,g,bb):
        i=-1
        dist= 1000
        for c in range(len(Viewer.cores)):
            auxDist = ((float(r)-Viewer.cores[c][0])/255)**2
            auxDist += ((float(g)-Viewer.cores[c][1])/255)**2
            auxDist += ((float(bb)-Viewer.cores[c][2])/255)**2
            if auxDist < dist or i==-1:
                dist = auxDist
                i =c
        return Viewer.cores[i]
    
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

    @staticmethod
    def generate_gif(inp2,path="Gifs/", suf="", ajeita=False):
        frames = []
        u = inp2.shape
        # Converte para RGB e reorganiza
        print("uuu ",u)
        #inp = Viewer.one_hot_to_rgb(inp2, Viewer.palette).reshape(u[1], 3, u[2], u[3])

        f, channels, height, width = inp2.shape

        for i in range(f):

            img = Image.new("RGB", (height, width), "black")
            for a in range(height):
                for b in range(width):
                    r,g,bb= inp2[i][0][a][b],inp2[i][1][a][b],inp2[i][2][a][b]
                    r = max(min(1,r),0)
                    g = max(min(1,g),0)
                    bb = max(min(1,bb),0)
                    r=np.uint8(r*255)
                    g=np.uint8(g*255)
                    bb=np.uint8(bb*255)
                    if ajeita:
                        r,g,bb = Viewer.trocarCores(r,g,bb)


                    img.putpixel((b, a), (r, g, bb))
            frames.append(img.resize((400, 400), Image.NEAREST))

        output_name = f"{path}temp{suf}.gif"
        print("frames ",len(frames))
        frames[0].save(output_name, save_all=True, append_images=frames[1:], duration=400, loop=0)
        return output_name

    @staticmethod
    def build_gif(inp,path="Gifs/", suf=""):
        name = Viewer.generate_gif(inp.permute(1, 0, 2, 3).detach().cpu().numpy(),path, suf)
        print(f"GIF gerado: {name}")

    @staticmethod
    def generateDoubleGif(inp, inp2,path="Gifs/", suf=""):

     
        name1 = Viewer.generate_gif(inp,path, suf + "_1")
        name2 = Viewer.generate_gif(inp,path, suf + "_2", True)
        name3 = Viewer.generate_gif(inp2,path, suf + "_3")

        def img_to_base64(src):
            with open(src, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode("utf-8")

        img1_base64 = img_to_base64(name1)
        img2_base64 = img_to_base64(name2)
        img3_base64 = img_to_base64(name3)

        html_code = f'''
        <html>
        <body>
        <div style="display: flex; gap: 10px;">
            <img src="data:image/gif;base64,{img1_base64}" width="200" />
            <img src="data:image/gif;base64,{img2_base64}" width="200" />
            <img src="data:image/gif;base64,{img3_base64}" width="200" />
        </div>
        </body>
        </html>
        '''

        # Salvar HTML em arquivo
        with open(f"{path}temp{suf}_duplo.html", "w") as f:
            f.write(html_code)
        


