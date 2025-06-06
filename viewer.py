from PIL import Image
import base64
import torch
import numpy as np
import os

from readDatas import ReadDatas


class Viewer:

    @staticmethod
    def one_hot_to_rgb(one_hot, palette):
        """
        Converte um tensor one-hot (soft ou hard) para RGB usando uma paleta.
        Args:
            one_hot: Tensor de shape (K, T, H, W).
            palette: Tensor de shape (K, 3), com cores RGB normalizadas entre 0 e 1.
        Retorna:
            rgb: Tensor de shape (3, T, H, W), com valores RGB.
        """
        # (K, T, H, W) → (T, H, W, K)
        one_hot = one_hot.permute(1, 2, 3, 0)
        # Multiplica cada pixel pela paleta: (T, H, W, 3)
        rgb = torch.einsum("thwk,kc->thwc", one_hot, palette)
        # (T, H, W, 3) → (3, T, H, W)
        return rgb.permute(3, 0, 1, 2)

    @staticmethod
    def generate_gif(inp2,path="Gifs/", suf="", ajeita=False):
        frames = []
        u = inp2.shape
        # Converte para RGB e reorganiza
        inp = Viewer.one_hot_to_rgb(inp2, ReadDatas.palette).reshape(u[1], 3, u[2], u[3])

        f, channels, height, width = inp.shape

        for i in range(f):
            img = Image.new("RGB", (width, height), "black")
            for a in range(height):
                for b in range(width):
                    k = inp2[:, i, a, b].argmax().item()
                    r, g, bb = ReadDatas.palette[k]
                    r = np.uint8(np.clip(r, 0, 1) * 255)
                    g = np.uint8(np.clip(g, 0, 1) * 255)
                    bb = np.uint8(np.clip(bb, 0, 1) * 255)
                    img.putpixel((b, a), (r, g, bb))
            img = img.resize((400, 400), Image.NEAREST)
            frames.append(img)

        output_name = f"{path}temp{suf}.gif"
        frames[0].save(output_name, save_all=True, append_images=frames[1:], duration=400, loop=0)
        return output_name

    @staticmethod
    def build_gif(inp,path="Gifs/", suf=""):
        name = Viewer.generate_gif(inp,path, suf)
        print(f"GIF gerado: {name}")

    @staticmethod
    def generateDoubleGif(inp, inp2,path="Gifs/", suf=""):
        inp=inp.permute(1, 0, 2, 3)
        inp2=inp2.permute(1, 0, 2, 3)
        print(inp.shape)
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
        


if __name__ == "__main__":
     datas = ReadDatas.readDatas(64)
     print("readDatas",len(datas),len(datas[0]))
     Viewer.generateDoubleGif(datas[0],datas[0])
