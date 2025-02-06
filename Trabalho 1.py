import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def showImg(img, title, cmap = None):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()
def decoder(R, G, B):
    nl, nc = R.shape
    imgRec = np.zeros((nl,nc,3), dtype = np.uint8)
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    return imgRec

def encoder(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    return R, G, B

def main():
    fName = "./imagens/airport.bmp"
    
    cmRed = clr.LinearSegmentedColormap.from_list("Red", [(0,0,0), (1,0,0)], N=256)
    
    cmGreen = clr.LinearSegmentedColormap.from_list("Green", [(0,0,0), (0,1,0)], N=256)
    
    cmBlue = clr.LinearSegmentedColormap.from_list("Blue", [(0,0,0), (0,0,1)], N=256)

    cmGray = clr.LinearSegmentedColormap.from_list("Blue", [(0,0,0), (1,1,1)], N=256)

   
    img = plt.imread(fName)

    print(type(img))
    print(img.shape)
    print(img[0:8, 0:8, 0])
    print(img.dtype)


    R, G, B = encoder(img)

    imgRec = decoder(R, G, B)

    showImg(R,"Codificação a Vermelho",cmGray)

    showImg(imgRec, "Imagem Descodificada")

# PONTO 4 np.repeat 70/

if __name__ == "__main__":
    main()
