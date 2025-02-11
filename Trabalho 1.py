import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def showImg(img, title, cmap = None):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()
    
def padding(color):
    nl, nc = color.shape

    xresto = nl % 32
    yresto = nc % 32

     
    if xresto != 0:
        xfalta = 32 - xresto
    else:
        xfalta = 0
        
    if yresto != 0:
        yfalta = 32 - yresto
    else:
        yfalta = 0


    last_col = color[:, -1:]
    new_cols = np.repeat(last_col, yfalta, axis=1)

    color_padded = np.hstack((color, new_cols))

    last_row = color_padded[-1:, :] 
    new_rows = np.repeat(last_row, xfalta, axis=0)

    color_padded = np.vstack((color_padded, new_rows))

    return color_padded

def removePadding(color):
    last_col = color[:, -1]
    previous_col = color[:, -2]
    
    while np.array_equal(previous_col, last_col):
        color = np.delete(color, -1, axis=1)  # Removendo a última coluna
        
        last_col = color[:, -1]
        previous_col = color[:, -2]

    last_row = color[-1, :]
    previous_row = color[-2, :]

    while np.array_equal(previous_row, last_row):
        color = np.delete(color, -1, axis=0)  # Removendo a última linha
        
        last_row = color[-1, :]
        previous_row = color[-2, :]

    return color

    
def decoder(R, G, B):
    R = removePadding(R)
    G = removePadding(G)
    B = removePadding(B)
    nl, nc = R.shape
    imgRec = np.zeros((nl,nc,3), dtype = np.uint8)
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    return imgRec

def encoder(img):
    R = img[:,:,0]
    R = padding(R)
    G = img[:,:,1]
    G = padding(G)
    B = img[:,:,2]
    B = padding(B)

    return R, G, B

def main():
    fName = "./imagens/airport.bmp"
    
    cmRed = clr.LinearSegmentedColormap.from_list("Red", [(0,0,0), (1,0,0)], N=256)
    
    cmGreen = clr.LinearSegmentedColormap.from_list("Green", [(0,0,0), (0,1,0)], N=256)
    
    cmBlue = clr.LinearSegmentedColormap.from_list("Blue", [(0,0,0), (0,0,1)], N=256)

    cmGray = clr.LinearSegmentedColormap.from_list("Gray", [(0,0,0), (1,1,1)], N=256)

   
    img = plt.imread(fName)

    print(type(img))
    print(img.shape)
    print(img[0:8, 0:8, 0])
    print(img.dtype)


    R, G, B = encoder(img)

    imgRec = decoder(R, G, B)

    

    showImg(img, "Imagem Descodificada")
    showImg(R,"Codificação a Cinzento", cmGray)
    showImg(R,"Codificação a Vermelho", cmRed)
    showImg(G,"Codificação a Verde", cmGreen)
    showImg(B,"Codificação a Azul", cmBlue)
    
    showImg(imgRec,"Imagem após decoder", cmGray)

# PONTO 4 np.repeat 70/

if __name__ == "__main__":
    main()
