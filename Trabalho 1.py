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

def removePadding(color, img):
    x, y, _ = img.shape
    color = color[:x, :y]
    
def decoder(R, G, B, img):
    removePadding(R, img)
    removePadding(G, img)
    removePadding(B, img)
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

def RGB_to_YCbCr(img, YCbCr):
    new_img = img.copy()
    new_img = new_img @ YCbCr.T 
    new_img[:, :, 1:] += 128
    return new_img

def YCbCr_to_RGB(img, YCbCr_inv):
    new_img = img.copy()
    new_img[:, :, 1:] -= 128
    new_img = new_img @ YCbCr_inv.T 
    return np.round(new_img).astype(np.uint8)

def choose_img(op):
    if op == 1:
        fName = r"./imagens/airport.bmp"
    elif op == 2:
        fName = r"./imagens/geometric.bmp"
    elif op == 3:
        fName = r"./imagens/nature.bmp"
    else:
        fName= "algo correu mal"
    return fName

def main():
    
    img_opt = 0
    fName = "./imagens/airport.bmp"
    

    while(img_opt != 1 and img_opt != 2 and img_opt != 3):
        print("1- airport.bmp\n2- geometric.bmp\n3- nature.bmp")
        img_opt = int(input("Opt: "));
        fName =  choose_img(img_opt)

    
    cmRed = clr.LinearSegmentedColormap.from_list("Red", [(0,0,0), (1,0,0)], N=256)
    
    cmGreen = clr.LinearSegmentedColormap.from_list("Green", [(0,0,0), (0,1,0)], N=256)
    
    cmBlue = clr.LinearSegmentedColormap.from_list("Blue", [(0,0,0), (0,0,1)], N=256)

    cmGray = clr.LinearSegmentedColormap.from_list("Gray", [(0,0,0), (1,1,1)], N=256)

   
    img = plt.imread(fName)

    # print(type(img))
    # print(img.shape)
    # print(img[0:8, 0:8, 0])
    # print(img.dtype)


    R, G, B = encoder(img)

    imgRec = decoder(R, G, B, img)

    YCbCr = np.array([[0.299,0.587,0.114],
             [-0.168736,-0.331264,0.5],
             [0.5,-0.418688,-0.081312]])
    
    YCbCr_INV = np.linalg.inv(YCbCr)

    #print("Função para converter para YCbCr")
    img_YCbCr = RGB_to_YCbCr(img,YCbCr)

    #print("Função para converter para RGB")
    img_rec = YCbCr_to_RGB(img_YCbCr,YCbCr_INV)

    print("\n1-R, G, B\n2-YCbCr\n3-Y, Cb, Cr\n4-Convertida\n0-Sair\n")
    opt = int(input("Opt: "));

    while(opt != 0):
        if opt == 1:
            showImg(R,"Codificação a Vermelho", cmRed)
            showImg(G,"Codificação a Verde", cmGreen)
            showImg(B,"Codificação a Azul", cmBlue)
        elif opt == 2:
            showImg(np.round(img_YCbCr).astype(np.uint8), "Imagem YCbCr")
        elif opt == 3:
            showImg(np.round(img_YCbCr[:,:,0]).astype(np.uint8), "Y", cmGray)
            showImg(np.round(img_YCbCr[:,:,1]).astype(np.uint8), "Cb", cmGray)
            showImg(np.round(img_YCbCr[:,:,2]).astype(np.uint8), "Cr", cmGray)
        elif opt == 4:
            showImg(img_rec, "Imagem Convertida")
        
        print("\n1-R, G, B\n2-YCbCr\n3-Y, Cb, Cr\n4-Convertida\n0-Sair\n")
        opt = int(input("Opt: "));
    
    # print("Invertida")
    # print(YCbCr_INV)
    
    # showImg(imgRec,"Imagem após decoder", cmGray)

# PONTO 4 np.repeat 70/
# PONTO 5 fzr YCbCr matriz e cálculo nos slides para o decoder inverter matriz com np.linalg.inv

if __name__ == "__main__":
    main()
