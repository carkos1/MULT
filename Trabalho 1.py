import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
#import os
import cv2

down = 2                             #Change for downsampling:            1-> 4:2:2      2->4:2:0

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


def RGB_to_YCbCr(img, YCbCr):
    new_img = img.copy()
    new_img = new_img @ YCbCr.T 
    new_img[:, :, 1:] += 128
    return new_img

def downsappling(img_YCbCr): #no cv2.resize() linear mas se testarmos ganhamos um bónus
    new_img = img_YCbCr.copy()

    #4:2:2
    Y_d = new_img[:,:,0]      
    xsize = int((Y_d.shape[1]) /2)
    ysize = int(Y_d.shape[0] / down)
                             
    Cb_d_l= cv2.resize(np.round(new_img[:, :, 1]).astype(np.uint8), (xsize, ysize), interpolation = cv2.INTER_LINEAR) 
    Cr_d_l= cv2.resize(np.round(new_img[:, :, 2]).astype(np.uint8), (xsize, ysize), interpolation = cv2.INTER_LINEAR)
    
    Cb_d_c= cv2.resize(np.round(new_img[:, :, 1]).astype(np.uint8), (xsize, ysize), interpolation = cv2.INTER_CUBIC) 
    Cr_d_c= cv2.resize(np.round(new_img[:, :, 2]).astype(np.uint8), (xsize, ysize), interpolation = cv2.INTER_CUBIC)
    
    Cb_d_a= cv2.resize(np.round(new_img[:, :, 1]).astype(np.uint8), (xsize, ysize), interpolation = cv2.INTER_AREA) 
    Cr_d_a= cv2.resize(np.round(new_img[:, :, 2]).astype(np.uint8), (xsize, ysize), interpolation = cv2.INTER_AREA)
    
    return Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a

def encoder(img, YCbCr):
    R = img[:,:,0]
    R = padding(R)
    G = img[:,:,1]
    G = padding(G)
    B = img[:,:,2]
    B = padding(B)
    
    img_YCbCr = RGB_to_YCbCr(img,YCbCr)
    
    Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a = downsappling(img_YCbCr)
 
    return R, G, B, img_YCbCr, Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a


def removePadding(color, img):
    x, y, _ = img.shape
    color = color[:x, :y]
    
def YCbCr_to_RGB(img, YCbCr_inv):
    new_img = img.copy()
    new_img[:, :, 1:] -= 128
    new_img = new_img @ YCbCr_inv.T 
    return np.round(new_img).astype(np.uint8)

def upsampling(Y_d, Cb_d, Cr_d):   
#Para simplificar só vou dar upsampling das cenas em que usei interpolação linear
     Y_r = Y_d
    
     xsize = int(Y_d.shape[1])
     ysize = int(Y_d.shape[0])
    
     Cb_rebuilt= cv2.resize(np.round(Cb_d).astype(np.uint8), (xsize, ysize), interpolation = cv2.INTER_LINEAR) 
     Cr_rebuilt= cv2.resize(np.round(Cb_d).astype(np.uint8), (xsize, ysize), interpolation = cv2.INTER_LINEAR)
     
     return Y_r, Cb_rebuilt, Cr_rebuilt
    
    
def decoder(R, G, B, img, img_YCbCr,YCbCr_INV, Y_d, Cb_d, Cr_d):
    
    Y_r, Cb_rebuilt, Cr_rebuilt = upsampling(Y_d, Cb_d, Cr_d)
    
    YCbCr_rebuilt = YCbCr_to_RGB(img_YCbCr,YCbCr_INV)
    
    removePadding(R, img)
    removePadding(G, img)
    removePadding(B, img)
    nl, nc = R.shape
    imgRec = np.zeros((nl,nc,3), dtype = np.uint8)
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    
    return imgRec, YCbCr_rebuilt, Y_r, Cb_rebuilt, Cr_rebuilt


def choose_img(op):
    if op == 1:
        fName = r"./imagens/airport.bmp"
        #os.system('cls' if os.name == 'nt' else 'clear')
    elif op == 2:
        fName = r"./imagens/geometric.bmp"
        #os.system('cls' if os.name == 'nt' else 'clear')
    elif op == 3:
        fName = r"./imagens/nature.bmp"
        #os.system('cls' if os.name == 'nt' else 'clear')
    else:
        fName= "algo correu mal"
    return fName


def main():

    img_opt = 0
    
    
    while(img_opt != 1 and img_opt != 2 and img_opt != 3):
        print("\nOpções de Imagem:\n1- airport.bmp\n2- geometric.bmp\n3- nature.bmp")
        img_opt = int(input("Opt: "));
        fName =  choose_img(img_opt)
    
    cmRed = clr.LinearSegmentedColormap.from_list("Red", [(0,0,0), (1,0,0)], N=256)
    
    cmGreen = clr.LinearSegmentedColormap.from_list("Green", [(0,0,0), (0,1,0)], N=256)
    
    cmBlue = clr.LinearSegmentedColormap.from_list("Blue", [(0,0,0), (0,0,1)], N=256)

    cmGray = clr.LinearSegmentedColormap.from_list("Gray", [(0,0,0), (1,1,1)], N=256)

    YCbCr = np.array([[0.299,0.587,0.114],
             [-0.168736,-0.331264,0.5],
             [0.5,-0.418688,-0.081312]])
    
    YCbCr_INV = np.linalg.inv(YCbCr)
   
    img = plt.imread(fName)

    R, G, B, img_YCbCr, Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a = encoder(img, YCbCr)

    imgRec, YCbCr_rebuilt, Y_r, Cb_rebuilt, Cr_ebuilt  = decoder(R, G, B, img, img_YCbCr, YCbCr_INV, Y_d, Cb_d_l, Cr_d_l)

    print("\n\nOpções de Operação:\n1- R, G, B\n2- YCbCr\n3- Y, Cb, Cr\n4- Convertida\n5- Downsampling\n6- Upsampling\n0- Sair\n")
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
            showImg(YCbCr_rebuilt, "Imagem Convertida")
        elif opt == 5:
            inter_opt = 9

            while(inter_opt != 0):
                print("\n\nOpções de Interpolação:\n1- Interpolação Linear\n2- Interpolação Cúbica \n3- Interpolação Area\n0- Voltar Atrás\n")
                inter_opt = int(input("Opt: "));
                
                if inter_opt == 1:
                    showImg((Y_d), "Y_d", cmGray)
                    showImg(Cb_d_l, "Cb_d: Linear", cmGray)
                    showImg(Cr_d_l, "Cr_d: Linear", cmGray)
                elif inter_opt == 2:
                    showImg(Y_d, "Y_d", cmGray)
                    showImg(Cb_d_c, "Cb_d: Cúbica", cmGray)
                    showImg(Cr_d_c, "Cr_d: Cúbica", cmGray)
                elif inter_opt == 3:
                    showImg(Y_d, "Y_d", cmGray)
                    showImg(Cb_d_a, "Cb_d: Area", cmGray)
                    showImg(Cr_d_a, "Cr_d: Area", cmGray)
                    
        elif opt == 6:
            showImg(Y_r, "Y reconstruído", cmGray)
            showImg(Cb_rebuilt, "Cb reconstruído", cmGray)
            showImg(np.round(img_YCbCr[:,:,2]).astype(np.uint8), "Cr reconstruído", cmGray)
         
        print("\n\nOpções de Operação:\n1- R, G, B\n2- YCbCr\n3- Y, Cb, Cr\n4- Convertida\n5- Downsampling\n6- Upsampling\n0- Sair\n")            
        opt = int(input("Opt: "));
    
if __name__ == "__main__":
    main()
