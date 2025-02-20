import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy
import cv2


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

def downsappling(img_YCbCr, down): 

    new_img = img_YCbCr.copy()

    Y_d = new_img[:,:,0]      
    xsize = int((Y_d.shape[1]) /2)
    ysize = int(Y_d.shape[0] / down)

    #Estou a usar os 3 tipos de interpolação para testes e para meter informação no relatório, no trabalho final é só para deixar a linear
                             
    Cb_d_l= cv2.resize(new_img[:, :, 1], (xsize, ysize), interpolation = cv2.INTER_LINEAR) 
    Cr_d_l= cv2.resize(new_img[:, :, 2], (xsize, ysize), interpolation = cv2.INTER_LINEAR)
    
    Cb_d_c= cv2.resize(new_img[:, :, 1], (xsize, ysize), interpolation = cv2.INTER_CUBIC) 
    Cr_d_c= cv2.resize(new_img[:, :, 2].astype(np.uint8), (xsize, ysize), interpolation = cv2.INTER_CUBIC)
    
    Cb_d_a= cv2.resize(new_img[:, :, 1], (xsize, ysize), interpolation = cv2.INTER_AREA) 
    Cr_d_a= cv2.resize(new_img[:, :, 2], (xsize, ysize), interpolation = cv2.INTER_AREA)
    
    return Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a

def DCT(Y_d, Cb_d_l, Cr_d_l):
    Y_d = scipy.fft.dct(Y_d,norm = "ortho").T
    Y_d = scipy.fft.dct(Y_d,norm = "ortho").T

    Cb_d_l = scipy.fft.dct(Cb_d_l,norm = "ortho").T
    Cb_d_l = scipy.fft.dct(Cb_d_l,norm = "ortho").T
        
    Cr_d_l = scipy.fft.dct(Cr_d_l,norm = "ortho").T
    Cr_d_l = scipy.fft.dct(Cr_d_l,norm = "ortho").T
    
    return Y_d, Cb_d_l, Cr_d_l

def DCT_Blocks(img, block_size):
    h, w = img.shape
    dct = np.zeros((h,w))

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            dct_block = scipy.fft.dct(block, norm = "ortho").T
            dct_block = scipy.fft.dct(dct_block, norm = "ortho").T
            dct[i:i+block_size, j:j+block_size] = dct_block

    return dct

def quality_coeficient(quality):
    if quality < 50:
        return 5000/quality
    elif quality < 100:
        return (200-2*quality)
    else:
        return 1


def qauntize_block(block, q):
    
    block_sf = np.zeros((8,8))
    
    for i in range(0, 8):
        for j in range(0, 8):
            quantized = np.round(block[i,j] / q[i,j])
            block_sf[i,j] = quantized
            
    return block_sf.astype(int)
            
    
def sf(dct,q, quality):
   h, w = dct.shape
   sf = np.zeros((h,w))
   
   S = quality_coeficient(quality)
   
   q = ((S * q + 50)/100)
   
   for i in range(0, h, 8):
       for j in range(0, w, 8):
           block = dct[i:i+8, j:j+8]
           
           
           block_sf = qauntize_block(block, q)
           
           
           sf[i:i+8,j:j+8] = block_sf
           
   return sf


def encoder(img, YCbCr, cmGray, down,Q_Y,Q_CbCr, quality):
    R = img[:,:,0]
    R = padding(R)
    G = img[:,:,1]
    G = padding(G)
    B = img[:,:,2]
    B = padding(B)
    
    nl, nc = R.shape
    img_YCbCr = np.zeros((nl,nc,3), dtype = np.uint8)
    
    img_YCbCr[:,:,0] = R
    img_YCbCr[:,:,1] = G
    img_YCbCr[:,:,2] = B
    
    img_YCbCr = RGB_to_YCbCr(img_YCbCr,YCbCr)
    
    Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a = downsappling(img_YCbCr, down)
    
    Y_dct8 = DCT_Blocks(Y_d, 8)
    Cb_dct8 = DCT_Blocks(Cb_d_l, 8)
    Cr_dct8 = DCT_Blocks(Cr_d_l, 8)

    Y_dct64 = DCT_Blocks(Y_d, 64)
    Cb_dct64 = DCT_Blocks(Cb_d_l, 64)
    Cr_dct64 = DCT_Blocks(Cr_d_l, 64)

    Yb_Q = sf(Y_dct8,Q_Y, quality)

    Cbb_Q = sf(Cb_dct8,Q_CbCr, quality)

    Crb_Q = sf(Cr_dct8,Q_CbCr, quality)

    Y_dct, Cb_dct, Cr_dct = DCT(Y_d, Cb_d_l, Cr_d_l)
    
    return R, G, B, img_YCbCr, Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a, Y_dct, Cb_dct, Cr_dct, Y_dct8, Cb_dct8, Cr_dct8, Y_dct64, Cb_dct64, Cr_dct64, Yb_Q, Cbb_Q, Crb_Q


def removePadding(color, img):
    x, y, _ = img.shape
    color = color[:x, :y]
    return color
    
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
    
     Cb_rebuilt= cv2.resize(Cb_d, (xsize, ysize), interpolation = cv2.INTER_LINEAR) 
     Cr_rebuilt= cv2.resize(Cr_d, (xsize, ysize), interpolation = cv2.INTER_LINEAR)
     
     return Y_r, Cb_rebuilt, Cr_rebuilt
 
def DCT_inv(Y_d, Cb_d_l, Cr_d_l):
    Y_d = scipy.fft.idct(Y_d,norm = "ortho").T
    Y_d = scipy.fft.idct(Y_d,norm = "ortho").T

    Cb_d_l = scipy.fft.idct(Cb_d_l,norm = "ortho").T
    Cb_d_l = scipy.fft.idct(Cb_d_l,norm = "ortho").T
    
    Cr_d_l = scipy.fft.idct(Cr_d_l,norm = "ortho").T
    Cr_d_l = scipy.fft.idct(Cr_d_l,norm = "ortho").T

    return Y_d, Cb_d_l, Cr_d_l
    
def decoder(R, G, B, img, img_YCbCr,YCbCr_INV, Y_d, Cb_d, Cr_d, Y_dct, Cb_dct, Cr_dct ,cmGray):
    
    Y_d, Cb_d, Cr_d = DCT_inv(Y_dct, Cb_dct, Cr_dct)
  
    Y_r, Cb_rebuilt, Cr_rebuilt = upsampling(Y_d, Cb_d, Cr_d)
    
    YCbCr_rebuilt = img_YCbCr.copy()
    
    YCbCr_rebuilt[:,:,0] = Y_r
    YCbCr_rebuilt[:,:,1] = Cb_rebuilt
    YCbCr_rebuilt[:,:,2] = Cr_rebuilt
    
    YCbCr_rebuilt = YCbCr_to_RGB(YCbCr_rebuilt ,YCbCr_INV)
    
    R = removePadding(YCbCr_rebuilt[:,:,0], img)
    G = removePadding(YCbCr_rebuilt[:,:,1], img)
    B = removePadding(YCbCr_rebuilt[:,:,2], img)
    
    nl, nc = R.shape
    imgRec = np.zeros((nl,nc,3), dtype = np.uint8)
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    
    return imgRec, YCbCr_rebuilt, Y_r, Cb_rebuilt, Cr_rebuilt


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
    
    down = 0
    
    while(img_opt != 1 and img_opt != 2 and img_opt != 3):
        print("\nOpções de Imagem:\n1- airport.bmp\n2- geometric.bmp\n3- nature.bmp")
        img_opt = int(input("Opt: "));
        fName =  choose_img(img_opt)
        
    while(down != 1 and down != 2):
         print("\nOpções de Downsampling:\n1- 4:2:1\n2- 4:2:0")
         down = int(input("Opt: "));
         
    quality = int(input("Digite a fator de qualidade da matriz de quantizição\n"))
    
    cmRed = clr.LinearSegmentedColormap.from_list("Red", [(0,0,0), (1,0,0)], N=256)
    
    cmGreen = clr.LinearSegmentedColormap.from_list("Green", [(0,0,0), (0,1,0)], N=256)
    
    cmBlue = clr.LinearSegmentedColormap.from_list("Blue", [(0,0,0), (0,0,1)], N=256)

    cmGray = clr.LinearSegmentedColormap.from_list("Gray", [(0,0,0), (1,1,1)], N=256)

    YCbCr = np.array([[0.299,0.587,0.114],
             [-0.168736,-0.331264,0.5],
             [0.5,-0.418688,-0.081312]])
    
    Q_Y = np.array([[16,11,10,16,24,40,51,61],
                [12,12,14,19,26,58,60,55],
                [14,13,16,24,40,57,69,56],
                [14,17,22,29,51,87,80,62],
                [18,22,37,56,68,109,103,77],
                [24,35,55,64,81,104,113,92],
                [49,64,78,87,103,121,120,101],
                [72,92,95,98,112,100,103,99]])

    Q_CbCr = np.array([[17,18,24,47,99,99,99,99],
                       [18,21,26,66,99,99,99,99],
                       [24,26,56,99,99,99,99,99],
                       [47,66,99,99,99,99,99,99],
                       [99,99,99,99,99,99,99,99],
                       [99,99,99,99,99,99,99,99],
                       [99,99,99,99,99,99,99,99],
                       [99,99,99,99,99,99,99,99]])

    YCbCr_INV = np.linalg.inv(YCbCr)
   
    img = plt.imread(fName)

    R, G, B, img_YCbCr, Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a, Y_dct, Cb_dct, Cr_dct, Y_dct8, Cb_dct8, Cr_dct8, Y_dct64, Cb_dct64, Cr_dct64, Yb_Q, Cbb_Q, Crb_Q  = encoder(img, YCbCr, cmGray, down, Q_Y, Q_CbCr, quality)

    imgRec, YCbCr_rebuilt, Y_r, Cb_rebuilt, Cr_ebuilt  = decoder(R, G, B, img, img_YCbCr, YCbCr_INV, Y_d, Cb_d_l, Cr_d_l, Y_dct, Cb_dct, Cr_dct, cmGray)

    print("\n\nOpções de Operação:\n1- Imagem Original\n2- R, G, B\n3- YCbCr\n4- Y, Cb, Cr\n5- Downsampling\n6- Upsampling\n7- DCT nos canais completos\n8- DCT 8x8\n9- DCT 64x64\n10- Quantização dos coeficientes DCT\n11- Imagem Convertida\n0- Sair\n")
    opt = int(input("Opt: "));

    while(opt != 0):
        
        if opt == 1:
            showImg(img, "Imagem Original")
        elif opt == 2:
            showImg(R,"Codificação a Vermelho", cmRed)
            showImg(G,"Codificação a Verde", cmGreen)
            showImg(B,"Codificação a Azul", cmBlue)
        elif opt == 3:
            showImg(np.round(img_YCbCr).astype(np.uint8), "Imagem YCbCr")
        elif opt == 4:
            showImg(np.round(img_YCbCr[:,:,0]).astype(np.uint8), "Y", cmGray)
            showImg(np.round(img_YCbCr[:,:,1]).astype(np.uint8), "Cb", cmGray)
            showImg(np.round(img_YCbCr[:,:,2]).astype(np.uint8), "Cr", cmGray)
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
        elif opt == 7:
            plt.figure()
            plt.imshow(np.log(abs(Y_dct)+0.0001), cmGray)
            plt.axis('off')
            plt.title("Y_DCT")
            plt.show()
            
            plt.figure()
            plt.imshow(np.log(abs(Cb_dct)+0.0001), cmGray)
            plt.axis('off')
            plt.title("cb_DCT")
            plt.show()
            
            plt.figure()
            plt.imshow(np.log(abs(Cr_dct)+0.0001), cmGray)
            plt.axis('off')
            plt.title("cr_DCT")
            plt.show()
        elif opt == 8:
            plt.figure()
            plt.imshow(np.log(abs(Y_dct8)+0.0001), cmGray)
            plt.axis('off')
            plt.title("Y_DCT 8x8")
            plt.show()
            
            plt.figure()
            plt.imshow(np.log(abs(Cb_dct8)+0.0001), cmGray)
            plt.axis('off')
            plt.title("cb_DCT 8x8")
            plt.show()
            
            plt.figure()
            plt.imshow(np.log(abs(Cr_dct8)+0.0001), cmGray)
            plt.axis('off')
            plt.title("cr_DCT 8x8")
            plt.show()
        elif opt == 9:
            plt.figure()
            plt.imshow(np.log(abs(Y_dct64)+0.0001), cmGray)
            plt.axis('off')
            plt.title("Y_DCT 64x64")
            plt.show()
            
            plt.figure()
            plt.imshow(np.log(abs(Cb_dct64)+0.0001), cmGray)
            plt.axis('off')
            plt.title("cb_DCT 64x64")
            plt.show()
            
            plt.figure()
            plt.imshow(np.log(abs(Cr_dct64)+0.0001), cmGray)
            plt.axis('off')
            plt.title("cr_DCT 64x64")
            plt.show()
        elif opt == 10:
            plt.figure()
            plt.imshow(Yb_Q,cmGray)
            plt.axis('off')
            plt.title("Yb_Q")
            plt.show()
            
            plt.figure()
            plt.imshow(Cbb_Q, cmGray)
            plt.axis('off')
            plt.title("Cbb_Q")
            plt.show()

            plt.figure()
            plt.imshow(Crb_Q, cmGray)
            plt.axis('off')
            plt.title("Crb_Q")
            plt.show()
        elif opt == 11:
            showImg(imgRec, "Imagem Reconstruída")
         
        print("\n\nOpções de Operação:\n1- Imagem Original\n2- R, G, B\n3- YCbCr\n4- Y, Cb, Cr\n5- Downsampling\n6- Upsampling\n7- DCT nos canais completos\n8- DCT 8x8\n9- DCT 64x64\n10- Quantização dos coeficientes DCT\n11- Imagem Convertida\n0- Sair\n")            
        opt = int(input("Opt: "));
    
if __name__ == "__main__":
    main()
