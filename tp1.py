
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
    if quality > 0 and quality < 50:
        return 50/quality
    elif quality <= 100:
        return (100-quality)/50
    else:
        return 0
            
    
def quantization(dct,q, quality):
   h, w = dct.shape
   sf = np.zeros((h,w))
   
   S = quality_coeficient(quality)
   
   
   if S == 0:
       return dct
   
   q_s = np.clip(np.round(q * S), 1, 255)

   
   for i in range(0, h, 8):
       for j in range(0, w, 8):
           sf[i:i+8,j:j+8] = np.round(dct[i:i+8, j:j+8] / q_s)
           
   return sf

def DPCM(dc):
    h, w = dc.shape
    diff = np.zeros(dc.shape)

    for i in range(h):
        for j in range(w):
            if (i % 8 == 0 and j % 8 == 0):
                if i == 0 and j == 0:
                    diff[i, j] = dc[i, j]
                elif i != 0 and j == 0:
                    diff[i, j] = dc[i, j] - dc[i - 8, w - 8]
                else:
                    diff[i, j] = dc[i, j] - dc[i, j - 8]
            else:  
                diff[i, j] = dc[i, j]
                
    return diff

def calc_erros(img, imgRec,y_diff):
    m, n, _ = img.shape
    mse = 1/(m*n)*np.sum((img-imgRec)**2)
    rmse = np.sqrt(mse)
    p = 1/(m*n)*np.sum(img**2)
    snr = 10*np.log(p/mse)
    psnr = 10*np.log((np.max(img)**2)/mse)
    max_dif = np.max(y_diff)
    avg_dif = np.average(y_diff)
    return mse, rmse, snr, psnr, max_dif, avg_dif

def encoder(img, YCbCr, cmRed, cmGreen, cmBlue ,cmGray, down, Q_Y, Q_CbCr, quality):
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

    Y_dct, Cb_dct, Cr_dct = DCT(Y_d, Cb_d_l, Cr_d_l)
    
    Y_dct8 = DCT_Blocks(Y_d, 8)
    Cb_dct8 = DCT_Blocks(Cb_d_l, 8)
    Cr_dct8 = DCT_Blocks(Cr_d_l, 8)
    
    Y_dct64 = DCT_Blocks(Y_d, 64)
    Cb_dct64 = DCT_Blocks(Cb_d_l, 64)
    Cr_dct64 = DCT_Blocks(Cr_d_l, 64)

    Yb_Q = quantization(Y_dct8,Q_Y, quality)
    Cbb_Q = quantization(Cb_dct8,Q_CbCr, quality)
    Crb_Q = quantization(Cr_dct8,Q_CbCr, quality)
    
    x, y = Yb_Q.shape
    x2, y2 = Cbb_Q.shape
    
    Yb_DPCM = DPCM(Yb_Q)
    Cbb_DPCM = DPCM(Cbb_Q)
    Crb_DPCM = DPCM(Crb_Q)
    
    return R, G, B, img_YCbCr, Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a, Y_dct, Cb_dct, Cr_dct, Y_dct8, Cb_dct8, Cr_dct8, Y_dct64, Cb_dct64, Cr_dct64, Yb_Q, Cbb_Q, Crb_Q, Yb_DPCM, Cbb_DPCM, Crb_DPCM


def removePadding(color, img):
    x, y, _ = img.shape
    color = color[:x, :y]
    return color
    
def YCbCr_to_RGB(img, YCbCr_inv):
    new_img = img.copy()
    new_img[:, :, 1:] -= 128
    new_img = new_img @ YCbCr_inv.T
    return np.clip(np.round(new_img), 0, 255).astype(np.uint8)

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

def DCT_Blocks_inv(img, block_size):
    h, w = img.shape
    dct = np.zeros((h,w))

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            dct_block = scipy.fft.idct(block, norm = "ortho").T
            dct_block = scipy.fft.idct(dct_block, norm = "ortho").T
            dct[i:i+block_size, j:j+block_size] = dct_block

    return dct

def reverse_quantization(dct,q, quality):
   h, w = dct.shape
   sf = np.zeros((h,w))
   
   S = quality_coeficient(quality)
   
   
   if S == 0:
       return dct

   q_s = np.clip(np.round(q * S), 1, 255)
   
   for i in range(0, h, 8):
       for j in range(0, w, 8):
           sf[i:i+8,j:j+8] = dct[i:i+8, j:j+8] * q_s
           
   return sf
    
    
def decoder(R, G, B, img, img_YCbCr,YCbCr_INV, Y_dct, Cb_dct, Cr_dct ,Yb_q, Cbb_q, Crb_q, Q_Y, Q_CbCr, quality, cmGray, cmRed, cmGreen, cmBlue):
    Y_dct = reverse_quantization(Yb_q, Q_Y, quality)
    Cb_dct = reverse_quantization(Cbb_q, Q_CbCr, quality)
    Cr_dct = reverse_quantization(Crb_q, Q_CbCr, quality)
     
    Y_d = DCT_Blocks_inv(Y_dct, 8)
    Cb_d = DCT_Blocks_inv(Cb_dct, 8)
    Cr_d = DCT_Blocks_inv(Cr_dct, 8)
     
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
         print("\nOpções de Downsampling:\n1- 4:2:2\n2- 4:2:0")
         down = int(input("Opt: "));
         
    quality = int(input("Digite a fator de qualidade da matriz de quantização (fator recomendado: 75)\n"))
    
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

    R, G, B, img_YCbCr, Y_d, Cb_d_l, Cr_d_l, Cb_d_c, Cr_d_c, Cb_d_a, Cr_d_a, Y_dct, Cb_dct, Cr_dct, Y_dct8, Cb_dct8, Cr_dct8, Y_dct64, Cb_dct64, Cr_dct64, Yb_Q, Cbb_Q, Crb_Q, Yb_DPCM, Cbb_DPCM, Crb_DPCM  = encoder(img, YCbCr, cmRed, cmGreen, cmBlue ,cmGray, down, Q_Y, Q_CbCr, quality)

    imgRec, YCbCr_rebuilt, Y_r, Cb_rebuilt, Cr_rebuilt  =  decoder(R, G, B, img, img_YCbCr, YCbCr_INV, Y_d, Cb_d_l, Cr_d_l, Yb_Q, Cbb_Q, Crb_Q, Q_Y, Q_CbCr, quality, cmGray, cmRed, cmGreen, cmBlue)

    y_diff = np.abs(Y_d - Y_r)

    mse,rmse,snr,psnr,max_dif, avg_dif = calc_erros(img,imgRec,y_diff)
    
    print(f"mse\n {mse}\nrmse \n {rmse}\nsnr\n{snr}\n psnr\n{psnr}\nmax_dif\n{max_dif}\navg_dif\n{avg_dif}")

    print("\n\nOpções de Operação:\n1- Imagem Original\n2- R, G, B\n3- YCbCr\n4- Y, Cb, Cr\n5- Downsampling\n6- Upsampling\n7- DCT nos canais completos\n8- DCT 8x8\n9- DCT 64x64\n10- Quantização dos coeficientes DCT\n11- Codificação dos coeficientes DC\n12- Imagem Convertida\n13- Diferenças e Erros\n0- Sair\n")
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
            showImg(np.log(abs(Y_dct)+0.0001), "Y_DCT", cmGray)
            showImg(np.log(abs(Cb_dct)+0.0001), "Cb_DCT", cmGray)
            showImg(np.log(abs(Cr_dct)+0.0001), "Cr_DCT", cmGray)
        elif opt == 8:
            showImg(np.log(abs(Y_dct8)+0.0001), "Y_DCT 8x8", cmGray)
            showImg(np.log(abs(Cb_dct8)+0.0001), "Cb_DCT 8x8", cmGray)
            showImg(np.log(abs(Cr_dct)+0.0001), "Cr_DCT 8x8", cmGray)
        elif opt == 9:
            showImg(np.log(abs(Y_dct64)+0.0001), "Y_DCT 64x64", cmGray)
            showImg(np.log(abs(Cb_dct64)+0.0001), "Cb_DCT 64x64", cmGray)
            showImg(np.log(abs(Cr_dct64)+0.0001), "Cr_DCT 64x64", cmGray)
        elif opt == 10:
            showImg(np.log(abs(Yb_Q)+0.0001), "Yb_Q", cmGray)
            showImg(np.log(abs(Cbb_Q)+0.0001), "Cbb_Q", cmGray)
            showImg(np.log(abs(Crb_Q)+0.0001), "Crb_Q", cmGray)
        elif opt == 11:
            showImg(np.log(abs(Yb_DPCM)+0.0001), "Yb_DPCM", cmGray)
            showImg(np.log(abs(Cbb_DPCM)+0.0001), "Cbb_DPCM", cmGray)
            showImg(np.log(abs(Crb_DPCM)+0.0001), "Crb_DPCM", cmGray)
        elif opt == 12:
            showImg(imgRec, "Imagem Reconstruída")
        elif opt == 13:
            showImg(y_diff, "Imagem diferenças", cmGray)
         
        print("\n\nOpções de Operação:\n1- Imagem Original\n2- R, G, B\n3- YCbCr\n4- Y, Cb, Cr\n5- Downsampling\n6- Upsampling\n7- DCT nos canais completos\n8- DCT 8x8\n9- DCT 64x64\n10- Quantização dos coeficientes DCT\n11- Codificação dos coeficientes DC\n12- Imagem Convertida\n13- Diferenças e Erros\n0- Sair\n")            
        opt = int(input("Opt: "));
        
    
if __name__ == "__main__":
    main()
