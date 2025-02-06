import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def decoder():
    pass

def encoder():
    
    pass

def main():
    fName = "./imagens/airport.bmp"
    

    cmRed = clr.LinearSegmentedColormap.from_list("Red", [(0,0,0), (1,0,0)], N=256)
    
    cmGreen = clr.LinearSegmentedColormap.from_list("Green", [(0,0,0), (0,1,0)], N=256)
    
    cmBlue = clr.LinearSegmentedColormap.from_list("Blue", [(0,0,0), (0,0,1)], N=256)

    encoder()
    decoder()
    img = plt.imread(fName)

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    plt.figure()
    
    plt.imshow(B, cmBlue)
    plt.axis('off')
    plt.title(fName)
    plt.show()

    print(type(img))
    print(img.shape)
    print(img[0:8, 0:8, 0])
    print(img.dtype)


    


    encoder()
    decoder()









if __name__ == "__main__":
    main()