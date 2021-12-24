import torch



def TmirrorZong(image):
    #image[T,C,H,W] W->L
    T,C,H,L = image.size()
    index = list(range(0,L,1))
    index.reverse()
    return  image[:,:,:,index]

def TmirrorHeng(image):
    #image[T,C,H,W] 
    T,C,H,W = image.size()
    index = list(range(0,H,1))
    index.reverse()
    return image[:,:,index]

def TmirrorZong3D(image):
    #image[C,H,W] W->L
    C,H,L = image.size()
    index = list(range(0,L,1))
    index.reverse()
    return  image[:,:,index]

def TmirrorHeng3D(image):
    #image[C,H,W] 
    C,H,W = image.size()
    index = list(range(0,H,1))
    index.reverse()
    return image[:,index]

a = torch.arange(0,3*3*3).view(3,3,3)
print(TmirrorZong3D(a))


