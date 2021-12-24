import torch
import random


def create5Dimages(images):
    # images : 4D tensor with shape [BT,C,H,W]
    T = 16
    B = images.size(0) // T
    C = images.size(1)
    H = images.size(2)
    W = images.size(3)
    image = torch.tensor([]).cuda()
    for b in range(B):
        bimage = images[b * T:(b + 1) * T, :, :, :].float().view(1, T, C, H, W).cuda()
        image = torch.cat([image, bimage], dim=0)
    return image  # [B,T,C,H,W]

def create4Dimages(images):
    # images : 5D tensor with shape [B,T,C,H,W]
    B,T,C,H,W = images.size()
    image = torch.tensor([]).cuda()
    for b in range(B):
        image = torch.cat([image,images[b]],dim=0)
    return image


def mirror2DZong(image):
    #image[H,L]
    image = image.float().cuda()
    result = torch.Tensor().cuda()
    H,L = image.size()
    for i in range(L):
        #print(image[:,L-i-1])
        result = torch.cat([result,image[:,L-i-1]],0)
    return result.view(L,H).t() 


def mirror2DHeng(image):
    #image(H,L)
    image = image.float().cuda()
    result = torch.Tensor().cuda()
    H,L = image.size()
    for j in range(H):
        #print(image[H-j-1,:])
        result = torch.cat([result,image[H-j-1,:]],0)
    return result.view(H,L)

def mirror3DZong(image):
    #image[C,H,L]
    image = image.float().cuda()
    result = torch.Tensor().cuda()
    C,H,L = image.size()
    for i in range(C):
        result = torch.cat([result,mirror2DZong(image[i,:,:])],0)
    return result.view(C,H,L)

def mirror3DHeng(image):
    #image[C,H,L]
    image = image.float().cuda()
    result = torch.Tensor().cuda()
    C,H,L = image.size()
    for j in range(C):
        result = torch.cat([result,mirror2DHeng(image[j,:,:])],0)
    return result.view(C,H,L)

def mirror(images,labels):
    #images[T,C,H,W]
    #labels[T]

    T,C,H,W = images.size()
    result = torch.Tensor().cuda()
    for l in range(len(labels)):
        if(labels[l]==0):
            image = mirror3DZong(images[l]).view(1,C,H,W)
            result = torch.cat([result,image],0)
        elif(labels[l]==1):
            image = mirror3DHeng(images[l]).view(1,C,H,W)
            result = torch.cat([result,image],0)
    return result 
        
def getLabel():
    T = 16
    for t in range(T):

        label_T = [random.randint(0,1) for _ in range(T)]
    return torch.tensor(label_T).float().cuda()


def Mirror_Self_Supervision(images):
    #images [BT,C,H,W]
    Bt,C,H,W = images.size()
    T = 16
    B = Bt//T
    label_domain = (0,1)

    image = create5Dimages(images) #[B,T,C,H,W]
    mirrorImage = torch.Tensor().cuda()
    mirrorLabel = torch.Tensor().cuda()
    for b in range(B):
        label_T = getLabel() #[T]
        mirror_image_T = mirror(image[b],label_T)  # image[b]:[T,C,H,W]   label_T:[T]
        mirrorLabel = torch.cat([mirrorLabel,label_T],0)
        mirrorImage = torch.cat([mirrorImage,mirror_image_T.view(1,T,C,H,W)],0)
    #5D->4D
    mirrorImage = create4Dimages(mirrorImage)
    return mirrorImage,mirrorLabel
    






    



