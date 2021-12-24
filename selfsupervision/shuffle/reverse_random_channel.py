import torch
import torch.nn as nn
import random


def create5images(images):
    # images : 4D tensor with shape [BT,C,H,W]
    T = 16
    B = images.size(0) // T
    C = images.size(1)
    H = images.size(2)
    W = images.size(3)
    image = torch.tensor([]).cuda()
    for b in range(B):
        bimage = images[b * T:(b + 1) * T, :, :, :].view(1, T, C, H, W)
        image = torch.cat([image, bimage], dim=0)
    return image  # [B,T,C,H,W]

def reverse_random_channels(image):
    #image [BT,C,H,W]
    '''4D image -> 5D image'''
    images = create5images(image) #[B,T,C,H,W]
    B,T,C,H,W = images.size() #4,16,2048,7,7
    #B_label = [random.randint(0,1) for _ in range(B)]

    result_ = torch.tensor([]).cuda()
    result = torch.tensor([]).cuda()
    for b in range(B):
        cha = random.randint(0,C-1) 

        images_B = images[b,:,cha,:,:].view(1,T,1,H,W)

        images_T = torch.tensor([]).cuda()
        for t in range(T): #[0,...,15]
            images_T = torch.cat([images_T,images_B[:,T-t-1,:,:,:].view(1,1,H,W)],dim=0)
        images_T = images_T.view(1,T,1,H,W)
        #print(images_T.shape) #[1,16,1,7,7]
        result_ = torch.cat([result_,images_T])
    
    result = torch.cat([result_[0],result_[1],result_[2],result_[3]],dim=0)
    
    labels = torch.ones(64).long().cuda()

    return result,labels
        

        
        

    

