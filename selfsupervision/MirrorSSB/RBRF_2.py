import torch
import random

def create5Dimages(images):#[BT,C,H,W]->[B,T,C,H,W]
    # images : 4D tensor with shape [BT,C,H,W]
    T = 8
    B = images.size(0) // T
    C = images.size(1)
    H = images.size(2)
    W = images.size(3)
    image = torch.tensor([]).cuda()
    for b in range(B):
        bimage = images[b * T:(b + 1) * T, :, :, :].float().view(1, T, C, H, W).cuda()
        image = torch.cat([image, bimage], dim=0)
    return image  # [B,T,C,H,W]


def create4Dimages(images):#[B,T,C,H,W]->[BT,C,H,W]
    # images : 5D tensor with shape [B,T,C,H,W]
    B, T, C, H, W = images.size()
    image = torch.tensor([]).cuda()
    for b in range(B):
        image = torch.cat([image, images[b]], dim=0)
    return image

def TmirrorZong(image):
    # image[C,H,W] W->L
    C, H, L = image.size()
    index = list(range(0, L, 1))
    index.reverse()
    return image[:, :, index]


def TmirrorHeng(image):
    # image[C,H,W]
    C, H, W = image.size()
    index = list(range(0, H, 1))
    index.reverse()
    return image[:, index]



def RBRF(images):#image->[BT,C,H,W]
    image = create5Dimages(images)#image->[B,T,C,H,W]
    B,T,C,H,W = image.size()

    labelb = [random.randint(0, 1) for _ in range(B)]
    label_B = torch.tensor(labelb).cuda()

    labels = torch.tensor([]).long().cuda()

    mirrorImage = torch.tensor([]).cuda()
    
    
    for l in range(B):

        label_1_t = torch.tensor([]).long().cuda()
        if(label_B[l]==0):
            labels = torch.cat([labels,torch.zeros(8).long().cuda()])
            
            mirrorImage = torch.cat([mirrorImage,image[l].view(1,T,C,H,W)],dim=0) #[1,T,C,H,W]

        elif(label_B[l]==1):
            result_B = torch.tensor([]).cuda()
            labelt_2 = [random.randint(0, 1) for _ in range(T)] 
            label_t_2 = torch.tensor(labelt_2).cuda()

            for t in range(T):
                if(label_t_2[t]==0):
                    labelt_3 = [random.randint(0, 1) for _ in range(1)]
                    label_t_3 = torch.tensor(labelt_3).cuda()
                    if(label_t_3[0]==0):
                        #Z
                        ll = [1]
                        l_l = torch.tensor(ll).cuda() #list->tensor
                        label_1_t = torch.cat([label_1_t,l_l])
                        result_t = TmirrorZong(image[l][t]) #image[l][t]->[C,H,W]
                        result_B = torch.cat([result_B,result_t.view(1,C,H,W).cuda()])

                    elif(label_t_3[0]==1):
                        #H
                        ll = [1]
                        l_l = torch.tensor(ll).cuda()
                        label_1_t = torch.cat([label_1_t,l_l])
                        result_t = TmirrorHeng(image[l][t]) #image[l][t]->[C,H,W]
                        result_B = torch.cat([result_B,result_t.view(1,C,H,W).cuda()])
                elif(label_t_2[t]==1):
                    labelt_3 = [random.randint(0, 1) for _ in range(1)]
                    label_t_3 = torch.tensor(labelt_3).cuda()
                    if(label_t_3[0]==0):
                        #Z
                        ll = [1]
                        l_l = torch.tensor(ll).cuda()
                        label_1_t = torch.cat([label_1_t,l_l])
                        tt = TmirrorZong(image[l][t]).view(1,C,H,W)
                        result_t = torch.tensor([]).cuda()
                        index = list(range(0, C, 1))
                        index.reverse()
                        result_t = tt[:, index]
                        result_B = torch.cat([result_B,result_t.view(1,C,H,W).cuda()])
                    elif(label_t_3[0]==1):
                        #Z
                        ll = [1]
                        l_l = torch.tensor(ll).cuda()
                        label_1_t = torch.cat([label_1_t,l_l])
                        tt = TmirrorHeng(image[l][t]).view(1,C,H,W)
                        result_t = torch.tensor([]).cuda()
                        index = list(range(0, C, 1))
                        index.reverse()
                        result_t = tt[:, index]
                        result_B = torch.cat([result_B,result_t.view(1,C,H,W).cuda()])
            mirrorImage = torch.cat([mirrorImage, result_B.view(1, T, C, H, W)],dim=0)

        labels = torch.cat([labels,label_1_t])#[BT]

        

    mirrorImage = create4Dimages(mirrorImage) #[BT,C,H,W]

    return mirrorImage,labels
