import torch
import random

def create5images(images):
    # images : 4D tensor with shape [BT,C,H,W]
    T = 16
    B = images.size(0)//T
    C = images.size(1)
    H = images.size(2)
    W = images.size(3)
    image = torch.tensor([]).cuda()
    for b in range(B):
        bimage = images[b*T:(b+1)*T,:,:,:].view(1,T,C,H,W)
        image = torch.cat([image,bimage],dim = 0)
    return image #[B,T,C,H,W]

def randomly_create2frames_labels():
    label = torch.IntTensor(random.sample(range(0,16),2))
    if label[0].item() > label[1].item():
        label = torch.cat([label[1].view(1,1),label[0].view(1,1)],dim=0).view(2)
    return label #[2]

def create_shuffle_images(images,label):
    #images:[T,C,H,W]
    T,C,H,W = images.size()
    label1 = label[0].item() #3
    label2 = label[1].item() #5
    shuffle_imageT = torch.tensor([]).cuda()
    for t in range(T):
        if t!=label1 and t!= label2 :
            shuffle_imageT = torch.cat([shuffle_imageT,images[t].view(1,C,H,W)],dim = 0)
        elif t==label1:
            shuffle_imageT = torch.cat([shuffle_imageT,images[label2].view(1,C,H,W)],dim = 0)
        elif t == label2:
            shuffle_imageT = torch.cat([shuffle_imageT,images[label1].view(1,C,H,W)],dim = 0)
    
    shuffle_imageT = shuffle_imageT
    return shuffle_imageT # shuffled image with shape [T,C,H,W]

#if T = 16
def create_shuffled_label(T,label):#[2]
    shuffled_label = torch.tensor([])
    label1 = label[0].item()
    label2 = label[1].item()

    l = [0 for i in range(T)]
    l[label1] = 1
    l[label2] = 1
    shuffled_label = torch.tensor(l).cuda()
    return shuffled_label #[16]

def create_shuffle_images_with_batch(images):
    #images : 4D [BT,C,H,W]
    image = create5images(images).cuda() #[BT,C,H,W] -> [B,T,C,H,W]
    B,T,C,H,W = image.size()
    shuffled_images = torch.tensor([]).cuda()
    shuffled_labels = torch.tensor([]).cuda()
    for b in range(B):
        image_T = image[b] 
        label = randomly_create2frames_labels()
        shuffled_label = create_shuffled_label(T,label).float()
        image_T = create_shuffle_images(image_T,label)
        #[BT,C,H,W]
        shuffled_images = torch.cat([shuffled_images,image_T],dim = 0)
        shuffled_labels = torch.cat([shuffled_labels,shuffled_label])
    shuffled_labels = shuffled_labels.long()
    return shuffled_images,shuffled_labels

