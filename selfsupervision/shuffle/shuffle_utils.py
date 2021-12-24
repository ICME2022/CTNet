import torch
import torch.nn as nn
import random
import numpy as np
'''
images = torch.arange(1,64*1*3*3+1).view(64,1,3,3)
#print(images)
labels1 = torch.IntTensor(random.sample(range(0,16),16))
labels2 = torch.IntTensor(random.sample(range(0,16),16))
labels3 = torch.IntTensor(random.sample(range(0,16),16))
labels4 = torch.IntTensor(random.sample(range(0,16),16))
labels = torch.cat([labels1,labels2,labels3,labels4],dim = 0).view(4,16)
#print(images.shape)
#print(labels)
'''
def create_shuffle_labels(images):
    T = 16
    B = images.size(0)//T
    labels = torch.IntTensor([])
    for i in range(B):
        label = torch.IntTensor(random.sample(range(0,T),T))
        labels = torch.cat([labels,label],dim = 0)
    return labels.view(4,16)

labels = create_shuffle_labels(images)

def randomly_shuffle(images,labels):
    #frame = 16
    T = 16
    B = images.size(0)//T
    C = images.size(1)
    H = images.size(2)
    W = images.size(3)
    x = torch.tensor([])
    #[BT,C,H,W] -> [B,T,C,H,W]
    for b in range(B):
        k = images[T*b:T*(b+1),:,:,:].view(1,T,C,H,W)
        x = torch.cat([x,k],dim = 0)
    #print(x.shape)[4,16,1,3,3]
    #shuffle
    result_T = torch.Tensor([])
    shuffle_images = torch.Tensor([])
    for b in range(B):
        result_T = torch.Tensor([])
        image = x[b] #[16,1,3,3]
        #get B label , and shape of label is [16]
        label = labels[b]
        #print(label)
        for t in range(T):
            result_T = torch.cat([result_T,image[label[t]]],dim = 0)
        shuffle_images = torch.cat([shuffle_images,result_T],dim = 0)
    
    return shuffle_images,labels.view(-1)

ima,lab = randomly_shuffle(images,labels)  
print(lab)
print(ima)  


    

        

