import numpy 
import numpy as np
import torch

def arraytostring(array):
    #array[51,51]
    result = '['
    x = array.shape[0]
    y = array.shape[1]
    for i in range(x):
        result = result+'['
        for j in range(y):
            if(j==0):
                result = result+str(array[i][j])
            else:
                result = result+','+str(array[i][j])
        if(i==x-1):
            result = result+']'
        else:
            result = result+'],'
    result = result+']'
    return result


'''array[64,2048,7,7]'''
def arraytostringwithBT(array):
    result = '['
    #array [BT,C,H,W]
    Bt = array.shape[0]
    C = array.shape[1]
    H = array.shape[2]
    W = array.shape[3]

    for bt in range(Bt):
        result = result+'['
        for c in range(C):
            result = result+arraytostring(array[bt,c,:,:])
            if(c!=C-1):
                result = result+','
        if(bt!=Bt-1):
            result = result+'],'
        else:
            result = result+']'
    result = result+']'
    return result

def indices2string(indices):
    length = indices.shape[0]
    result = '['
    for i in range(length):
        result = result + str(indices[i])
        if(i!=length-1):
            result = result + ','
        else:
            result = result + ']'
    return result 
              



