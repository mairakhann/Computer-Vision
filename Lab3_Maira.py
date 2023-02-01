import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
from numpy.linalg import inv, det
import scipy 

def kmeans_segm(I, K, L, seed = 42):
    np.random.seed(seed)
    Image=np.reshape(I,(-1,3))
    seg=Image
    random_centroid=np.random.randint(low=0,high=np.unique(Image,axis=0).shape[0],size=K)
    centroid=np.unique(Image,axis=0)[random_centroid,:]
    prev_centroid=np.zeros(np.shape(centroid))
    for i in range(L):
        Distance=distance_matrix(Image,centroid)
        seg=np.argmin(Distance,axis=1)
        
        for j in range(K):
            prev_centroid[j,:]=centroid[j,:]
            points=np.reshape(np.nonzero(seg==j),-1)
            centroid[j,:]=np.mean(Image[points])
        
        if np.max(np.max(abs(prev_centroid-centroid))) < 1e-6:
            print("Converges at:",i)
            break
    
    if (len(np.shape(I)))==3:
        seg=np.reshape(seg,(np.shape(I)[0],np.shape(I)[1]))
    else:
        seg=np.reshape(seg,np.shape(I)[0])

    return seg, centroid


def mixture_prob(img, K, L, mask):
    img=img/255
    I=np.reshape(img,(-1,3)).astype(np.float32)
    Mask_ones=I[np.reshape(np.where(np.reshape(mask,-1)==1),-1)]
    seg,centers=kmeans_segm(Mask_ones,K,L)
    
    cov=[np.eye(3)*0.1 for i in range(K)]
    w=np.zeros(K)
    for i in range(K):
        w[i]=len(np.nonzero(seg==i))/Mask_ones.shape[0]
    
    prob=np.zeros((I.shape[0],K))
    p=np.zeros((Mask_ones.shape[0],K))
    g=np.zeros((Mask_ones.shape[0],K))
    
    for l in range(L):
        for k in range(K):
            mean=centers[k].ravel()
            print("L==",l,"K==",k)
            print(det(cov[k]))
            if det(cov[k]) < 1e-20:
                break
            
            g[:,k]=w[l]*scipy.stats.multivariate_normal.pdf(Mask_ones,mean, cov[k])
        
        for k in range(K):
            if(np.sum(g,axis=1).any()!=0):
                p[:,k]=g[:,k]/np.sum(g,axis=1)
            
        for k in range(K):
            w[k]=np.mean(p[:,k])
            centers[k,:]=np.dot(np.transpose(p[:,k]),Mask_ones)/np.sum(p[:,k])
            diff=Mask_ones-centers[k,:]
            qq=np.reshape(p[:,k],(-1,1))
            cov[k]=np.dot(np.transpose(diff),diff*qq)/np.sum(p[:,k])
    
    for k in range(K):
        mean=centers[k].ravel()
        prob[:,k]=w[k]*scipy.stats.multivariate_normal.pdf(I,mean, cov[k])
        prob[:,k]=prob[:,k]/np.sum(prob[:,k])

    prob=np.sum(prob,axis=1)
    prob=np.reshape(prob,(np.shape(img)[0],np.shape(img)[1]))   
  
    return  prob        
