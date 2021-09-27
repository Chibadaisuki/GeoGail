from math import *
import numpy
import matplotlib.pyplot as plt
import numpy as np
def matrix_factorization(R,P,Q,K,steps=1000,alpha=0.01,beta=0.02): 
    Q=Q.T                
    result=[]
    for step in range(steps): 
        for i in range(len(R)):
            for j in range(len(R[i])):
                    eij=R[i][j]-numpy.dot(P[i,:],Q[:,j])     
                    for k in range(K):
                      if R[i][j]>0:        
                        P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])   
                        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j]) 
        eR=numpy.dot(P,Q)  
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
              if R[i][j]>0:
                    e=e+pow(R[i][j]-numpy.dot(P[i,:],Q[:,j]),2)      
                    for k in range(K):
                        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2)) 
        result.append(e)
        if e<0.001:          
            break
    return P,Q.T,result

if __name__ == '__main__':   
    R=np.load('train_add.npy')
    R=numpy.array(R)
    N=len(R)    
    M=len(R[0]) 
    K=30    
    P=numpy.random.rand(N,K) 
    Q=numpy.random.rand(M,K) 
    nP,nQ,result=matrix_factorization(R,P,Q,K)
    print(R)         
    R_MF=numpy.dot(nP,nQ.T)
    np.save('train_add_result.npy',R_MF)
