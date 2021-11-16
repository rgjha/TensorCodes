# In progress
# 09 Jan 2021
# Not tested

import sys
import math
from math import sqrt
import numpy as np  
from scipy import special
from numpy import linalg as LA
from itertools import product
import scipy.linalg as scl
import scipy as sp 
from scipy.linalg import sqrtm
import time
import datetime 
from matplotlib import pyplot as plt
from opt_einsum import contract
from ncon import ncon 

kmax = 1 
Dtheta = 3
Niters = 20 
beta = 0.2
Dbeta = int(1 - (2+kmax)*pow(N, kmax+1) + (1+kmax)*pow(N,kmax+2))/(pow((1-N),2))
Dtot = int(Dbeta*Dtheta)
Aten = np.zeros([int(Dtot)])
Dcut = Dtot
PI = 3.141592653589793



startTime = time.time()
print ("STARTED:" , datetime.datetime.now().strftime("%d %B %Y %H:%M:%S")) 

def tensorsvd(input,left,right,D):
    '''Reshape an input tensor into a rectangular matrix with first index corresponding
    to left set of indices and second index corresponding to right set of indices. Do SVD
    and then reshape U and V to tensors [left,D] x [D,right]  
    '''
    T = input 
    left_index_list = []
    for i in range(len(left)):
        left_index_list.append(T.shape[i])
    xsize = np.prod(left_index_list) 
    right_index_list = []
    for i in range(len(left),len(left)+len(right)):
        right_index_list.append(T.shape[i])
    ysize = np.prod(right_index_list)
    T = np.reshape(T,(xsize,ysize))
    
    U, s, V = np.linalg.svd(T,full_matrices = False)
    
    #'''
    if D < len(s):
        s = np.diag(s[:D])
        U = U[:,:D]
        V = V[:D,:]
    else:
        D = len(s)
        s = np.diag(s)
    #'''


    U = np.reshape(U,left_index_list+[D])
    V = np.reshape(V,[D]+right_index_list)   
    return U, s, V


def Jtensor(index,theta): 

    #'''
    if index != 0 and theta != 0:
        return 2.0 * np.sin((theta + (2*PI*index))*0.50)/(theta + (2*PI*index)) 

    if index == 0 and theta != 0:
        return np.sin((theta/2.))/(theta/2.)

    if index != 0 and theta == 0:
        return 0.

    if index == 0 and theta == 0:
        return 1.
    #'''

    return 2.0 * np.sin((theta + (2*PI*index))*0.50)/(theta + (2*PI*index)) 


def get_tuple(length, total):

    return filter(lambda x:sum(x)==total,product(range(total+1),repeat=length))


def get_index(in1, in2, in3, in4):

    return int(in4*Dbeta + (2*in1) + in3) 

 
def Zconstruct(beta, theta):

    kc = 0
    count = 0
    lsrange = []
    msrange = [] 
    sthetarange = [] 
    while kc <= kmax:
        for ls, ms in list(get_tuple(2, kc)):
            lsrange.append(ls)
            msrange.append(ms)
        kc += 1 

    for k in range (Dtheta):
        sthetarange.append(k)


    for stheta in sthetarange: 
        for ls, ms in zip(lsrange, msrange):

            st = 0 if ls == ms == 0 else 1

            for brac_a in range (st, pow(N, ls+ms)+st):
                index = get_index(ls, ms, brac_a, stheta)
                #print (ls, ms, brac_a, stheta, index)
                #print (index) 
                Aten[index] = np.sqrt(special.iv(ls+ms+N-1,2*N*beta)) * Jtensor(stheta, theta)

            if index > Dtot:
                print ("WARNING: Index count not correct")
                sys.exit(1)

     
    #print ("Norm A", LA.norm(Aten))
    #print (np.shape(Aten)) 
    out = contract("a,b,c,d -> abcd", Aten, Aten, Aten, Aten)
    #print ("OUT_0123", out[5][6][7][12])

    a = np.sqrt(special.iv(1,2*N*beta))*Jtensor(1, theta)
    b = np.sqrt(special.iv(2,2*N*beta))*Jtensor(1, theta)
    c = np.sqrt(special.iv(2,2*N*beta))*Jtensor(1, theta)
    d = np.sqrt(special.iv(2,2*N*beta))*Jtensor(2, theta)

    #print ("Alt", a*b*c*d) 
    #print ("NNZ", np.count_nonzero(out))

    '''
    for i  in range (count):
        for j  in range (count):
            for k  in range (count):
                for l  in range (count):

                    index = i+j-k-l
                    if index != 0:

                        out[i][j][k][l] = 0.0
    '''

    return out 


def CG_step(matrix):

    T = matrix  
    #AAdag = ncon([T,T,T,T],[[-2,1,2,5],[-1,5,3,4],[-4,1,2,6],[-3,6,3,4]])
    AAdag = contract("jabe, iecd, labf, kfcd  -> ijkl", T, T, T, T)
    U, s, V = tensorsvd(AAdag,[0,1],[2,3],Dcut) 
    #A = ncon([U,T,T,U],[[1,2,-1],[2,-2,4,3],[1,3,5,-4],[5,4,-3]])
    A = contract("abi, bjdc, acel, edk  -> ijkl", U, T, T, U)

    #AAAAdag = ncon([A,A,A,A],[[1,-1,2,3],[2,-2,4,5],[1,-3,6,3],[6,-4,4,5]])
    AAAAdag = contract("aibc, bjde, akfc, flde  -> ijkl", A, A, A, A)
    U, s, V = tensorsvd(AAAAdag,[0,1],[2,3],Dcut)  
    #AA = ncon([U,A,A,U],[[1,2,-2],[-1,1,3,4],[3,2,-3,5],[4,5,-4]])
    AA = contract("abj, iacd, cbke, del  -> ijkl", U, A, A, U)

    maxAA = np.max(AA)
    AA = AA/maxAA # Normalize by the largest element of the tensor
        
    return AA, maxAA


if __name__ == "__main__":

    theta = np.linspace(0.05*np.pi, 1.5*np.pi, 3)
    xaxis = [x/np.pi for x in theta]
    Nsteps = int(np.shape(theta)[0])
    f = np.zeros([Nsteps])

    #theta = [0.0] 

    #print ("VAL", Jtensor(1, 2*math.pi))
    #print ("VAL", Jtensor(2, 2*math.pi))
    #print ("VAL", Jtensor(3, 2*math.pi))


    for p in range (0, Nsteps):

      T = Zconstruct(beta, theta[p])
      norm = LA.norm(T)
      #print ("N", norm)
      T /= norm 
      #Z = ncon([T,T,T,T],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
      Z = contract("geca, cfgb, hade, dbhf" , T, T, T, T)

      C = np.log(norm)

      for i in range (Niters):

          T, norm = CG_step(T)
          C += np.log(norm)/(4**(i+1)) 
          if i == Niters-1:

            #Z1 = ncon([T,T],[[1,-1,2,-2],[2,-3,1,-4]])
            Z1 = contract("aibj, bkal -> ijkl" , T, T)
            #Z = ncon([Z1,Z1],[[1,2,3,4],[2,1,4,3]])
            Z = contract("abcd, badc" , Z1, Z1)
            lnZ = (C + (np.log(Z)/(4**(Niters)))) 
            f[p] = -lnZ 
            print (round(theta[p],10), round(abs(f[p]),10))


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    dx = theta[1]-theta[0] # Assuming equal spacing ...

    dfdx = np.gradient(f, dx)
    d2fdx2 = np.gradient(dfdx, dx)
    out = []
    for i in range(0, len(dfdx)): 
        out.append(dfdx[i]*-1.) 

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(r'$\frac{\theta}{\pi}$',fontsize=14)
    ax1.set_ylabel('-log(Z)', fontsize=14)

    #if beta != 0.:
    #    ax1.plot(xaxis, out, marker="o", color=color) 
    #else:
    ax1.plot(xaxis, f, marker="*", color=color) 

    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('', color=color,fontsize=13) 

    with open('data_2d.txt', 'w') as file:
        res = "\n".join("{} {}".format(x, y) for x, y in zip(xaxis, out)) 
        file.write("%s\n" % (res))
        #print ("%s\n" % (res))


    #fu = -np.log((2.0/theta)*np.sin(theta/2.))  
    #print ([x/y for x, y in zip(f, fu)])

    plt.title(r"2d model",fontsize=16, color='black')
    #plt.show()
    print ("COMPLETED:" , datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"))
