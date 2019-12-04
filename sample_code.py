#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# A guide to code magnetization. 

import numpy as np
import scipy as sc
import scipy.linalg as scl
from tensorfunc import tensorsvd, tensoreig
import matplotlib.pyplot as plt
from dncon import dncon as ncon
import scipy.integrate as integrate

D = 2
h = 1
hs = 1.0
Ds=2
temp = 2.0


def compute_free_energy_hs(Ds,hs,temp):

    results = {}
    beta = 1/temp 
    results[D] = []
    print ('h = %s' %(h))
    a = isingtpf2d(beta,h)
    converg_criteria = 0.0001   
            
    i = 0
    Z = ncon([a,a,a,a],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
    f_i = -temp*(np.log(Z))/(4)
    C = 0
    N = 1
    
    
    a, s, maxAA = coarse_graining_step_2d(a,None,2)
    C = np.log(maxAA)+4*C
    N *= 4.
    if i >= 10:
        Z = ncon([a,a,a,a],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
        f = -temp*(np.log(Z)+4*C)/(4*N)
        delta_f = np.abs((f - f_i)/f)

        if delta_f <= converg_criteria:
            print (i)
            break
        else:
            f_i = f
            i += 1
   
        Z = ncon([a,a,a,a],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
        f = -temp*(np.log(Z)+4*C)/(4*N)
            
        results[D].append(f)
            
    return results
    


#to compute Ising 2D tensor partition function
def isingtpf2d(beta,h):
    H_local = np.array([[-1.-h/2,1.],[1.,-1.+h/2]]) #h is intended to be a small magnetic field to break symmetry
    print (-beta*H_local)
    Q = np.exp(-beta*H_local)
    
    delta = np.zeros((2,2,2,2))
    delta[0,0,0,0] = 1.
    delta[1,1,1,1] = 1.

    Qsr = scl.sqrtm(Q)

    a = ncon([delta,Qsr,Qsr,Qsr,Qsr],[[1,2,3,4],[-1,1],[-2,2],[3,-3],[4,-4]])
    
    return a

#To compute the tensor that is used to measure one-site magnetization
def isings2d(beta,h):
    H_local = np.array([[-1.-h/2,1.],[1.,-1.+h/2]])
    Q = np.exp(-beta*H_local)
    
    g = np.zeros((2,2,2,2))
    g[0,0,0,0] = 1.
    g[1,1,1,1] = -1.

    Qsr = scl.sqrtm(Q)

    b = ncon([g,Qsr,Qsr,Qsr,Qsr],[[1,2,3,4],[-1,1],[-2,2],[3,-3],[4,-4]])
    
    return b


    


#coarse-grain a sub-network with the form
# b - a
# a - a
def coarse_graining_step_2d(a,b=None,D='infinity'):
    
    A = ncon([a,a],[[-2,-3,-4,1],[-1,1,-5,-6]])
    U, s, V = tensorsvd(A,[0,1],[2,3,4,5],D) 
    A = ncon([U,A,U],[[1,2,-1],[1,2,-2,3,4,-4],[4,3,-3]])

    if b != None:
        B = ncon([b,a],[[-2,-3,-4,1],[-1,1,-5,-6]])    
        B = ncon([U,B,U],[[1,2,-1],[1,2,-2,3,4,-4],[4,3,-3]])
    
    AA = ncon([A,A],[[-1,-2,1,-6],[1,-3,-4,-5]])
    U, s, V = tensorsvd(AA,[1,2],[0,3,4,5],D)  
    AA = ncon([U,AA,U],[[1,2,-2],[-1,1,2,-3,4,3],[3,4,-4]])  
    if b != None:
        BA = ncon([B,A],[[-1,-2,1,-6],[1,-3,-4,-5]])
        BA = ncon([U,BA,U],[[1,2,-2],[-1,1,2,-3,4,3],[3,4,-4]])  
    
    maxAA = np.max(AA)
     
    AA = AA/maxAA #divides over largest value in the tensor
    if b != None:
        BA = BA/maxAA
    else:
        BA = AA
    
        
    return AA, BA, maxAA


    
def final_contraction(dim,a,b=None):

    ap = ncon([a,a,a,a],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
    if b != None:
        bp = ncon([b,a,a,a],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
    if b == None:
        bp = None
    return ap, bp
    


def compute_magnetization_hs(dim,Ds,hs,temp):
    results = {}
    beta = 1/temp
    for D in Ds:
        results[D] = []
        for h in hs:
            print('h: %s' %(h))
            a = isingtpf2d(beta,h)
            b = isings2d(beta,h)
            cconverg_criteria = 10**(-6)
            
            i = 0
            r_i = 2
            while True:
                a, b, maxAA = coarse_graining_step_2d(a,b=b,D=D)
                if i >= 10:
                    ap, bp = final_contraction(dim,a,b)
            
                    r = (bp/ap)
                    delta_r = np.abs((r - r_i)/r)
                    if delta_r <= converg_criteria:
                        print(i)
                        break
                    else:
                        r_i = r
                i += 1
    
            ap, bp = final_contraction(dim,a,b)
            
            results[D].append(bp/ap)
            
    return results



def write_results(results,xaxis,docu_name):
	f = open('./%s.csv' %(docu_name),'w')
	for x in xaxis:
		f.write('%s,' %(x))
	f.write('/n')
	for D in results:
		f.write(str(D))
		f.write('/n')
		for y in results[D]:
			f.write('%s,' %(y))
		f.write('/n')
	f.close()
	



if __name__ == "__main__":

    results_f = compute_free_energy_hs(D,hs,temp)
    magnet_results = compute_magnetization_hs(D,hs,temp)


