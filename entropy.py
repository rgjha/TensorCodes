# This code generates a random state of "N" qubits and 
# then computes the reduced density matrix of first "p" qubits. 
# Then it calculates the entanglement entropy. Note that 
# the entropy will be "p". 

import sys
import math
from math import *
import numpy as np 
from scipy import special
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
import time

N=24 
p=4   # Becomes expensive with increasing $p$ towards N/2

Psi = np.random.randn(2**N) 
#print ("Shape of Psi", np.shape(Psi)) # 2^24 # Generates this many complex coefficients 
Psi = Psi/LA.norm(Psi)

A = Psi.reshape(2**p, 2**(N-p))
Rho = np.dot(A, np.transpose(A).conj()) 

def comEE(Rho):
	u,v = LA.eig(Rho)
	chi = u.shape[0] 
	#print (u)    All elements of 'u' same and equal to 1/(2^p) 
	#print ("Shape of u", np.shape(u))  # 2^p 
	#print ("Shape of v", np.shape(v))  # 2^p x 2^p
	EE = 0
	for n in range (0 , chi):
		if u[n] > 0:
			EE += -u[n] * math.log(u[n],2)
	return EE 

if __name__ == "__main__":

	entropy = comEE(Rho)
	print ("Entanglement Entropy is", entropy)

# S_exact = -rho log2(rho) = -1/d * ln (1/d) summed 'd' times i.e. log2(d) = log2(2^p) = p 
