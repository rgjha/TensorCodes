# This code generates a random state of "N" qubits and 
# then computes the reduced density matrix of first "p" qubits. 
# Then it calculates the entanglement entropy. Note that 
# the maximum entropy will be "p". 

import sys
import math
from math import sqrt
import numpy as np 
from scipy import special
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
import time

N=24 
p=7

Psi = np.random.randn(2**N) 
Psi = Psi/LA.norm(Psi)

A = Psi.reshape(2**p, 2**(N-p))
Rho = np.dot(A, np.transpose(A).conj()) 

def comEE(Rho):
	u,v = LA.eig(Rho)
	chi = u.shape[0] 
	EE = 0
	for n in range (0 , chi):
		if u[n] > 0:
			EE += -u[n] * math.log2(u[n])
	return EE 

if __name__ == "__main__":

	entropy = comEE(Rho)
	print ("Entanglement Entropy is", entropy)



