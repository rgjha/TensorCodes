# A simple code to check strong sub-additivity (SSA) of 
# entanglement entropy inspired by the exercise given
# by J.Maldacena at Mathematica Summer School on Theoretical 
# Physics (2016) which is available at: http://msstp.org/?q=node/298

# 23/3/2019


import sys
import math
import random
from math import sqrt
import numpy as np
import scipy as sp                 
from scipy import special
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
import time
import datetime 
from numpy.linalg import matrix_power
from scipy.stats import ortho_group
from numpy.linalg import multi_dot
from random import randrange
from math import log
from scipy.linalg import logm, expm


if len(sys.argv) < 4:
  print("Usage:", str(sys.argv[0]), "dimA " "dimB" "dimC" )
  sys.exit(1)

dimA = int(sys.argv[1])
dimB = int(sys.argv[2])
dimC = int(sys.argv[3])
# H = HA ⊗ HB ⊗ HC 


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 


dim = int(dimA * dimB * dimC)
dimAux = int(7 * dim)  
Ia = np.eye(dimA, dimA)
Ib = np.eye(dimB, dimB)
Ic = np.eye(dimC, dimC)	
a = np.zeros([dimA])
b = np.zeros([dimB])
c = np.zeros([dimC])
dimAC = dimA * dimC
dimBC = dimB * dimC


##############################
def dagger(a):

    return np.transpose(a).conj()
##############################



##############################
def e(i,dimension):

	mat = np.zeros([dimension])
	mat[i] = 1.0 

	return mat
##############################


if __name__ == "__main__":


	X = np.zeros((dim,dimAux),dtype=complex)
	for i in range (0,dim):
		for k in range (0,dimAux):
			#X[i][k] = randrange(-1, 1) + (1j * randrange(-1, 1))
			X[i][k] = np.random.uniform(-1,1) + (1j * np.random.uniform(-1,1))


	rho_AC = np.zeros([dimAC, dimAC],dtype=complex) 
	rho_BC = np.zeros([dimBC, dimBC],dtype=complex)
	rho_C = np.zeros([dimC, dimC],dtype=complex)
	rho_ABC = np.matmul(X, dagger(X))
	rho_ABC /= np.trace(rho_ABC)
	 

	for i in range (0,dimB):

		tmp = np.kron(np.kron(Ia, e(i,dimB)), Ic)
		rho_AC += multi_dot([tmp, rho_ABC, tmp.T])


	for i in range (0,dimA):

		tmp = np.kron(np.kron(e(i,dimA), Ib), Ic)
		rho_BC += multi_dot([tmp, rho_ABC, tmp.T])


	for i in range (0,dimA):
		for k in range (0,dimB):

			tmp = np.kron(np.kron(e(i,dimA), e(i,dimB)), Ic)
			rho_C += multi_dot([tmp, rho_ABC, tmp.T])


	SBC = -1.0 * np.trace(np.dot(rho_BC, logm(rho_BC))).real
	print ("SBC =", SBC)
	SAC = -1.0 * np.trace(np.dot(rho_AC, logm(rho_AC))).real
	print ("SAC =", SAC)
	SC = -1.0 * np.trace(np.dot(rho_C, logm(rho_C))).real
	print ("SC =", SC)
	SABC = -1.0 * np.trace(np.dot(rho_ABC, logm(rho_ABC))).real  # logm - Matrix log!
	print ("SABC is", SABC)
	print ("SAC + SBC - SABC - SC = ", SAC + SBC - SABC - SC) # Must be + always for SSA to hold!
	print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 




