import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
from numpy import ndarray
from matplotlib import pyplot as plt
import time
import datetime
import itertools

if len( sys.argv ) == 2 :
    filename = sys.argv[1]
if len( sys.argv ) == 0 or len( sys.argv ) > 2:
    print("Requires one argument : FILE with format $kappa$ $f$ ")
    print("Run this as : python num_dif.py FILE ")
    sys.exit()

temp=[]
dfdT=[]
d2FdT2 = [] 

file = open(filename, "r")
for line in itertools.islice(file, 0, None):
    
    line = line.split()
    temp.append(float(line[0]))
    dfdT.append(float(line[1]))
    d2FdT2.append(float(line[2]))

out = [] 
for i in range(0, len(dfdT)): 
    out.append(dfdT[i] * temp[i] * temp[i]) 
out1 = [] 
for i in range(0, len(d2FdT2)):
    out1.append(d2FdT2[i] * temp[i] * temp[i] * temp[i] * temp[i])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

f = plt.figure()
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('T',fontsize=13)
ax1.set_ylabel('U', color=color,fontsize=13)
ax1.plot(temp, out, marker="*", color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.set_ylabel('Cv', color=color,fontsize=13) 
ax2.plot(temp, out1, marker="o", color=color)
plt.grid(True)
ax2.tick_params(axis='y', labelcolor=color)
plt.title(r"3d classical Ising model using Triad TRG",fontsize=16, color='black')
fig.tight_layout()
plt.show()

    
