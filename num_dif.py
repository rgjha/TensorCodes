# python3
# Pass a file with COL1 = variable and COL2 = usually log(Z) or free energy  
# and compute the 1st/2nd numerical derivative 
# Dec 13, 2018

import sys
import itertools
import scipy 
from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt
from numpy import diff


if len( sys.argv ) == 2 :
    filename = sys.argv[1]
if len( sys.argv ) == 0 or len( sys.argv ) > 2:
    print("Requires one argument : FILE with format $kappa$ $f$ ")
    print("Run this as : python num_dif.py FILE ")
    sys.exit()

x=[]
y=[]
dydx = [] 

file = open(filename, "r")
for line in itertools.islice(file, 0, None):
    
    line = line.split()
    tmp = float(line[0])
    tmp1 = float(line[1])
    x.append(tmp)
    y.append(tmp1)


dx = x[1]-x[0]  # Asssuming equal spacing 
dydx = np.gradient(y, dx)
d2ydx2 = np.gradient(dydx, dx)

# dydx = diff(y)/diff(x) Unequal spacing!

# Print the 1st numerical derivative (column 2) array to a file
# with the variable (column 1) and 2nd numerical derivative (column 3)
with open('output_num_dif', 'w') as f:
    res = "\n".join("{} {} {}".format(x, y, z) for x, y, z in zip(x, -dydx, -d2ydx2)) 
    f.write("%s\n" % (res))


# Now plot the variable and numerical derivative.

out = [] 
for i in range(0, len(dydx)): 
    out.append(-dydx[i] * x[i] * x[i]) 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
f = plt.figure()
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel(r'$\beta$',fontsize=13)
ax1.set_ylabel('.', color=color,fontsize=13)
ax1.plot(x, out, marker="*", color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.grid(True)
plt.title(r"Numerical derivative",fontsize=16, color='black')
fig.tight_layout()
plt.show()
