#python3
# Pass a file with COL1 = kappa and COL2 = free energy 
# and compute the second numerical derivative 
# Dec 13, 2018

import sys
import itertools
import scipy 
from scipy import integrate
import numpy as np

if len( sys.argv ) == 2 :
    filename = sys.argv[1]
if len( sys.argv ) == 0 or len( sys.argv ) > 2:
    print("Requires one argument : FILE with format $kappa$ $f$ ")
    print("Run this as : python num_dif.py FILE ")
    sys.exit()

x=[]
y=[]

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

# Print the out array to a file for plotting 
with open('output.dat', 'w') as f:
    for item in d2ydx2:
        f.write("%s\n" % item)