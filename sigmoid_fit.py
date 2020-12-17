#!/usr/bin/python3
import os
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammainc
# ------------------------------------------------------------------
# Fit 3d  data to
#   dln(Z)/d(beta) = D - A / (1 + exp[B * (T/mu - C)])

if len(sys.argv) < 2:
  print("Usage:", str(sys.argv[0]), "<file>")
  sys.exit(1)
filename = str(sys.argv[1])

if not os.path.isfile(filename):
  print("ERROR:", filename, "does not exist")
  sys.exit(1)

expfunc = lambda p, x: p[0] - p[1] / (1.0 + np.exp(p[2] * (x - p[3])))
errfunc = lambda p, x, y, err: (expfunc(p, x) - y) / err
p_in = np.array([0.9, 0.9, 100.0, 0.1])
lower = np.array([0.0, 0.0, 0.0, 0.0])
upper = np.array([1.0, 1.0, np.inf, 1.0])

def jac(p, x, y, err):
  J = np.empty((x.size, p.size), dtype = np.float)
  num = np.exp(p[2] * (x - p[3]))
  den = 1.0 + np.exp(p[2] * (x - p[3]))
  J[:, 0] = 1.0
  J[:, 1] = -1.0 / den
  J[:, 2] = p[1] * (x - p[3]) * num / den**2
  J[:, 3] = -1.0 * p[1] * p[2] * num / den**2
  for i in range(p.size):
    J[:, i] /= err
  return J
# ------------------------------------------------------------------

TList = []
datList = []
errList = []
for line in open(filename):
  if len(line) == 1 or line.startswith('#') or line.startswith('!'):
    continue
  temp = line.split()
  TList.append(float(temp[0]))
  datList.append(float(temp[1]))
  errList.append(float(temp[2]))
T = np.array(TList)
dat = np.array(datList)
err = np.array(errList)

dof = len(T) - len(p_in)
if dof < 1:
  print("ERROR: dof > 0 required")
  sys.exit(1)

all_out = least_squares(errfunc, p_in, bounds=[lower, upper],
                        jac=jac, method='trf', args=(T, dat, err))
p_out = all_out.x
tj = all_out.jac
cov = np.linalg.inv(np.dot(np.transpose(tj), tj))

if all_out.success < 0 or all_out.success > 4:
  print("ERROR: Fit failed with the following error message")
  print(errmsg)
  sys.exit(1)

loT = p_out[0] - p_out[1]
derivs = np.array([1.0, -1.0, 0.0, 0.0])
loTerr = np.sqrt(np.dot(derivs, np.dot(cov, derivs)))

print("  Critical beta: %.6g %.4g" % (p_out[3], np.sqrt(cov[3][3])))

# Compute chiSq and confidence level of fit
chiSq = ((errfunc(p_out, T, dat, err))**2).sum()
CL = 1.0 - gammainc(0.5 * dof, 0.5 * chiSq)
print("chiSq/dof = %.4g/%d = %.4g --> CL = %.4g" \
      % (chiSq, dof, chiSq / dof, CL))

# Format to copy+paste into gnuplot
print("\nf(x)=%.4g - %.4g / (1 + exp(%.4g * (x - %.4g)))" \
      % (p_out[0], p_out[1], p_out[2], p_out[3]))
# ------------------------------------------------------------------
