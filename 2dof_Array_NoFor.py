# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:20:09 2016

@author: gs13g11
"""

import numpy as np

import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.matlib import rand,zeros,ones,empty,eye

f=np.logspace(1,4,500)
w=f*2*np.pi
Nf=np.size(w)

m1=10
m2=10
k1=1e8
k2=1e8
c1=1e3
c2=1e3

M=np.array([[m1, 0], [0, m2]])
K=np.array([[k1, -k1], [-k1, k1+k2]])
C=np.array([[c1, -c1], [c1, c1+c2]])
F=np.array([[1 , 0],[0 , 1]])
F=F.transpose()

M=np.tile(M,(Nf,1,1))
K=np.tile(K,(Nf,1,1))
C=np.tile(C,(Nf,1,1))
# ATTENTION!!! F does not need tile because of definion of .dot for nd arrays
# dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
F=np.tile(F,(1,1,1))

w=np.tile(w,(2,2,1))
w=w.transpose()

A=M.dot(F)

u=inv(-w**2*M+1j*w*C+K).dot(F)
u=u.squeeze()

plt.figure()
plt.subplot(2,2,1)
plt.loglog(f,abs(u[:,0,0]))
plt.show()
plt.xlabel('Freq, Hz')
plt.ylabel('Abs(u)')
plt.subplot(2,2,2)
plt.loglog(f,abs(u[:,0,1]))
plt.show()
plt.xlabel('Freq, Hz')
plt.ylabel('Abs(u)')
plt.subplot(2,2,3)
plt.loglog(f,abs(u[:,1,0]))
plt.show()
plt.xlabel('Freq, Hz')
plt.ylabel('Abs(u)')
plt.subplot(2,2,4)
plt.loglog(f,abs(u[:,1,1]))
plt.show()
plt.xlabel('Freq, Hz')
plt.ylabel('Abs(u)')