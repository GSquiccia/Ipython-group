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

m1=10
m2=10
k1=1e8
k2=1e1
c1=1e2
c2=1e2

M=np.array([[m1, 0], [0, m2]])
K=np.array([[k1, -k1], [-k1, k1+k2]])
C=np.array([[c1, -c1], [c1, c1+c2]])
F=np.array([[1 , 0],[0 , 1]])
#F=F.transpose()
u=np.empty([2,2,0])
u=np.zeros([len(f),2,2],'complex')
cc=0

eigval,eigvect=np.linalg.eig(inv(M).dot(K))

f0=np.sqrt(eigval)/2/np.pi
f02=np.sqrt(k1/(m1/2))/2/np.pi
# compare f02 with f0[2], they are equa lfor low k2

for iw in w:
    ut=inv(-iw**2*M+1j*iw*C+K).dot(F)
    #u=np.append(u,np.atleast_3d(ut),axis=2)
    u[cc,:,:]=ut
    cc+=1

#u=u.transpose()

uI=inv(u)

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