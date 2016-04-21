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
k2=1e8
c1=1e3
c2=1e3

M=np.matrix([[m1, 0], [0, m2]])
K=np.matrix([[k1, -k1], [-k1, k1+k2]])
C=np.matrix([[c1, -c1], [c1, c1+c2]])
F=np.matrix([1, 0])
F=F.transpose()
u=np.empty([2,0])

for iw in w:
    ut=(-iw**2*M+1j*iw*C+K).I*F
    u=np.append(u,ut,axis=1)

u=u.transpose()
plt.loglog(f,abs(u))
plt.show()
plt.xlabel('Freq, Hz')
plt.ylabel('Abs(u)')