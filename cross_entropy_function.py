# -*- coding: utf-8 -*-
"""
Cross Entropy function
"""
import numpy as np

# We write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    ce = 0
    for i in xrange(len(Y)):
        if Y[i] == 1:
            ce1 = - np.log(P[i])
        else:
            ce1 = - np.log(1-P[i])
        ce += ce1
    return ce


Y=[1,0,1,1]
P=[0.4,0.6,0.1,0.5]
cross_entropy(Y, P)

#The correct answer is 4.8283137373 
#And your code returned 4.8283137373 

# Another SOLUTION
import numpy as np

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))