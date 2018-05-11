# -*- coding: utf-8 -*-
"""
Softmax function
"""
import numpy as np

# 1. Function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    p = []
    for i in xrange(len(L)):    
        en = np.exp(L[i]) / np.sum(np.exp(L))
        p.append(en)
    return p

# alternative SOLUTION
import numpy as np

def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
    
    # Note: The function np.divide can also be used here, as follows:
    # def softmax(L):
    #     expL = np.exp(L)
    #     return np.divide (expL, expL.sum())

#Trying for L=[5,6,7]. 
#The correct answer is [0.09003057317038046, 0.24472847105479764, 0.6652409557748219] 
#And your code returned [0.09003057317038046, 0.24472847105479764, 0.6652409557748219] 
#Correct!

# 2. Calculate location of points in blue/red zone using Sigmoid:
from math import exp
x1 = [1, 2, 5, -4] 
x2 = [1, 4, -5, 5]
cnt = 0
for i in xrange(len(x1)):
    x = 4*x1[i] + 5*x2[i] - 9
    sig = 1 / (1+ exp(-x))
    print '\ncount = {}'.format(cnt)
    print x
    print sig
    cnt += 1
# Check
x1 = -4
x2 = 5
x = 4*x1 + 5*x2 - 9
sig = 1 / (1+ exp(-x))
x
sig

exp(1) / (exp(0)+exp(1))
exp(0) / (exp(0)+exp(1))
