#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:53:52 2020

@author: cjburke
Based on Ben Botto Blog post
https://medium.com/@benjamin.botto/implementing-an-optimal-rubiks-cube-solver-using-korf-s-algorithm-bf750b332cf9
See that for inspiration
"""
import numpy as np

class lehmer_code():
    MAXN = 12
    def __init__(self, n, k=None):
        # Build factorial or pick function table
        # n [int] - item size
        # k [int] - number selected from items
        if k is None:
            k = n
        if n > self.MAXN:
            print('lehmer code cannot handle n > {0:d}'.format(self.MAXN))
            print('Increase np.uint sizes if you want larger n')
            print('But remember just because you can doesnt mean you should')
            exit()
            
        self.n = n
        self.k = k
        # Build the factorial and n pick k functions ahead of time
        if n == k:
            self.useFacts = np.zeros((n,), dtype=np.uint32)
            # factorial in reverse order
            for i in range(n):
                self.useFacts[i] = np.math.factorial(n-i-1)
                #print(i, self.useFacts[i])
        else:
            self.useFacts = np.zeros((k,), dtype=np.uint32)
            # n choose k in reverse order
            for i in range(k):
                self.useFacts[i] = np.math.factorial(n-i-1) / np.math.factorial(n-k)
                #print(i, self.useFacts[i])
        # Build count seen lookup dictionary
        lrgInt = np.power(2, n)
        self.countbits = np.zeros((lrgInt,), dtype=np.int8)
        #countstr = ''
        for i in range(lrgInt):
            self.countbits[i] = bin(i).count("1")
        #    countstr = countstr + ',' + str(self.countbits[i])
        #print(countstr)        
        
    
    def encode(self, p):
        lehmer = np.zeros((self.k), dtype=np.int8)
        lehmer[0] = np.int8(p[0])
        
        seen = 0b0
        seen = seen | (0b1 << (self.n - p[0] - 1))
        #saveseen = 0b1 << (self.k - p[0] -1)
        for i in range(1, self.k):
            seen = seen | (0b1 << (self.n - p[i] - 1 ))
            #tmp = bin(seen)
            rshift = self.n - p[i]
            numOnes = self.countbits[seen >> rshift]
            lehmer[i] = p[i] - numOnes
            #if i == 3:
            #    saveseen = lehmer[i]

        return np.sum(lehmer*self.useFacts)
        
if __name__ == '__main__':
    
    #lc = lehmer_code(8)
    
    #lcidx = lc.encode([7,6,5,4,3,2,1,0])
    #print(lcidx)
    
    lc = lehmer_code(4,2)
    lcidx = lc.encode([0,2])
    print(lcidx)