#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:28:28 2020

@author: cjburke
python setup.py build_ext --inplace
"""
cimport cython

cimport libc.stdlib as lib
from libc cimport stdint
from libc.string cimport memcpy
import numpy
cimport numpy

ctypedef numpy.int_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

#https://stackoverflow.com/questions/776508/best-practices-for-circular-shift-rotate-operations-in-c
cdef void rollface( stdint.uint64_t* arr, int movecount):
    cdef stdint.uint64_t fullval, n, newval
    fullval = arr[0]
    if movecount < 0: # only -2 value is used
        newval = (fullval>>16) | (fullval<<48)
    else:
        n = movecount * 8 # 2 or 4 bytes so *8 for bits
        newval = (fullval<<n) | (fullval>>(64-n))
    arr[0] = newval

cdef int[18][8] lhs = [[26,28,2,4,10,12,18,20],[10,12,18,20,26,28,2,4],[18,20,26,28,2,4,10,12],
                [30,24,6,0,14,8,22,16],[14,8,22,16,30,24,6,0],[22,16,30,24,6,0,14,8],
                [44,46,4,6,32,34,16,18],[32,34,16,18,44,46,4,6],[16,18,44,46,4,6,32,34],
                [40,42,20,22,36,38,0,2],[36,38,0,2,40,42,20,22],[20,22,36,38,0,2,40,42],
                [42,44,12,14,34,36,24,26],[34,36,24,26,42,44,12,14],[24,26,42,44,12,14,34,36],
                [46,40,28,30,38,32,8,10],[38,32,8,10,46,40,28,30],[28,30,38,32,8,10,46,40]]
cdef int[18][8] rhs = [[2,4,10,12,18,20,26,28],[2,4,10,12,18,20,26,28],[2,4,10,12,18,20,26,28],
                [6,0,14,8,22,16,30,24],[6,0,14,8,22,16,30,24],[6,0,14,8,22,16,30,24],
                [4,6,32,34,16,18,44,46],[4,6,32,34,16,18,44,46],[4,6,32,34,16,18,44,46],
                [0,2,40,42,20,22,36,38],[0,2,40,42,20,22,36,38],[0,2,40,42,20,22,36,38],
                [12,14,34,36,24,26,42,44],[12,14,34,36,24,26,42,44],[12,14,34,36,24,26,42,44],
                [8,10,46,40,28,30,38,32],[8,10,46,40,28,30,38,32],[8,10,46,40,28,30,38,32]]


cdef void dosides( stdint.uint8_t* out, stdint.uint8_t* orig, int cmv):

    cdef int* l
    cdef int* r
    l = lhs[cmv]
    r = rhs[cmv]
    # copy 2 8bit bytes at a time
    (<stdint.uint16_t*>(&out[l[0]]))[0] = (<stdint.uint16_t*>(&orig[r[0]]))[0]
    # copy 1 8bit byte and repeat for the 3 other sides
    out[l[1]] = orig[r[1]]
    (<stdint.uint16_t*>(&out[l[2]]))[0] = (<stdint.uint16_t*>(&orig[r[2]]))[0]
    out[l[3]] = orig[r[3]]
    (<stdint.uint16_t*>(&out[l[4]]))[0] = (<stdint.uint16_t*>(&orig[r[4]]))[0]
    out[l[5]] = orig[r[5]]
    (<stdint.uint16_t*>(&out[l[6]]))[0] = (<stdint.uint16_t*>(&orig[r[6]]))[0]
    out[l[7]] = orig[r[7]]


# Here is conversion from move # into which face is doing roll
#face B/yellow = 0; R/blue = 1; F/white = 2; L/green = 3; U/orange = 4; D/red = 5
cdef int[18] move2face = [5,5,5,4,4,4,1,1,1,3,3,3,2,2,2,0,0,0]
# give the number and direction of the roll needed to move face elements during a move
cdef int[18] move2shifts = [2,-2,4,-2,2,4,2,-2,4,-2,2,4,2,-2,4,-2,2,4]

cdef move_with_cython(stdint.uint8_t* fc, stdint.uint8_t* newmoves, int* allowed_moves):
    
    cdef Py_ssize_t k1, kuse, doface_idx
    # Copy the python list to c memory
    cdef size_t nMem = 48
    
    # This is where we move faces from the 2D transition input matrix for all 18 rubik moves
    for k1 in range(18):
        kuse = allowed_moves[k1]
        if not kuse == -1:
            # First copy the original faces into the newmoves array
            memcpy(&newmoves[k1*48], fc, nMem * sizeof(stdint.uint8_t))
            # Now do the rolling of faces for this move in place
            doface_idx = move2face[kuse] * 8
            rollface(<stdint.uint64_t*>&newmoves[k1*48+doface_idx], move2shifts[kuse])
            dosides(&newmoves[k1*48], fc, kuse)
    return 0

# Hard code factors for corner lehmer coding
cdef int[8] corner_p_idx = [42,44,40,46,36,34,38,32]
cdef int[8] corner_factors = [5040,720,120,24,6,2,1,1]
cdef int[256] corner_bincount = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8]
cdef int[7] corner_ternaryfactors = [729,243,81,27,9,3,1]

# Hard code factors for edge lehmer coding
cdef int[12] edge_p_idx = [43,41,45,47,25,13,29,9,35,37,33,39]
cdef int[12] edge_factors = [39916800,3628800,362880,40320,5040,720,120,24,6,2,1,1]
cdef int[4096] edge_bincount = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,7,8,8,9,8,9,9,10,8,9,9,10,9,10,10,11,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,7,8,8,9,8,9,9,10,8,9,9,10,9,10,10,11,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,7,8,8,9,8,9,9,10,8,9,9,10,9,10,10,11,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,7,8,8,9,8,9,9,10,8,9,9,10,9,10,10,11,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,7,8,8,9,8,9,9,10,8,9,9,10,9,10,10,11,5,6,6,7,6,7,7,8,6,7,7,8,7,8,8,9,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,7,8,8,9,8,9,9,10,8,9,9,10,9,10,10,11,6,7,7,8,7,8,8,9,7,8,8,9,8,9,9,10,7,8,8,9,8,9,9,10,8,9,9,10,9,10,10,11,7,8,8,9,8,9,9,10,8,9,9,10,9,10,10,11,8,9,9,10,9,10,10,11,9,10,10,11,10,11,11,12]

# Hard code factors for edge1 lehmer coding
# Factors for 12 pick 7
cdef int[7] edge1_p_idx = [43,45,25,29,35,33,41]
cdef int[7] edge1_factors = [332640,30240,3024,336,42,6,1]
cdef int[7] edge1_binaryfactors = [64,32,16,8,4,2,1]
cdef int[7] edge2_p_idx = [41,47,13,9,37,39,43]

cdef void lehmer_code_faces(stdint.uint8_t* fc, int* finalstates):
    
    cdef int nCrnr = 8
    cdef int nCrnrOrn = 7
    cdef int nEdge = 12
    cdef int nEdge1 = 7
    cdef Py_ssize_t  i2
    cdef int rshift, numOnes

    cdef int[8] corner_ids
    cdef unsigned char[8] corner_p
    cdef unsigned char[7] corner_op
    cdef unsigned char[8] corner_lehmer
    cdef unsigned char corner_seen
    # Variables to deal with edge lehmer coding
    cdef int[12] edge_ids
    cdef unsigned short[12] edge_p
    cdef int[18] edge_state
    cdef unsigned short[12] edge_lehmer
    cdef unsigned short edge_seen
    # Variables to deal with edge1 lehmer coding
    cdef int[7] edge1_ids
    cdef unsigned short[7] edge1_p
    cdef unsigned short[7] edge1_op
    cdef unsigned short[7] edge1_lehmer
    cdef unsigned short edge1_seen
    
    #str = ''
    for i2 in range(nCrnr):
        corner_ids[i2] = fc[corner_p_idx[i2]]
    #    str = str + ',{0:d}'.format(corner_ids[i2])
    #print('Corner Faces Used ',str)
    for i2 in range(nCrnr):
        corner_p[i2] = corner_ids[i2] >> 2
        if i2 < nCrnr - 1:
            corner_op[i2] = corner_ids[i2] & 3
    corner_lehmer[0] = corner_p[0]
    corner_seen = 0
    corner_seen = corner_seen | (0b1 << (nCrnr - corner_p[0] - 1))
    for i2 in range(1,nCrnr):
        corner_seen = corner_seen | (0b1 << (nCrnr - corner_p[i2] - 1 ))
        rshift = nCrnr - corner_p[i2]
        numOnes = corner_bincount[corner_seen >> rshift]
        corner_lehmer[i2] = corner_p[i2] - numOnes

    rshift = 0
    for i2 in range(nCrnr):
        rshift = rshift + corner_lehmer[i2] * corner_factors[i2]
    rshift = rshift * 2187
    numOnes = 0
    for i2 in range(nCrnrOrn):
        numOnes = numOnes + corner_op[i2] * corner_ternaryfactors[i2]
    finalstates[0] = rshift + numOnes
    
    # Calculate the edge state
    #str = ''
    for i2 in range(nEdge):
        edge_ids[i2] = fc[edge_p_idx[i2]]
        edge_p[i2] = (edge_ids[i2] >> 2) - 8 # The -8 is there because the corners are listed first
    #    str = str + ',{0:d}'.format(edge_ids[i2])
    #print('Edge Faces Used: ',str)
    edge_lehmer[0] = edge_p[0]
    edge_seen = 0
    edge_seen = edge_seen | (0b1 << (nEdge - edge_p[0] - 1))
    for i2 in range(1,nEdge):
        edge_seen = edge_seen | (0b1 << (nEdge - edge_p[i2] - 1 ))
        rshift = nEdge - edge_p[i2]
        numOnes = edge_bincount[edge_seen >> rshift]
        edge_lehmer[i2] = edge_p[i2] - numOnes
    rshift = 0
    for i2 in range(nEdge):
        rshift = rshift + edge_lehmer[i2] * edge_factors[i2]
    finalstates[1] = rshift
    
    # Calculate the edge1 state
    for i2 in range(nEdge1):
        edge1_ids[i2] = fc[edge1_p_idx[i2]]
    for i2 in range(nEdge1):
        edge1_p[i2] = (edge1_ids[i2] >> 2) - 8 # The -8 is there because the corners are listed first
        edge1_op[i2] = edge1_ids[i2] & 3
    edge1_lehmer[0] = edge1_p[0]
    edge1_seen = 0
    edge1_seen = edge1_seen | (0b1 << (nEdge - edge1_p[0] - 1))
    for i2 in range(1,nEdge1):
        edge1_seen = edge1_seen | (0b1 << (nEdge - edge1_p[i2] - 1 ))
        rshift = nEdge - edge1_p[i2]
        numOnes = edge_bincount[edge1_seen >> rshift]
        edge1_lehmer[i2] = edge1_p[i2] - numOnes

    rshift = 0
    for i2 in range(nEdge1):
        rshift = rshift + edge1_lehmer[i2] * edge1_factors[i2]
    # 12 pick 6
    #rshift = rshift * 64
    # 12 pick 7
    rshift = rshift * 128
    numOnes = 0
    for i2 in range(nEdge1):
        numOnes = numOnes + edge1_op[i2] * edge1_binaryfactors[i2]

    finalstates[2] = rshift + numOnes

    # Calculate the edge12 state
    for i2 in range(nEdge1):
        edge1_ids[i2] = fc[edge2_p_idx[i2]]
    for i2 in range(nEdge1):
        edge1_p[i2] = (edge1_ids[i2] >> 2) - 8 # The -8 is there because the corners are listed first
        edge1_op[i2] = edge1_ids[i2] & 3
    edge1_lehmer[0] = edge1_p[0]
    edge1_seen = 0
    edge1_seen = edge1_seen | (0b1 << (nEdge - edge1_p[0] - 1))
    for i2 in range(1,nEdge1):
        edge1_seen = edge1_seen | (0b1 << (nEdge - edge1_p[i2] - 1 ))
        rshift = nEdge - edge1_p[i2]
        numOnes = edge_bincount[edge1_seen >> rshift]
        edge1_lehmer[i2] = edge1_p[i2] - numOnes

    rshift = 0
    for i2 in range(nEdge1):
        rshift = rshift + edge1_lehmer[i2] * edge1_factors[i2]
    # 12 pick 6
    #rshift = rshift * 64
    # 12 pick 7
    rshift = rshift * 128
    numOnes = 0
    for i2 in range(nEdge1):
        numOnes = numOnes + edge1_op[i2] * edge1_binaryfactors[i2]

    finalstates[3] = rshift + numOnes


def DFS_cython_solve(bytes fc, int maxdeldep, DTYPE_t [:] corner, \
                     DTYPE_t [:] alledge, DTYPE_t [:] edge1, DTYPE_t [:] edge2):
    
    cdef int MAXLEVEL
    cdef int MAXBUFF
    MAXBUFF = 1000
    # Table to prune redundant back to back moves
    cdef int[19][18] ignore_moves = [
    [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,-1,-1],
    [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,-1,-1],
    [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,-1,-1,-1],
    [5,6,7,8,9,10,11,12,13,14,15,16,17,-1,-1,-1,-1,-1],
    [5,6,7,8,9,10,11,12,13,14,15,16,17,-1,-1,-1,-1,-1],
    [6,7,8,9,10,11,12,13,14,15,16,17,-1,-1,-1,-1,-1,-1],
    [0,1,2,3,4,5,8,9,10,11,12,13,14,15,16,17,-1,-1],
    [0,1,2,3,4,5,8,9,10,11,12,13,14,15,16,17,-1,-1],
    [0,1,2,3,4,5,9,10,11,12,13,14,15,16,17,-1,-1,-1],
    [0,1,2,3,4,5,11,12,13,14,15,16,17,-1,-1,-1,-1,-1],
    [0,1,2,3,4,5,11,12,13,14,15,16,17,-1,-1,-1,-1,-1],
    [0,1,2,3,4,5,12,13,14,15,16,17,-1,-1,-1,-1,-1,-1],
    [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,-1,-1],
    [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,-1,-1],
    [0,1,2,3,4,5,6,7,8,9,10,11,15,16,17,-1,-1,-1],
    [0,1,2,3,4,5,6,7,8,9,10,11,17,-1,-1,-1,-1,-1],
    [0,1,2,3,4,5,6,7,8,9,10,11,17,-1,-1,-1,-1,-1],
    [0,1,2,3,4,5,6,7,8,9,10,11,-1,-1,-1,-1,-1,-1],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
    
    cdef int bp
    cdef stdint.uint8_t[1000][48]  buffmoves 
    cdef stdint.uint8_t[18][48] newmoves
    cdef stdint.uint8_t[48] tmpfc
    cdef stdint.uint8_t* mvp
    cdef stdint.uint8_t* bpt
    cdef int[1000][23] buffdata
    cdef int[23] tmpdat
    cdef int* dpt
    
    cdef int lastMove, curLevel, cmv, turn1level
    cdef long totCnt
    cdef int notSolved, notEmpty
    cdef Py_ssize_t i, k, kk
    cdef int highn
    cdef int lehcode[4]
    cdef int score, tmpscore, start_dist, cs, ce, ce1, ce2
    
    # copy the original input face vector into the c variables
    for i in range(48):
        tmpfc[i] = fc[i]
    
    # configure tmpdat to start with -1
    for i in range(23):
        tmpdat[i] = -1
    # Test lehmer coding on solved cube
    #lehmer_code_faces(fc, lehcode)
    #print(lehcode[0], lehcode[1], lehcode[2], lehcode[3])
    # Results using original face ordering
    # 87913026  439084673  463509376  501877120
    
    # Get the initial lehmer cube code distances
    lehmer_code_faces(fc, lehcode)
    score = corner[lehcode[0]]
    tmpscore = alledge[lehcode[1]]
    if tmpscore > score:
        score = tmpscore
    tmpscore = edge1[lehcode[2]]
    if tmpscore > score:
        score = tmpscore
    tmpscore = edge2[lehcode[3]]
    if tmpscore > score:
        score = tmpscore
    start_dist = score
    #print('Starting Distance: {0:d}'.format(start_dist))
    MAXLEVEL = start_dist + maxdeldep

    
    # Do the initial filling of buff moves and buffdata with the first moves
    lastMove = 18
    bp = -1
    curLevel = 1
    turn1level = 0
    totCnt = 0
    highn = 0
    move_with_cython(tmpfc, <stdint.uint8_t*>newmoves, ignore_moves[lastMove])
    for i in range(18):
        cmv = ignore_moves[lastMove][i]
        if not cmv == -1:
            # Get Score of this configuration
            mvp = <stdint.uint8_t*>&(newmoves[i])
            lehmer_code_faces(mvp, lehcode)
            
            score = corner[lehcode[0]]
            tmpscore = alledge[lehcode[1]]
            if tmpscore > score:
                score = tmpscore
            tmpscore = edge1[lehcode[2]]
            if tmpscore > score:
                score = tmpscore
            tmpscore = edge2[lehcode[3]]
            if tmpscore > score:
                score = tmpscore
            if score <= MAXLEVEL:            
                bp = bp + 1
                bpt = <stdint.uint8_t*>&(buffmoves[bp])
                memcpy(bpt, mvp, 48)
                tmpdat[0] = curLevel
                tmpdat[1] = cmv
                tmpdat[curLevel +1] = cmv
                dpt = <int*>&(buffdata[bp])
                memcpy(dpt, tmpdat, sizeof(int)*23)

    # DEBUG the first moves are correct
#    facecodechars = ["c012","e051","c051","e091","c062","e061","c021","e011",\
#                 "c022","e060","c061","e101","c072","e070","c031","e021",\
#                 "c032","e071","c071","e111","c082","e081","c041","e031",\
#                 "c042","e080","c081","e121","c052","e050","c011","e041",\
#                 "c020","e020","c030","e030","c040","e040","c010","e010",\
#                 "c050","e120","c080","e110","c070","e100","c060","e090"]
#    ia = numpy.argsort(facecodechars)
#    for i in range(18):
#        for k in range(48):
#            j = ia[k]
#            print("{0} {1:d} {2:d}".format(facecodechars[j], newmoves[i][j], i))
 
    notSolved = 1
    notEmpty = 1
    frstpass = 0
    while notSolved and notEmpty:
        # keep of track of when we get back to the first level by printing out move
        curLevel = buffdata[bp][0]
        if curLevel == 1:
            turn1level = turn1level + 1
            #print("Starting First Turn Level: {0:d} {1:d}".format(turn1level, buffdata[bp][1]))
        if (curLevel < MAXLEVEL):
            bpt = <stdint.uint8_t*>&(buffmoves[bp])
            memcpy(tmpfc, bpt, 48)
            lastMove = buffdata[bp][1]
            move_with_cython(tmpfc, <stdint.uint8_t*>newmoves, ignore_moves[lastMove])
            dpt = <int*>&(buffdata[bp])
            memcpy(tmpdat, dpt, sizeof(int)*23)
            bp = bp - 1
            for i in range(18):
                cmv = ignore_moves[lastMove][i]
                if not cmv == -1:
                    # Get Score of this configuration
                    mvp = <stdint.uint8_t*>&(newmoves[i])
                    lehmer_code_faces(mvp, lehcode)
                    score = corner[lehcode[0]]
                    cs = score
                    ce = alledge[lehcode[1]]
                    if ce > score:
                        score = ce
                    ce1 = edge1[lehcode[2]]
                    if ce1 > score:
                        score = ce1
                    ce2 = edge2[lehcode[3]]
                    if ce2 > score:
                        score = ce2
                    # Look to see if this solves cube
                    if lehcode[0] == 87913026 and lehcode[1] == 439084673 and lehcode[2] == 463509376 and lehcode[3] == 501877120:
                        # Solved!
                        #str = ''
                        #for i in range(48):
                        #    str = str + ",{0:d}".format((mvp+i)[0])
                        #print(str)
                        print("Max Buffer Fill: {0:d}".format(highn))
                        print("Total Moves: {0:d}".format(totCnt))
                        tmpdat[0] = curLevel + 1
                        print("Solve Cube Moves N: {0:d}".format(tmpdat[0]))
                        tmpdat[1] = cmv
                        tmpdat[tmpdat[0] +1] = cmv
                        str = ''
                        for i in range(2,23):
                            if not tmpdat[i] == -1:
                                str = str + "_{0:d}".format(tmpdat[i])
                        print(str)
                        notSolved = 0
                        return 2
                    # Debug turns 
                    #if tmpdat[2] == 1 and tmpdat[3] == 4 and tmpdat[4] == 13 and tmpdat[5] == 16 and tmpdat[6] == 7 and cmv == 10 and tmpdat[8] == -1 and tmpdat[9] == -1:
                    #    print(score, curLevel, MAXLEVEL, cs, ce, ce1, ce2)
                    if score+curLevel < MAXLEVEL and curLevel < MAXLEVEL:         
                        bp = bp + 1
                        bpt = <stdint.uint8_t*>&(buffmoves[bp])
                        memcpy(bpt, mvp, 48)
                        tmpdat[0] = curLevel +1
                        tmpdat[1] = cmv
                        tmpdat[tmpdat[0] +1] = cmv
                        dpt = <int*>&(buffdata[bp])
                        memcpy(dpt, tmpdat, sizeof(int)*23)
    
                        totCnt = totCnt + 1
        else:
            # pop off list be decrementing buffer position
            bp = bp - 1
        
        # check for empty
        if bp == -1:
            notEmpty = 0
        # report high buffer count
        if (bp > highn):
            highn = bp
        # Check for too close to buffer limit
        if (bp > MAXBUFF - 20):
            print("Ran Out Of Buffer Space Increase Buffer! Exiting")
            return 1
    print("No Solution Found")
    print("Max Buffer Fill: {0:d}".format(highn))
    print("Total Moves: {0:d}".format(totCnt))
    return 0
            
            
