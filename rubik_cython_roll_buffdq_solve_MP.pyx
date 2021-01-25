#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:28:28 2020

@author: cjburke
python setup.py build_ext --inplace
Solve the rubik's cube with a Iterative Deepening Depth First Search
With a precalculated pattern database of minimum length to solve 
  for various configurations.
  This borrows heavily from the excellent Blog piece and code by
  Benjamin Botto https://github.com/benbotto
  https://medium.com/@benjamin.botto/implementing-an-optimal-rubiks-cube-solver-using-korf-s-algorithm-bf750b332cf9
  Definitely read the blog about this before you dive into the code and comments
  Hereafter in the comments I will refer to this blog post as BottoB
  Note: The cython code here is what is run for finding the solution
  There are surrogates for nearly all these functions in the python
  side. I will only add comments here that have not been addressed
  on the python side, in the README.md or BottoB.
"""
cimport cython

cimport libc.stdlib as lib
from libc cimport stdint
from libc.string cimport memcpy
import numpy
cimport numpy

ctypedef numpy.int16_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

#https://stackoverflow.com/questions/776508/best-practices-for-circular-shift-rotate-operations-in-c
# Here is where the circular shift of bits takes place
# for the face move
# BottoB uses an assembly intrinsic to do this
#  Since the move is not the bottleneck I have not tried to the assembly
#   intrinsic
cdef void rollface( stdint.uint64_t* arr, int movecount):
    cdef stdint.uint64_t fullval, n, newval
    fullval = arr[0]
    if movecount < 0: # only -2 value is used
        newval = (fullval>>16) | (fullval<<48)
    else:
        n = movecount * 8 # 2 or 4 bytes so *8 for bits
        newval = (fullval<<n) | (fullval>>(64-n))
    arr[0] = newval

# These are all the indices needed to perform the side face moves
# It is defined in the cython module rather than in the function
# since it appeared to save time defining it once rather than every
# call to the side move. Cython variables with cdef are static and
# remain defined between calls and they are visible in the funciton scope
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

# Here is where the sides are moved using the lhs and rhs constants above
# The principle is to move and copy two bytes at a time casting them
#  to uint16_t when they are stored as uint8_t
#  This idea comes straight from BottoB code
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

# This is where the move is done. It is restricted to only do the move ids
# that are given in allowed_moves vector. The only pruning done here
#  is the redundant back-to-back moves move score pruning is done elsewhere
# INPUT
# fc - The initial 48 faceids
# newmoves (also output) - Store the resulting faceids after move
# allowed_moves - Array of moves that are allowed  
cdef move_with_cython(stdint.uint8_t* fc, stdint.uint8_t* newmoves, int* allowed_moves):
    
    cdef Py_ssize_t k1, kuse, doface_idx
    # Copy the python list to c memory
    cdef size_t nMem = 48
    
    # This is where we move faces using the face roll and
    #  side faces separately
    for k1 in range(18):
        kuse = allowed_moves[k1]
        if not kuse == -1:
            # First copy the original faces into the newmoves array
            memcpy(&newmoves[k1*48], fc, nMem * sizeof(stdint.uint8_t))
            # Now do the rolling of faces for this move in place
            doface_idx = move2face[kuse] * 8
            # The doface_idx and move number tells use where to point to
            #  in the face array. 64 bit cast to move all 8 bytes at once
            rollface(<stdint.uint64_t*>&newmoves[k1*48+doface_idx], move2shifts[kuse])
            # Do the side face moves
            dosides(&newmoves[k1*48], fc, kuse)
    return 0

# again putting constant factors once in module saves time
# and they are availble in the function scope
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

# See BottoB for description
#  This is the slowest function; 3 times slower than the face move
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


# Main entry point for performing DFS search from the initial face configuration
# up to maxlev for the search
# INPUT
# fc - input 48 faceids 
# maxlev - stop search at this level
# strtlev - commence search at this level
# strtmv - last move that lead to the input fc cube configuration
# corner, alledge, edge1, edge2 - reference to the pattern databases
#  that provide the number of moves needed to solve the given sub configuration
# OUTPUT - retval == 2 if solution found ; 0 if not
def DFS_cython_solve(bytes fc, int maxlev, int strtlev, int strtmv, DTYPE_t [:] corner, \
                     DTYPE_t [:] alledge, DTYPE_t [:] edge1, DTYPE_t [:] edge2):
    
    cdef int MAXLEVEL # max level searched
    cdef int MAXBUFF # The DFS stack is maintained in a 2D array
                # This sets the maximum size of the stack
                # even up to level 19 I have not seen more than 200
                # maxfill so this buffer is comfortably large
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
    
    cdef int bp # Points to the current head of the stack
    cdef stdint.uint8_t[1000][48]  buffmoves # The faceid buffer
        # each row contains the 48 faceids for a cube configuration
    cdef stdint.uint8_t[18][48] newmoves # This stores output configurations
        # after the allowed moves
    cdef stdint.uint8_t[48] tmpfc # store faceids for a cube configureation
    cdef stdint.uint8_t* mvp # pointer for faceid data
    cdef stdint.uint8_t* bpt # pointer to move buffer data
    cdef int[1000][23] buffdata # This auxillary data buffer is filled
        # in parallel with the faceid buffer
        #  it stores the current level, last moveid, and move history
        #  room to store a history up to 20 moves
    cdef int[23] tmpdat # temp storage for auxillary data
    cdef int* dpt # pointer to auxillary data
    
    cdef int lastMove, curLevel, cmv, turn1level
    cdef long totCnt # keep track of total # of moves
    cdef int notSolved, notEmpty
    cdef Py_ssize_t i, k, kk
    cdef int highn # keep track of largest buffer fill encountered
    cdef int lehcode[4] # keep lehmer codes
    cdef int score, tmpscore, start_dist, cs, ce, ce1, ce2 # scores/ distance
        # to solving for the pattern databases
    curLevel = strtlev
    lastMove = strtmv
    # copy the original input face vector into the c variables
    #str = ''
    for i in range(48):
        tmpfc[i] = fc[i]
    #    str = str +',{0:d}'.format(fc[i])
    #print('intput fc: ',str)
    
    # configure tmpdat to start with -1
    for i in range(23):
        tmpdat[i] = -1
    # DEBUG LINES
    # Test lehmer coding on solved cube
    #print('Move:',strtmv)
    #lehmer_code_faces(fc, lehcode)
    #print(lehcode[0], lehcode[1], lehcode[2], lehcode[3])
    # Results using original face ordering
    # 87913026  439084673  463509376  501877120
    
    # Get the initial lehmer cube code distances
#    lehmer_code_faces(fc, lehcode)
#    score = corner[lehcode[0]]
#    tmpscore = alledge[lehcode[1]]
#    if tmpscore > score:
#        score = tmpscore
#    tmpscore = edge1[lehcode[2]]
#    if tmpscore > score:
#        score = tmpscore
#    tmpscore = edge2[lehcode[3]]
#    if tmpscore > score:
#        score = tmpscore
#    start_dist = score
    #print('Starting Distance: {0:d}'.format(start_dist))
    MAXLEVEL = maxlev
    
    # Do the initial filling of buff moves and buffdata with the first moves
    bp = -1
    turn1level = 0
    totCnt = 0
    highn = 0
    # Perform the first set of moves
    move_with_cython(tmpfc, <stdint.uint8_t*>newmoves, ignore_moves[lastMove])
    for i in range(18):
        cmv = ignore_moves[lastMove][i] 
        if not cmv == -1: # This ignores the redundant moves
            # Get Score of this configuration
            mvp = <stdint.uint8_t*>&(newmoves[i])
            lehmer_code_faces(mvp, lehcode)
            # score is the maximum among all the databases
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
            if score <= MAXLEVEL: # This is the pruning by score step
                   # if there is too many steps needed for the max number
                   # allowed we can prune this configure
                   #  otherwise if it is <= MAXLEVEL record this step
                bp = bp + 1 # incremnt buffer head location
                bpt = <stdint.uint8_t*>&(buffmoves[bp]) # move buffer pointer
                memcpy(bpt, mvp, 48) # copy configuration to move buffer
                tmpdat[0] = curLevel # record auxillary data as well
                tmpdat[1] = cmv
                tmpdat[curLevel +1] = cmv
                dpt = <int*>&(buffdata[bp]) # pointer to aux data buffer
                memcpy(dpt, tmpdat, sizeof(int)*23) # copy aux data over

    # DEBUG to confirm that the first moves are correct
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
 
    # Keep iterating over the DFS stack
    notSolved = 1
    notEmpty = 1
    frstpass = 0
    while notSolved and notEmpty:
        curLevel = buffdata[bp][0]
        if (curLevel < MAXLEVEL): # make sure this move does not exceed level
            bpt = <stdint.uint8_t*>&(buffmoves[bp]) # copy face config to temporary storage
            memcpy(tmpfc, bpt, 48)
            lastMove = buffdata[bp][1]
            # perform all allowed moves
            move_with_cython(tmpfc, <stdint.uint8_t*>newmoves, ignore_moves[lastMove])
            dpt = <int*>&(buffdata[bp]) # copy aux data to temp storage
            memcpy(tmpdat, dpt, sizeof(int)*23)
            bp = bp - 1 # pop the move off just by decrementing head location

            # go through newmoves and see which ones pass the score test
            for i in range(18):
                cmv = ignore_moves[lastMove][i]
                if not cmv == -1:
                    # Get Score of this configuration
                    mvp = <stdint.uint8_t*>&(newmoves[i])
                    # get distance to end from databases
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
                    if lehcode[0] == 87913026 and lehcode[1] == 439084673 and lehcode[2] == 463509376 and lehcode[3] == 501877120 and curLevel < MAXLEVEL:
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
                        str1 = ''
                        str2 = ''
                        str3 = ''
                        rotnames = ["DR","DL","DH",\
                                    "UR","UL","UH",\
                                    "RU","RD","RH",\
                                    "LU","LD","LH",\
                                    "FC","FG","FH",\
                                    "BC","BG","BH"]
                        char_move_dict = ["D","D'","D2",\
                                          "U'","U","U2",\
                                          "R","R'","R2",\
                                          "L'","L","L2",\
                                          "F","F'","F2",\
                                          "B'","B","B2"]

                        for i in range(2,23):
                            if not tmpdat[i] == -1:
                                str1 = str1 + "_{0:d}".format(tmpdat[i])
                                str2 = str2 + "_{0}".format(rotnames[tmpdat[i]])
                                str3 = str3 + "_{0}".format(char_move_dict[tmpdat[i]])
                        print(str1)
                        print(str2)
                        print(str3)
                        notSolved = 0
                        return 2 # Found solution Bye!
                    # Debug turns 
                    #if tmpdat[2] == 1 and tmpdat[3] == 4 and tmpdat[4] == 13 and tmpdat[5] == 16 and tmpdat[6] == 7 and cmv == 10 and tmpdat[8] == -1 and tmpdat[9] == -1:
                    #    print(score, curLevel, MAXLEVEL, cs, ce, ce1, ce2)
                    if score+curLevel < MAXLEVEL and curLevel < MAXLEVEL:
                        # This moves passes the score check add it to the stack
                        # along with aux data
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
    #print("No Solution Found")
    #print("Max Buffer Fill: {0:d}".format(highn))
    #print("Total Moves: {0:d}".format(totCnt))
    return 0 # no solution found result
            
            