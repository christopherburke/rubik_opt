#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 23:32:28 2020

@author: cjburke
Solve the rubik's cube with a Iterative Deepening Depth First Search
With a precalculated pattern database of minimum length to solve 
  for various configurations.
  This borrows heavily from the excellent Blog piece and code by
  Benjamin Botto https://github.com/benbotto
  https://medium.com/@benjamin.botto/implementing-an-optimal-rubiks-cube-solver-using-korf-s-algorithm-bf750b332cf9
  Definitely read the blog about this before you dive into the code and comments
  Hereafter in the comments I will refer to this blog post as BottoB
  Note: The python code here is not used for finding the solution
  It is here because it was prototyped in python first and it is useful
  to have the functions to perform moves ahead of time.
  Thus, many of these functions have nearly identical surrogates in the 
  cython code as well.
"""
from multiprocessing import Pool, RawArray, cpu_count
import numpy as np
import rubik_cython_roll_buffdq_solve as rcm
import rubik_cython_roll_buffdq_solve_MP as rcmMP
import copy
import lehmer_code as lc
from collections import deque as dq
from timeit import default_timer as timer

class rubiks_cube():

    # Character names for my non-standard face moves
    #  see the README.md
    rotnames = ["DR", "DL", "DH",\
                "UR", "UL", "UH",\
                "RU", "RD", "RH",\
                "LU", "LD", "LH",\
                "FC", "FG", "FH",\
                "BC", "BG", "BH"]
                # above moves identified by integer
                # 0 ,   1 ,  2, \
                # 3,    4,   5, \
                # 6,   7,    8, \
                # 9,   10,  11, \
                # 12, 13,  14, \
                # 15, 16,  17
    # only do the following moves if the last move was as key
    # this is used to prune redundant moves
    #  discussed in BottoB
    ignore_moves = {0:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],\
                    1:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                    2:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],\
                    3:[5,6,7,8,9,10,11,12,13,14,15,16,17],\
                    4:[5,6,7,8,9,10,11,12,13,14,15,16,17],\
                    5:[6,7,8,9,10,11,12,13,14,15,16,17],\
                    6:[0,1,2,3,4,5,8,9,10,11,12,13,14,15,16,17],\
                    7:[0,1,2,3,4,5,8,9,10,11,12,13,14,15,16,17],\
                    8:[0,1,2,3,4,5,9,10,11,12,13,14,15,16,17],\
                    9:[0,1,2,3,4,5,11,12,13,14,15,16,17],\
                    10:[0,1,2,3,4,5,11,12,13,14,15,16,17],\
                    11:[0,1,2,3,4,5,12,13,14,15,16,17],\
                    12:[0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17],\
                    13:[0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17],\
                    14:[0,1,2,3,4,5,6,7,8,9,10,11,15,16,17],\
                    15:[0,1,2,3,4,5,6,7,8,9,10,11,17],\
                    16:[0,1,2,3,4,5,6,7,8,9,10,11,17],\
                    17:[0,1,2,3,4,5,6,7,8,9,10,11],\
                    18:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]}
    # In the circular shift implementation of moving the face cubes
    #  this designates which face corresponds to which move
    #face B/yellow = 0; R/blue = 1; F/white = 2; L/green = 3; U/orange = 4; D/red = 5
    move2face = [5,5,5,4,4,4,1,1,1,3,3,3,2,2,2,0,0,0]
    #  These are the byte shifts needed for circular roll
    # give the number and direction of the roll needed to move face elements during a move
    move2shifts = [2,-2,4,-2,2,4,2,-2,4,-2,2,4,2,-2,4,-2,2,4]
    # not used. but given for reference to show indices into the face array that are
    #  the ordering of side faces for a face turn.
    face2sideseq = [[8,9,10,46,47,40,28,29,30,38,39,32],\
                    [4,5,6,32,33,34,16,17,18,44,45,46],\
                    [12,13,14,34,35,36,24,25,26,42,43,44],\
                    [0,1,2,40,41,42,20,21,22,36,37,38],\
                    [0,7,6,8,15,14,16,23,22,24,31,30],\
                    [2,3,4,10,11,12,18,19,20,26,27,28]]
    
    # internally the cubies are identified by a number and their orientation
    # The following URL provides pictures for which cubies are given which number
    # and which faces have which orientation
    #https://www.sfu.ca/~jtmulhol/math302/puzzles-rc-cubology.html
    # The only difference is instead of starting of the cubie numbering
    # for edges, they continue the counting from the corners. Thus,
    #  the corners are numbered 1-8 and edges 9-19
    # The ordering of the cubies is also how they are ordered internally 
    #  in order to keep faces in order on a face to perform the circular shift
    #   move
    facecodechars = ["c012","e051","c051","e091","c062","e061","c021","e011",\
                 "c022","e060","c061","e101","c072","e070","c031","e021",\
                 "c032","e071","c071","e111","c082","e081","c041","e031",\
                 "c042","e080","c081","e121","c052","e050","c011","e041",\
                 "c020","e020","c030","e030","c040","e040","c010","e010",\
                 "c050","e120","c080","e110","c070","e100","c060","e090"]
    facecodechardict = {"c012":0,"e051":1,"c051":2,"e091":3,"c062":4,"e061":5,"c021":6,"e011":7,\
                 "c022":8,"e060":9,"c061":10,"e101":11,"c072":12,"e070":13,"c031":14,"e021":15,\
                 "c032":16,"e071":17,"c071":18,"e111":19,"c082":20,"e081":21,"c041":22,"e031":23,\
                 "c042":24,"e080":25,"c081":26,"e121":27,"c052":28,"e050":29,"c011":30,"e041":31,\
                 "c020":32,"e020":33,"c030":34,"e030":35,"c040":36,"e040":37,"c010":38,"e010":39,\
                 "c050":40,"e120":41,"c080":42,"e110":43,"c070":44,"e100":45,"c060":46,"e090":47}
    # This is just the cube numbering 
    cubieids  = {"c01":0,"c02":1,"c03":2,"c04":3,"c05":4,"c06":5,"c07":6,"c08":7,\
                 "e01":8,"e02":9,"e03":10,"e04":11,"e05":12,"e06":13,\
                 "e07":14,"e08":15,"e09":16,"e10":17,"e11":18,"e12":19}
    # convert_facechar2int.py generates these the cubeids range from 0-19 and take
    #   5 bits.  The orients range from 0-2 and take 2 bits
    #  the facecideints are generated 7 total bits.  The highest/left bits
    #  are the 5 bits to identify cube and the lower/right 2 bits give orient
    #  For instance the 10th edge cube 'e10' is cube# 17 or in binary 10001
    #  10th edge cube face given orientation 1 'e101' has 1 in binary 01
    #  thus the 7 bit pattern id for this face is 1000101 = 69
    # facecodeints  is for a solved cube following the facecodechars ordering that is used internally
    facecodeints = np.array([2, 49, 17, 65, 22, 53, 5, 33, \
                    6, 52, 21, 69, 26, 56, 9, 37, \
                    10, 57, 25, 73, 30, 61, 13, 41,\
                    14, 60, 29, 77, 18, 48, 1, 45,\
                    4, 36, 8, 40, 12, 44, 0, 32,\
                    16, 76, 28, 72, 24, 68, 20, 64])
#                             0   1   2  3   4   5   6   7
#    facecodeints = np.array([2, 49, 17, 65, 22, 53, 5, 33, \
    
#                    8   9  10  11  12  13 14  15    
#                    6, 52, 21, 69, 26, 56, 9, 37, \

#                    16  17  18  19  20  21  22  23
#                    10, 57, 25, 73, 30, 61, 13, 41,\
    
#                    24  25  26  27  28  29 30  31    
#                    14, 60, 29, 77, 18, 48, 1, 45,\
    
#                    32 33 34  35  36  37 38  39    
#                    4, 36, 8, 40, 12, 44, 0, 32,\
    
#                    40  41  42  43  44  45  46  47    
#                    16, 76, 28, 72, 24, 68, 20, 64])

    # The next few variables are used to convert the colors given in begcubefaces
    # into the internal faceid code scheme
    facename_facecolors_dict = {"01my":5, "01mz":6, "01mx":4, "02my":5, "02mz":6,"03my":5, "03px":2, "03mz":6,\
                 "04mx":4, "04mz":6,"06px":2, "06mz":6,\
                 "07mx":4, "07py":3, "07mz":6,"08py":3, "08mz":6,"09px":2, "09py":3, "09mz":6,\
                 "10mx":4, "10my":5,"12px":2, "12my":5,\
                 "16mx":4, "16py":3,"18px":2, "18py":3,\
                 "19mx":4, "19my":5, "19pz":1,"20my":5, "20pz":1,"21px":2, "21my":5, "21pz":1,\
                 "22mx":4, "22pz":1,"24px":2,"24pz":1,\
                 "25mx":4, "25py":3, "25pz":1,"26py":3, "26pz":1,"27px":2, "27py":3, "27pz":1}

    facename_faceid_dict = {"01my":30, "01mz":28, "01mx":29, "02my":73, "02mz":72,"03my":25, "03px":26, "03mz":24,\
                 "04mx":77, "04mz":76,"06px":69, "06mz":68,\
                 "07mx":18, "07py":17, "07mz":16,"08py":65, "08mz":64,"09px":21, "09py":22, "09mz":20,\
                 "10mx":60, "10my":61,"12px":56, "12my":57,\
                 "16mx":48, "16py":49,"18px":52, "18py":53,\
                 "19mx":14, "19my":13, "19pz":12,"20my":41, "20pz":40,"21px":9, "21my":10, "21pz":8,\
                 "22mx":45, "22pz":44,"24px":37,"24pz":36,\
                 "25mx":1, "25py":2, "25pz":0,"26py":33, "26pz":32,"27px":6, "27py":5, "27pz":4}
    corner_list_names = [["01my","01mz","01mx"], ["03my","03px","03mz"],\
                       ["07mx","07py","07mz"], ["09px","09py","09mz"],\
                       ["19mx","19my","19pz"], ["21px","21my","21pz"],\
                       ["25mx","25py","25pz"], ["27px","27py","27pz"]]
    edge_list_names = [["02my","02mz"], ["04mx","04mz"], ["06px","06mz"],\
                     ["08py","08mz"], ["10mx","10my"], ["12px","12my"],\
                     ["16mx","16py"], ["18px","18py"], ["20my","20pz"],\
                     ["22mx","22pz"], ["24px","24pz"], ["26py","26pz"]]
    corner_list_ids = [[20, 42, 26], [18, 12, 44], [28, 2, 40], [10, 4, 46], [24, 22, 36], [14, 16, 34], [30, 0, 38], [8, 6, 32]]
    edge_list_ids = [[19, 43], [27, 41], [11, 45], [3, 47], [25, 21], [13, 17], [29, 1], [9, 5], [23, 35], [31, 37], [15, 33], [7, 39]]

    # These are important because they set which face is used in the lehmer
    #  coding and which order they are in
    corner_faces = np.array([42,44,40,46,36,34,38,32], dtype=np.int)
    edge_faces = np.array([43,41,45,47,25,13,29,9,35,37,33,39], dtype=np.int)
    
    # The getstate functions generate the lehmer code for a given
    #  cube configuration
    #  This first one is for the 8 corner cubies with orientations included
    def getstate(self, fc, lc):
        corner_faceids = fc[self.corner_faces] # Get the corner cubie faces
        corner_cubieids = np.right_shift(corner_faceids, 2) # This picks out the
                    # cube ids in the higher bits
        corner_orientids = np.bitwise_and(corner_faceids, 3)[0:-1] # This picks
                    # out the lowest two bits that encode orientations
        corner_cubie_index = lc.encode(corner_cubieids)
        corner_cubie_index = corner_cubie_index * 2187
        corner_orient_index = np.sum(corner_orientids * np.array([729,243,81,27,9,3,1], dtype=np.int))
        return corner_cubie_index + corner_orient_index # combine cubie and orientation
                                                    # to get final lehmer code

    # Lehmer code generation for 12 edges disregarding their orientations
    #  see getstate for more detailed comments
    def getstate_edge(self, fc, lc):
        edge_faceids = fc[self.edge_faces]
        edge_cubieids = np.right_shift(edge_faceids, 2) - 8 # -8 is here since
                        # we start the edge cubie ids at 8 and lehmer code
                        # requires the permutations to start at zero
        edge_cubie_index = lc.encode(edge_cubieids)
        return edge_cubie_index
    # see getstate
    # This is for lehmer code for 7 of 12 edges including orientations
    def getstate_edgesplit(self, fc, lc, frsti):
        edge_faceids = fc[self.edge_faces][frsti::2] # nominally select
                                    # the odd for one set of 6 edges
                                    # and even for the other set of 6 edges
        # 12 pick 7
        #  add one edge cube to each set
        if frsti == 0:
            edge_faceids = np.append(edge_faceids,fc[41])
        else:
            edge_faceids = np.append(edge_faceids,fc[43])
        edge_cubieids = np.right_shift(edge_faceids, 2) - 8 # -8 here to set 
                                # cube ids to start at zero for edges
        edge_orientids = np.bitwise_and(edge_faceids, 3)
        edge_cubie_index = lc.encode(edge_cubieids)
        # 12 pick 6
        #edge_cubie_index = edge_cubie_index * 64
        #edge_orient_index = np.sum(edge_orientids * np.array([32,16,8,4,2,1], dtype=np.int))
        # 12 pick 7
        edge_cubie_index = edge_cubie_index * 128
        edge_orient_index = np.sum(edge_orientids * np.array([64,32,16,8,4,2,1], dtype=np.int))
        return edge_cubie_index + edge_orient_index

    # used to read the input dictionaries
    #  and facilitate matching of color faces with the
    #  internal face id codes
    def list_cubies(self, facedict):
        corner_faces = []
        for cl in self.corner_list_names:
            tmp = []
            for i in range(3):
                tmp.append(facedict[cl[i]])
            corner_faces.append(copy.copy(tmp))
        edge_faces = []
        for el in self.edge_list_names:
            tmp = []
            for i in range(2):
                tmp.append(facedict[el[i]])
            edge_faces.append(copy.copy(tmp))
        return corner_faces, edge_faces
    
    # This does the conversion from input cube color dictionary
    #  to the internal faceid numbers
    # The function also makes sure each cube is present
    #  It exits if it can't find all the cubes once
    #   which is likely to happen if the user has a typo
    #   in teh color input dictionary which is easy to do
    def get_start_faceids(self, facedict):
        # From a starting configuration assign the faceids to the 48 faces
        # in other words convert my human 01my:color, 01mz:color, ...
        # assignment to the faceid integers which encode the cube number and orientation
        outfaceids = np.zeros((48,), dtype=np.int)
        cf, ef = self.list_cubies(facedict) # These are the colors for corner and edge cubes at start
        cf2, ef2 = self.list_cubies(self.facename_facecolors_dict) # Theses are the colors for corner and edge cubes in solved cube
        cf3, ef3 = self.list_cubies(self.facename_faceid_dict) # There are the faceids "
        cfoundcount = np.zeros((8,), dtype=np.int)
        efoundcount = np.zeros((12,), dtype=np.int)
        # Start with corners
        for ii, curc in enumerate(cf):
            notFound = True
            i = 0
            while notFound and i < 8:
                check_values = cf2[i]
                if set(check_values) == set(curc): # the ordering can be
                                    # different thus we compare unordered
                                    #  sets rather than require ordered match
                    iFound = i
                    notFound = False
                    cfoundcount[iFound] = cfoundcount[iFound] + 1
                    for j in range(3):
                        idx = np.where(np.array(check_values) == curc[j])[0][0]
                        outfaceids[self.corner_list_ids[ii][j]] = cf3[iFound][idx]
                i = i+1
            if notFound:
                print('Could not find the input corner cube in the end state cube! Exiting')
                exit()
        # Ensure each corner was seen once and only oncde
        idx = np.where(cfoundcount == 1)[0]
        if not len(idx) == 8:
            print('Not all corners were found only once! Exiting')
            print(cfoundcount)
            exit()
        # Now do edges
        for ii, cure in enumerate(ef):
            notFound = True
            i = 0
            while notFound and i < 12:
                check_values = ef2[i]
                if set(check_values) == set(cure):
                    iFound = i
                    notFound = False
                    efoundcount[iFound] = efoundcount[iFound] + 1
                    for j in range(2):
                        idx = np.where(np.array(check_values) == cure[j])[0][0]
                        outfaceids[self.edge_list_ids[ii][j]] = ef3[iFound][idx]
                i = i+1
            if notFound:
                print('Could not find the input edge cube in the end state cube! Exiting')
                exit()
        # Ensure each edge was seen once and only oncde
        idx = np.where(efoundcount == 1)[0]
        if not len(idx) == 12:
            print('Not all edges were found only once! Exiting')
            print(efoundcount)
            exit()

        return outfaceids

    # This is where the faces are moved corresponding
    #  to the move_id integer
    #  This uses a circular shift to roll the faceids
    #  Thus in principle enabling a mvoe and store with single copy
    #  rather than 8 separate moves and copies
    # It also moves the faces along the side of a face
    #  See BottoB for discussion
    def roll_move(self, fc, move_id):
        # make copy not reference 
        newfc = list(fc)
        # do the roll move for this face
        doface = self.move2face[move_id]
        # each face has an id and that translates into the 8 faces
        # calculate the indicies to the start and end of these 8 faces
        strt_idx = doface*8
        end_idx = (doface+1)*8
        # Do the roll shifting by a number of bytes
        # BottoB does this with an assembly intrinsic in c++
        #  this is not that, but standard numpy function
        newfc[strt_idx:end_idx] = np.roll(fc[strt_idx:end_idx], self.move2shifts[move_id]).tolist()
        # Now do the moves for the faces on the side of the moving face
        newfc = self.side_move_switch(move_id, fc, newfc)
        return newfc

    # Each move has a separate function for moving the side 
    #  faces.  The side move functions are stored in a dicitonary
    #  and the side_move_switch() will call the one for the move_id
    #  The principle is that for faces that neighbor each other
    #  they can be moved at once and taking advantage of a wider
    #  integer. Here we use slice assignment. This is done on the
    #  python side
    #  to prototype and get all the indices correct.
    # Here are all the side face assignments for the moves
    # overall side sequence for D/red face moves
    # x      x        x        x
    #[2,3,4,10,11,12,18,19,20,26,27,28]   
    def side_move_DR00(self, fc, newfc):
        newfc[26:28] = fc[2:4]
        newfc[28] = fc[4]
        newfc[2:4] = fc[10:12]
        newfc[4] = fc[12]
        newfc[10:12] = fc[18:20]
        newfc[12] = fc[20]
        newfc[18:20] = fc[26:28]
        newfc[20] = fc[28]
        return newfc
    def side_move_DL01(self, fc, newfc):
        newfc[10:12] = fc[2:4]
        newfc[12] = fc[4]
        newfc[18:20] = fc[10:12]
        newfc[20] = fc[12]
        newfc[26:28] = fc[18:20]
        newfc[28] = fc[20]
        newfc[2:4] = fc[26:28]
        newfc[4] = fc[28]
        return newfc
    def side_move_DH02(self, fc, newfc):
        newfc[18:20] = fc[2:4]
        newfc[20] = fc[4]
        newfc[26:28] = fc[10:12]
        newfc[28] = fc[12]
        newfc[2:4] = fc[18:20]
        newfc[4] = fc[20]
        newfc[10:12] = fc[26:28]
        newfc[12] = fc[28]
        return newfc
    # overall side sequence for U/orange face moves
    #      x       x        x         x
    # [0,7,6,8,15,14,16,23,22,24,31,30]
    def side_move_UR03(self, fc, newfc):
        newfc[30:32] = fc[6:8]
        newfc[24] = fc[0]
        newfc[6:8] = fc[14:16]
        newfc[0] = fc[8]
        newfc[14:16] = fc[22:24]
        newfc[8] = fc[16]
        newfc[22:24] = fc[30:32]
        newfc[16] = fc[24]
        return newfc
    def side_move_UL04(self, fc, newfc):
        newfc[14:16] = fc[6:8]
        newfc[8] = fc[0]
        newfc[22:24] = fc[14:16]
        newfc[16] = fc[8]
        newfc[30:32] = fc[22:24]
        newfc[24] = fc[16]
        newfc[6:8] = fc[30:32]
        newfc[0] = fc[24]
        return newfc
    def side_move_UH05(self, fc, newfc):
        newfc[22:24] = fc[6:8]
        newfc[16] = fc[0]
        newfc[30:32] = fc[14:16]
        newfc[24] = fc[8]
        newfc[6:8] = fc[22:24]
        newfc[0] = fc[16]
        newfc[14:16] = fc[30:32]
        newfc[8] = fc[24]
        return newfc
    # overall side sequence for R/Blue face
    #  x     x        x        x    
    # [4,5,6,32,33,34,16,17,18,44,45,46]
    def side_move_RU06(self, fc, newfc):
        newfc[44:46] = fc[4:6]
        newfc[46] = fc[6]
        newfc[4:6] = fc[32:34]
        newfc[6] = fc[34]
        newfc[32:34] = fc[16:18]
        newfc[34] = fc[18]
        newfc[16:18] = fc[44:46]
        newfc[18] = fc[46]
        return newfc
    def side_move_RD07(self, fc, newfc):
        newfc[32:34] = fc[4:6]
        newfc[34] = fc[6]
        newfc[16:18] = fc[32:34]
        newfc[18] = fc[34]
        newfc[44:46] = fc[16:18]
        newfc[46] = fc[18]
        newfc[4:6] = fc[44:46]
        newfc[6] = fc[46]
        return newfc
    def side_move_RH08(self, fc, newfc):
        newfc[16:18] = fc[4:6]
        newfc[18] = fc[6]
        newfc[44:46] = fc[32:34]
        newfc[46] = fc[34]
        newfc[4:6] = fc[16:18]
        newfc[6] = fc[18]
        newfc[32:34] = fc[44:46]
        newfc[34] = fc[46]
        return newfc
    #overall side sequence for L/Green face
    # x     x        x        x
    #[0,1,2,40,41,42,20,21,22,36,37,38]
    def side_move_LU09(self, fc, newfc):
        newfc[40:42] = fc[0:2]
        newfc[42] = fc[2]
        newfc[20:22] = fc[40:42]
        newfc[22] = fc[42]
        newfc[36:38] = fc[20:22]
        newfc[38] = fc[22]
        newfc[0:2] = fc[36:38]
        newfc[2] = fc[38]
        return newfc
    def side_move_LD10(self, fc, newfc):
        newfc[36:38] = fc[0:2]
        newfc[38] = fc[2]
        newfc[0:2] = fc[40:42]
        newfc[2] = fc[42]
        newfc[40:42] = fc[20:22]
        newfc[42] = fc[22]
        newfc[20:22] = fc[36:38]
        newfc[22] = fc[38]
        return newfc
    def side_move_LH11(self, fc, newfc):
        newfc[20:22] = fc[0:2]
        newfc[22] = fc[2]
        newfc[36:38] = fc[40:42]
        newfc[38] = fc[42]
        newfc[0:2] = fc[20:22]
        newfc[2] = fc[22]
        newfc[40:42] = fc[36:38]
        newfc[42] = fc[38]
        return newfc
    #overall side sequence for F/White face
    # x        x        x        x   
    # 12,13,14,34,35,36,24,25,26,42,43,44
    def side_move_FC12(self, fc, newfc):
        newfc[42:44] = fc[12:14]
        newfc[44] = fc[14]
        newfc[12:14] = fc[34:36]
        newfc[14] = fc[36]
        newfc[34:36] = fc[24:26]
        newfc[36] = fc[26]
        newfc[24:26] = fc[42:44]
        newfc[26] = fc[44]
        return newfc
    def side_move_FG13(self, fc, newfc):
        newfc[34:36] = fc[12:14]
        newfc[36] = fc[14]
        newfc[24:26] = fc[34:36]
        newfc[26] = fc[36]
        newfc[42:44] = fc[24:26]
        newfc[44] = fc[26]
        newfc[12:14] = fc[42:44]
        newfc[14] = fc[44]
        return newfc
    def side_move_FH14(self, fc, newfc):
        newfc[24:26] = fc[12:14]
        newfc[26] = fc[14]
        newfc[42:44] = fc[34:36]
        newfc[44] = fc[36]
        newfc[12:14] = fc[24:26]
        newfc[14] = fc[26]
        newfc[34:36] = fc[42:44]
        newfc[36] = fc[44]
        return newfc
    # overall side sequence for B/yellow face
    # x      x        x        x
    # 8,9,10,46,47,40,28,29,30,38,39,32
    def side_move_BC15(self, fc, newfc):
        newfc[46:48] = fc[8:10]
        newfc[40] = fc[10]
        newfc[28:30] = fc[46:48]
        newfc[30] = fc[40]
        newfc[38:40] = fc[28:30]
        newfc[32] = fc[30]
        newfc[8:10] = fc[38:40]
        newfc[10] = fc[32]
        return newfc
    def side_move_BG16(self, fc, newfc):
        newfc[38:40] = fc[8:10]
        newfc[32] = fc[10]
        newfc[8:10] = fc[46:48]
        newfc[10] = fc[40]
        newfc[46:48] = fc[28:30]
        newfc[40] = fc[30]
        newfc[28:30] = fc[38:40]
        newfc[30] = fc[32]
        return newfc
    def side_move_BH17(self, fc, newfc):
        newfc[28:30] = fc[8:10]
        newfc[30] = fc[10]
        newfc[38:40] = fc[46:48]
        newfc[32] = fc[40]
        newfc[8:10] = fc[28:30]
        newfc[10] = fc[30]
        newfc[46:48] = fc[38:40]
        newfc[40] = fc[32]
        return newfc

    # This is a 'switch' based function wrapper
    #  that specifies which side move function to call based on move id    
    def side_move_switch(self, move_id, fc, newfc):
        func_switch={
                0:self.side_move_DR00, 1:self.side_move_DL01, 2:self.side_move_DH02,\
                3:self.side_move_UR03, 4:self.side_move_UL04, 5:self.side_move_UH05,\
                6:self.side_move_RU06, 7:self.side_move_RD07, 8:self.side_move_RH08,\
                9:self.side_move_LU09, 10:self.side_move_LD10, 11:self.side_move_LH11,\
                12:self.side_move_FC12, 13:self.side_move_FG13, 14:self.side_move_FH14,\
                15:self.side_move_BC15, 16:self.side_move_BG16, 17:self.side_move_BH17                
                }
        func = func_switch.get(move_id)
        return func(fc, newfc)
        
    # make all the subsequent moves that are allowed 
    #  where we prune moves that are redundant using self.ignore_moves
    def make_pathlist(self, fc, lastMove):
        tmp = dq([])
        for i in self.ignore_moves[lastMove]:
            curfc = self.roll_move(fc, i)
            tmp.append([curfc, i])
        return tmp

# This is the worker/child that will perform
# the search from the initial cube configuration its given
def child(inarr):
    #print('Starting Move: {0:d}'.format(inarr[0]))
    tmpfc = inarr[3:] # stores the faceids for the cube configuration
    # Get references to the databases that are in shared memory
    cornerDB = np.frombuffer(patternDB_Storage[0], dtype=np.int16)
    edgeDB = np.frombuffer(patternDB_Storage[2], dtype=np.int16)
    edge1DB = np.frombuffer(patternDB_Storage[4], dtype=np.int16)
    edge2DB = np.frombuffer(patternDB_Storage[6], dtype=np.int16)
    # The maximum level to search
    maxlev_v = inarr[1]
    #  The current level of the search
    curlev_v = inarr[2]
    # The last two moves that resulted in this cube configuration
    #  are stored as a single integer. first move is in ones and 10s
    #  digit and second move is in the 1000 and 100s digit
    cmv_v = inarr[0]
    # Need to decode the current move number depending on level
    clev3 = cmv_v //100
    clev2 = cmv_v - clev3*100
    if curlev_v == 3: # This is if weve done two moves
        usecmv = clev3
    if curlev_v == 2: # This is left over from when we did a single move
        usecmv = clev2
    # This is the main worker call to look from a solution from this
    #  cube configuration
    retval = rcmMP.DFS_cython_solve(bytes(tmpfc), maxlev_v, curlev_v, usecmv, cornerDB, edgeDB, edge1DB, edge2DB)
    # retval == 2 indicates the worker found a solution
    if retval == 2:
        # convert the move integers into character moves
        #  using my non standard move nomenclature and the more standard one
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

        if curlev_v == 3:
            print('Root moves where solution was found: {0:d}, {1:d}'.format(clev2, clev3))
            print('{0} {1}'.format(rotnames[clev2], rotnames[clev3]))
            print('{0} {1}'.format(char_move_dict[clev2], char_move_dict[clev3]))
        if curlev_v == 2:
            print('Root move where solution was found:{0:d}'.format(clev2))
        # Show elapsed time to solution
        print('Elapsed time for solution including setup time (s) {0:.1f}'.format(timer()-startts))

    return retval

# module level pointers for pattern db
patternDB_Storage=[]
# start time to keep track of elapsed run time
startts = timer()

# The method for terminating the worker children processes
#  after one of them finds a solution is from
# https://stackoverflow.com/questions/36962462/terminate-a-python-multiprocessing-program-once-a-one-of-its-workers-meets-a-cer
# https://stackoverflow.com/questions/34827250/how-to-keep-track-of-status-with-multiprocessing-and-pool-map
def log_quitter(retval):
    #print('Got retval: {0:d}'.format(retval))
    results.append(retval)
    if retval == 2: # a worker found a solution
        pmp.terminate() # Kill all pool workers if one returns a solution

if __name__ == '__main__':
    
    # Set N cores you want to use
    # default is all of them found by multiprocessing.cpu_count()
    print('Found {0:d} CPUS'.format(cpu_count()))
    USENCPUS = cpu_count()
    # See the README.md for the nomenclature for entering the scrambled
    #  cube that you want to solve. solvedfaces is the solved cube
    #  This veriable isn't used it is just here for reference

    # These are the integers that correspond to which color 
    # 1 - orange, 2 - blue, 3 - yellow, 4 - green, 5 - white, 6 - red
    solvedfaces = {"01my":5, "01mz":6, "01mx":4, "02my":5, "02mz":6,"03my":5, "03px":2, "03mz":6,\
                 "04mx":4, "04mz":6,"05mz":6,"06px":2, "06mz":6,\
                 "07mx":4, "07py":3, "07mz":6,"08py":3, "08mz":6,"09px":2, "09py":3, "09mz":6,\
                 "10mx":4, "10my":5,"11my":5,"12px":2, "12my":5,\
                 "13mx":4,"15px":2,\
                 "16mx":4, "16py":3,"17py":3,"18px":2, "18py":3,\
                 "19mx":4, "19my":5, "19pz":1,"20my":5, "20pz":1,"21px":2, "21my":5, "21pz":1,\
                 "22mx":4, "22pz":1,"23pz":1,"24px":2, "24pz":1,\
                 "25mx":4, "25py":3, "25pz":1,"26py":3, "26pz":1,"27px":2, "27py":3, "27pz":1}
    # HERE is where you put the cube you want to solve in begcubefaces dictionary
    # The standard is to have the white center cube face you, orange center 
    #   cube on top and blue center cube to the right
    # 15 turns
    begcubefaces = {"01my":6, "01mz":5, "01mx":2, "02my":6, "02mz":4,"03my":3, "03px":6, "03mz":2,\
                 "04mx":2, "04mz":5,"05mz":6,"06px":6, "06mz":2,\
                 "07mx":5, "07py":2, "07mz":1,"08py":6, "08mz":3,"09px":1, "09py":3, "09mz":2,\
                 "10mx":3, "10my":4,"11my":5,"12px":3, "12my":1,\
                 "13mx":4,"15px":2,\
                 "16mx":2, "16py":1,"17py":3,"18px":6, "18py":5,\
                 "19mx":6, "19my":5, "19pz":4,"20my":1, "20pz":5,"21px":5, "21my":4, "21pz":1,\
                 "22mx":1, "22pz":4,"23pz":1,"24px":2, "24pz":3,\
                 "25mx":6, "25py":3, "25pz":4,"26py":5, "26pz":4,"27px":4, "27py":1, "27pz":3}
    # empty template to use for filling in your own cube face colors/numbers
#    begcubefaces = {"01my":, "01mz":, "01mx":, "02my":, "02mz":,"03my":, "03px":, "03mz":,\
#                 "04mx":, "04mz":,"05mz":6,"06px":, "06mz":,\
#                 "07mx":, "07py":, "07mz":,"08py":, "08mz":,"09px":, "09py":, "09mz":,\
#                 "10mx":, "10my":,"11my":5,"12px":, "12my":,\
#                 "13mx":4,"15px":2,\
#                 "16mx":, "16py":,"17py":3,"18px":, "18py":,\
#                 "19mx":, "19my":, "19pz":,"20my":, "20pz":,"21px":, "21my":, "21pz":,\
#                 "22mx":, "22pz":,"23pz":1,"24px":, "24pz":,\
#                 "25mx":, "25py":, "25pz":,"26py":, "26pz":,"27px":, "27py":, "27pz":}

    # This is the internal integer name for a move with my non-standard
    #  character code for the move.  See README.md for description
#    rotnames = {0:"DR", 1:"DL", 2:"DH",\
#                3:"UR", 4:"UL", 5:"UH",\
#                6:"RU", 7:"RD", 8:"RH",\
#                9:"LU", 10:"LD", 11:"LH",\
#                12:"FC", 13:"FG", 14:"FH",\
#                15:"BC", 16:"BG", 17:"BH"}
    
    # initialize cube
    bcube = rubiks_cube()
    # Read in the cube color dictionary and convert it to the 
    # id and orientation for the cubies that are used internally
    init_faceids = bcube.get_start_faceids(begcubefaces).tolist()
    # if you want to hard code the internal cubie ids generated by 
    #  rubik_cube_debugpath_roll.py you can enter it here
    #  to bypass what is in the color dictionary
    # init_faceids = [9, 32, 29, 68, 13, 41, 24, 76, 25, 40, 12, 61, 21, 64, 1, 52, 2, 65, 20, 72, 6, 36, 18, 44, 16, 37, 5, 56, 30, 33, 8, 49, 26, 53, 0, 45, 17, 48, 10, 77, 28, 57, 4, 73, 22, 60, 14, 69]

    print('Start Loading Pattern DBs')
    # Load the corner config to solve turns DB
    with np.load('rubik_corner_db.npz') as data:
        cornerDB = data['db']
    # Fix -1 score for solved state
    idx = np.argmin(cornerDB)
    cornerDB[idx] = 0
    # Load the edge config to solve turns DB
    with np.load('rubik_alledge_db.npz') as data:
        edgeDB = data['db']
    # Fix -1 score for solved state
    idx = np.argmin(edgeDB)
    edgeDB[idx] = 0
    # Load the edge1 config to solve turns DB
    with np.load('rubik_edge1_DFS_12p7_db.npz') as data:
        edge1DB = data['db']
    # Fix -1 score for solved state
    idx = np.argmin(edge1DB)
    edge1DB[idx] = 0
    # Load the edge2 config to solve turns DB
    with np.load('rubik_edge2_DFS_12p7_db.npz') as data:
        edge2DB = data['db']
    # Fix -1 score for solved state
    idx = np.argmin(edge2DB)
    edge2DB[idx] = 0
    
    # Make shared raw arrays of pattern databases
    print('Start making DBs shared')
    # Note that the 'h' designation in RawArray was used because
    #   'i' was too big for the np.int16.However, multiple times reading
    #  the documentation seemed to me that 'i' should have worked for int16 as well (2 bytes), but it didn't
    #  this disagreement in element size may break on other computers or implementations.
    # make shared db storage
    cshDB = RawArray('h', cornerDB.shape[0])
    # make the numpy wrapper to this buffer
    cshDB_np = np.frombuffer(cshDB, dtype=np.int16)
    # now copy data into the shared storage
    np.copyto(cshDB_np, cornerDB.astype(np.int16))
    # repeat for other databases
    eshDB = RawArray('h', edgeDB.shape[0])
    eshDB_np = np.frombuffer(eshDB, dtype=np.int16)
    np.copyto(eshDB_np, edgeDB.astype(np.int16))
    e1shDB = RawArray('h', edge1DB.shape[0])
    e1shDB_np = np.frombuffer(e1shDB, dtype=np.int16)
    np.copyto(e1shDB_np, edge1DB.astype(np.int16))
    e2shDB = RawArray('h', edge2DB.shape[0])
    e2shDB_np = np.frombuffer(e2shDB, dtype=np.int16)
    np.copyto(e2shDB_np, edge2DB.astype(np.int16))
    patternDB_Storage.extend([cshDB, cornerDB.shape[0], \
                              eshDB, edgeDB.shape[0],\
                              e1shDB, edge1DB.shape[0],\
                              e2shDB, edge2DB.shape[0]])
    print('Done copying pattern db to shared memory')
    print('Elapsed time for setup (s) {0:.1f}'.format(timer()-startts))
    # Calculate the Lehmer Get the initial cube distance
    lehcode = lc.lehmer_code(8)
    statecode = bcube.getstate(np.array(init_faceids), lehcode)
    edge_lehcode = lc.lehmer_code(12)
    edge_statecode = bcube.getstate_edge(np.array(init_faceids), edge_lehcode)
    edge1_lehcode = lc.lehmer_code(12, 7)
    edge1_statecode = bcube.getstate_edgesplit(np.array(init_faceids), edge1_lehcode, 0)
    edge2_lehcode = lc.lehmer_code(12, 7)
    edge2_statecode = bcube.getstate_edgesplit(np.array(init_faceids), edge2_lehcode, 1)
    print(statecode, edge_statecode, edge1_statecode, edge2_statecode)
    # Based on the lehmer code look up the moves until end for each database
    cs = cornerDB[statecode]
    ce = edgeDB[edge_statecode]
    ce1 = edge1DB[edge1_statecode]
    ce2 = edge2DB[edge2_statecode]
    score = np.max([cs, ce, ce1, ce2])
    print('Max & Initial Scores')
    print(score, cs, ce, ce1, ce2)
    # Since cubes are always solvable in <=20 moves
    # this is the largets depth from the initial score we need to explore
    largest_MAXDELDEP = 20 - score


    retval = 0
    # The first few are so quick that don't bother with MP
    # Here is where we call the Iterative Depth Depth First search
    # MAXDELDEP sets the maximum depth we search each iteration
    #  the first few are single core.
    for MAXDELDEP in np.arange(0,5):
        useMaxLevel = MAXDELDEP +score
        if not retval == 2: # Found solution yet?
            print('Trying MAXDELDEP {0:d} MaxLevel: {1:d}'.format(MAXDELDEP, useMaxLevel))
            # Call the DFS cython that does all the work to MAXDELDEP
            retval = rcm.DFS_cython_solve(bytes(init_faceids), MAXDELDEP, cornerDB, edgeDB, edge1DB, edge2DB)
    print('Now Trying MP for larger rounds')
    # Save the cube states after the first set of turns
    # newmoves holds the facieds after the first 18 turns
    newmoves = bcube.make_pathlist(init_faceids, 18)
    # go to another level 2 turns starting from the first turns
    newmoves2 = []
    for i in range(18):
        new_faceids = newmoves[i][0]
        exmoves = bcube.make_pathlist(new_faceids, i)
        # we need to keep track of the two moves we do this by
        # Adjusting the move number to put the second move
        #  by multiplying second move by 100 and adding to first move
        for ex in exmoves:
            curmv = ex[1]*100
            ex[1] = curmv +i
            newmoves2.append(ex)
    print("Got {0:d} number of moves after 2nd level".format(len(newmoves2)))
    # Here is where we go to even deeper IDDFS searches but using Multiprocessing
    for MAXDELDEP in np.arange(5,largest_MAXDELDEP+1):
        useMaxLevel = useMaxLevel + 1
        if not retval == 2: #Found Solution yet?
            print('Trying MAXDELDEP {0:d} MaxLevel:{1:d} with MP'.format(MAXDELDEP, useMaxLevel))
            # pack the worker arguments
            work_args = []
            for i, curnewmoves in enumerate(newmoves2):
                curlevel = 3
                cmv = curnewmoves[1]
                holdlist = [cmv, useMaxLevel, curlevel]
                holdlist.extend(curnewmoves[0])
                work_args.append(holdlist)

            # Have all the worker arguments loaded 
            # initialize the pool of workes                
            pmp = Pool(processes = USENCPUS)

            results = [] # This will store results
                         # This gets populated in log_quitter() callback function
                         # callback is in scope of main so it is visible
            # Fill the wokeres with all the jobs
            for i in range(len(work_args)):
                pmp.apply_async(child, args=(work_args[i],), callback=log_quitter)
            # close the pool for any future jobs
            pmp.close()
            #  Block until all the workers finished or are terminated
            pmp.join()
            # Go through the results list to see if any workers found a solution
            fndSoln = False
            for rr in results:
                if rr == 2: # Worker finds solution!
                    fndSoln = True
            if fndSoln:
                retval = 2 # This terminates going to higher levels in the IDFFS search
