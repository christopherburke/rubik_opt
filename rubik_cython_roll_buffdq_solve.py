#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 23:32:28 2020

@author: cjburke
Solve the rubik's cube with a Depth First Search
With a precalculated corner database of minimum length to solve 
  for any corner configuration.  This means we can search until the depth
  reaches current depth + min length to solve corner depth == 20,
  then stop this path.  This is because all cubes can be solved in 20 or fewer
  moves in the half turn basis
  
  kernprof -l -v rubik_cube_turnonly_rollcy.py
"""

import numpy as np
import rubik_cython_roll_buffdq_solve as rcm
import copy
import lehmer_code as lc
# Apply the `profile` decorator
#rcm.DFS_cython_moveonly = profile(rcm.DFS_cython_moveonly)

class rubiks_cube():

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
    #face B/yellow = 0; R/blue = 1; F/white = 2; L/green = 3; U/orange = 4; D/red = 5
    move2face = [5,5,5,4,4,4,1,1,1,3,3,3,2,2,2,0,0,0]
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
                
    #https://www.sfu.ca/~jtmulhol/math302/puzzles-rc-cubology.html
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


    corner_faces = np.array([42,44,40,46,36,34,38,32], dtype=np.int)
    edge_faces = np.array([43,41,45,47,25,13,29,9,35,37,33,39], dtype=np.int)
    
    def getstate(self, fc, lc):
        corner_faceids = fc[self.corner_faces]
        corner_cubieids = np.right_shift(corner_faceids, 2)
        corner_orientids = np.bitwise_and(corner_faceids, 3)[0:-1]
        corner_cubie_index = lc.encode(corner_cubieids)
        corner_cubie_index = corner_cubie_index * 2187
        corner_orient_index = np.sum(corner_orientids * np.array([729,243,81,27,9,3,1], dtype=np.int))
        return corner_cubie_index + corner_orient_index

    def getstate_edge(self, fc, lc):
        edge_faceids = fc[self.edge_faces]
        edge_cubieids = np.right_shift(edge_faceids, 2) - 8
        edge_cubie_index = lc.encode(edge_cubieids)
        return edge_cubie_index

    def getstate_edgesplit(self, fc, lc, frsti):
        edge_faceids = fc[self.edge_faces][frsti::2]
        # 12 pick 7
        if frsti == 0:
            edge_faceids = np.append(edge_faceids,fc[41])
        else:
            edge_faceids = np.append(edge_faceids,fc[43])
        edge_cubieids = np.right_shift(edge_faceids, 2) - 8
        edge_orientids = np.bitwise_and(edge_faceids, 3)
        edge_cubie_index = lc.encode(edge_cubieids)
        # 12 pick 6
        #edge_cubie_index = edge_cubie_index * 64
        #edge_orient_index = np.sum(edge_orientids * np.array([32,16,8,4,2,1], dtype=np.int))
        # 12 pick 7
        edge_cubie_index = edge_cubie_index * 128
        edge_orient_index = np.sum(edge_orientids * np.array([64,32,16,8,4,2,1], dtype=np.int))
        return edge_cubie_index + edge_orient_index

    
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
                if set(check_values) == set(curc):
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

if __name__ == '__main__':
    

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

    solvedfaces = {"01my":5, "01mz":6, "01mx":4, "02my":5, "02mz":6,"03my":5, "03px":2, "03mz":6,\
                 "04mx":4, "04mz":6,"05mz":6,"06px":2, "06mz":6,\
                 "07mx":4, "07py":3, "07mz":6,"08py":3, "08mz":6,"09px":2, "09py":3, "09mz":6,\
                 "10mx":4, "10my":5,"11my":5,"12px":2, "12my":5,\
                 "13mx":4,"15px":2,\
                 "16mx":4, "16py":3,"17py":3,"18px":2, "18py":3,\
                 "19mx":4, "19my":5, "19pz":1,"20my":5, "20pz":1,"21px":2, "21my":5, "21pz":1,\
                 "22mx":4, "22pz":1,"23pz":1,"24px":2, "24pz":1,\
                 "25mx":4, "25py":3, "25pz":1,"26py":3, "26pz":1,"27px":2, "27py":3, "27pz":1}
    # 12 moves FC_RU_FC_RU_UL_RU_UL_RU_BG_LD_BG_LD
#    begcubefaces = {"01my":4, "01mz":5, "01mx":6, "02my":6, "02mz":2,"03my":4, "03px":3, "03mz":1,\
#                 "04mx":4, "04mz":5,"05mz":6,"06px":3, "06mz":1,\
#                 "07mx":1, "07py":2, "07mz":3,"08py":6, "08mz":4,"09px":6, "09py":2, "09mz":3,\
#                 "10mx":6, "10my":5,"11my":5,"12px":1, "12my":5,\
#                 "13mx":4,"15px":2,\
#                 "16mx":2, "16py":5,"17py":3,"18px":4, "18py":3,\
#                 "19mx":2, "19my":5, "19pz":6,"20my":2, "20pz":3,"21px":1, "21my":5, "21pz":4,\
#                 "22mx":4, "22pz":1,"23pz":1,"24px":2, "24pz":1,\
#                 "25mx":1, "25py":2, "25pz":5,"26py":3, "26pz":6,"27px":4, "27py":3, "27pz":6}
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
    # 4 moves FC_RU_BG_LD reverse LU_BC_RD_FG 9_15_7_13
    # 1 - orange, 2 - blue, 3 - yellow, 4 - green, 5 - white, 6 - red
#    begcubefaces = {"01my":4, "01mz":5, "01mx":6, "02my":5, "02mz":2,"03my":6, "03px":2, "03mz":3,\
#                 "04mx":6, "04mz":5,"05mz":6,"06px":2, "06mz":3,\
#                 "07mx":6, "07py":2, "07mz":5,"08py":3, "08mz":4,"09px":6, "09py":3, "09mz":4,\
#                 "10mx":4, "10my":1,"11my":5,"12px":2, "12my":6,\
#                 "13mx":4,"15px":2,\
#                 "16mx":4, "16py":6,"17py":3,"18px":6, "18py":3,\
#                 "19mx":5, "19my":1, "19pz":4,"20my":5, "20pz":4,"21px":1, "21my":2, "21pz":5,\
#                 "22mx":1, "22pz":3,"23pz":1,"24px":1, "24pz":5,\
#                 "25mx":1, "25py":4, "25pz":3,"26py":1, "26pz":2,"27px":3, "27py":1, "27pz":2}

    # 14 turns DR UR RU LU FC BC DR UR RU LU FC BC DR UR
#    begcubefaces = {"01my":4, "01mz":6, "01mx":3, "02my":6, "02mz":2,"03my":4, "03px":5, "03mz":6,\
#                 "04mx":3, "04mz":6,"05mz":6,"06px":5, "06mz":6,\
#                 "07mx":3, "07py":2, "07mz":6,"08py":1, "08mz":2,"09px":5, "09py":2, "09mz":6,\
#                 "10mx":3, "10my":4,"11my":5,"12px":5, "12my":4,\
#                 "13mx":4,"15px":2,\
#                 "16mx":3, "16py":2,"17py":3,"18px":5, "18py":2,\
#                 "19mx":3, "19my":4, "19pz":1,"20my":6, "20pz":4,"21px":5, "21my":4, "21pz":1,\
#                 "22mx":3, "22pz":1,"23pz":1,"24px":5, "24pz":1,\
#                 "25mx":3, "25py":2, "25pz":1,"26py":1, "26pz":4,"27px":5, "27py":2, "27pz":1}

#    rotnames = {0:"DR", 1:"DL", 2:"DH",\
#                3:"UR", 4:"UL", 5:"UH",\
#                6:"RU", 7:"RD", 8:"RH",\
#                9:"LU", 10:"LD", 11:"LH",\
#                12:"FC", 13:"FG", 14:"FH",\
#                15:"BC", 16:"BG", 17:"BH"}
    
    # initialize cube
    bcube = rubiks_cube()
    # This is the solved init_faceids
    #init_faceids = [2, 49, 17, 65, 22, 53, 5, 33, \
    #                6, 52, 21, 69, 26, 56, 9, 37, \
    #                10, 57, 25, 73, 30, 61, 13, 41,\
    #                14, 60, 29, 77, 18, 48, 1, 45,\
    #                4, 36, 8, 40, 12, 44, 0, 32,\
    #                16, 76, 28, 72, 24, 68, 20, 64]
    #begcubefaces = solvedfaces
    init_faceids = bcube.get_start_faceids(begcubefaces).tolist()
    #print(init_faceids)

    # OVERRIDE the begcubefaces result with 
    #  the 17 turn example from https://github.com/benbotto/rubiks-cube-cracker
    init_faceids = [9, 32, 29, 68, 13, 41, 24, 76, 25, 40, 12, 61, 21, 64, 1, 52, 2, 65, 20, 72, 6, 36, 18, 44, 16, 37, 5, 56, 30, 33, 8, 49, 26, 53, 0, 45, 17, 48, 10, 77, 28, 57, 4, 73, 22, 60, 14, 69]
    # Get the initial cube distance
    lehcode = lc.lehmer_code(8)
    statecode = bcube.getstate(np.array(init_faceids), lehcode)
    edge_lehcode = lc.lehmer_code(12)
    edge_statecode = bcube.getstate_edge(np.array(init_faceids), edge_lehcode)
    edge1_lehcode = lc.lehmer_code(12, 7)
    edge1_statecode = bcube.getstate_edgesplit(np.array(init_faceids), edge1_lehcode, 0)
    edge2_lehcode = lc.lehmer_code(12, 7)
    edge2_statecode = bcube.getstate_edgesplit(np.array(init_faceids), edge2_lehcode, 1)
    print(statecode, edge_statecode, edge1_statecode, edge2_statecode)
    print('Initial Scores')
    print(cornerDB[statecode], edgeDB[edge_statecode], edge1DB[edge1_statecode], edge2DB[edge2_statecode])
    print('Final corner cube faces used')
    print(np.array(init_faceids)[bcube.corner_faces])
    print('Final edge cube faces used')
    print(np.array(init_faceids)[bcube.edge_faces])

    retval = 0
#    retval = rcm.DFS_cython_solve(bytes(init_faceids), 0, cornerDB, edgeDB, edge1DB, edge2DB)
    for MAXDELDEP in np.arange(0,14):
        if not retval == 2:
            print('Trying MAXDELDEP {0:d}'.format(MAXDELDEP))
            retval = rcm.DFS_cython_solve(bytes(init_faceids), MAXDELDEP, cornerDB, edgeDB, edge1DB, edge2DB)
