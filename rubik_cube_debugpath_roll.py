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
"""

import numpy as np
from collections import deque as dq
import rubik_cython_movesonly as rcm
import copy
import lehmer_code as lc

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
            edge_faceids = np.append(edge_faceids,fc[9])
        else:
            edge_faceids = np.append(edge_faceids,fc[4])
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

    # This is the move to roll the faceids and fill in the side faces
    def roll_move(self, fc, move_id):
        # make copy not reference 
        newfc = list(fc)
        # do the roll move for this face
        doface = self.move2face[move_id]        
        strt_idx = doface*8
        end_idx = (doface+1)*8
        newfc[strt_idx:end_idx] = np.roll(fc[strt_idx:end_idx], self.move2shifts[move_id]).tolist()
        newfc = self.side_move_switch(move_id, fc, newfc)
        return newfc

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


    def DFS_Turn(self, init_faceids):
        MAXLEVEL = 6
        # initialize neighbor list
        nlist = dq([])
        curLevel = 1
        newMoves = self.make_pathlist(init_faceids, 18)
        #fp = open('turnonly_idxcy.txt','w')
        for i, curMove in enumerate(newMoves):
            ## debug first moves
            #print('help')
            #ia = np.argsort(self.facecodes)
            #for j in ia:
            #    fp.write('{0} {1:d} {2:d}\n'.format(self.facecodes[j],curMove[0][j],i))
            cmv = curMove[1]
            tmp = [curMove[0], curLevel, "_{}".format(self.rotnames[cmv]), cmv] # This third entry is used for keeping track of move history
            nlist.appendleft(tmp)
        #fp.close()
        notSolved = True
        notEmpty = True
        totMoves = 0
        turn1Level = 0
        while notSolved and notEmpty:
            curData = nlist.popleft()
            curFID = curData[0]
            curLevel = curData[1]
            curLastMove = curData[3]
            curHist = curData[2]
            if curLevel == 1:
                turn1Level = turn1Level + 1
                print('Starting First Turn Level {0:d}'.format(turn1Level))
            if curLevel < MAXLEVEL:
                newMoves = self.make_pathlist(curFID, curLastMove)                
                for i, curMove in enumerate(newMoves):
                    cmv = curMove[1]
                    tmp = curHist + '_' + self.rotnames[cmv]
                    nlist.appendleft([curMove[0], curLevel+1, tmp, cmv])
                    totMoves = totMoves + 1
            # Also check for empty nlist
            if len(nlist) == 0:
                notEmpty = False
                    
        return totMoves


if __name__ == '__main__':
    
    char_move = "L' U B D' L' F' D2 F2 R2 U' L' F' D R' L F2 D R2 D B U' B2 D2 R D2 F U2 L' B2 U2 R' L2 U2 B2 L' B2 D' U' F2 U' F' D R B' L' B2 L2 D' U L B' D U2 F2 L2 B D2 L' D2 U' L' F' U' F2 D L2 B2 R' F' U R D B' F' U' L' F2 U F2 D' F2 L2 B' U' L U L' B D2 L' D R' U2 F2 D' U' L U R2 L'"
    char_move = char_move.split()
    char_move_dict = {"D":0,"D'":1,"D2":2,\
                      "U'":3,"U":4,"U2":5,\
                      "R":6,"R'":7,"R2":8,\
                      "L'":9,"L":10,"L2":11,\
                      "F":12,"F'":13,"F2":14,\
                      "B'":15,"B":16,"B2":17}
    moves = []
    for curchar in char_move:
        moves.append(char_move_dict[curchar])
#    rotnames = {0:"DR", 1:"DL", 2:"DH",\
#                3:"UR", 4:"UL", 5:"UH",\
#                6:"RU", 7:"RD", 8:"RH",\
#                9:"LU", 10:"LD", 11:"LH",\
#                12:"FC", 13:"FG", 14:"FH",\
#                15:"BC", 16:"BG", 17:"BH"}
    print(moves)
    #moves = [0,3,6,9,12,15,0,3,6,9,12,15,0,3]
    # initialize cube
    bcube = rubiks_cube()
    
    # solved cube in roll 
    init_faceids = [2, 49, 17, 65, 22, 53, 5, 33, \
                    6, 52, 21, 69, 26, 56, 9, 37, \
                    10, 57, 25, 73, 30, 61, 13, 41,\
                    14, 60, 29, 77, 18, 48, 1, 45,\
                    4, 36, 8, 40, 12, 44, 0, 32,\
                    16, 76, 28, 72, 24, 68, 20, 64]

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
    
    print('Initial faceids')
    print(init_faceids)
    print('Initial lehmer codes')
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
    print('initial corner cube faces used')
    print(np.array(init_faceids)[bcube.corner_faces])
    print('initial edge cube faces used')
    print(np.array(init_faceids)[bcube.edge_faces])
    
    curLevel = 0
    new_faceid = init_faceids
    for curMove in moves:
        curLevel = curLevel +1
        newMoves = bcube.make_pathlist(new_faceid, 18)
        new_faceid = newMoves[curMove][0]
        print('Doing move {0:d} {1} at level: {2:d}'.format(curMove, bcube.rotnames[curMove], curLevel))
        print('New Faces')
        print(new_faceid)
        print('New Lehmer codes')
        statecode = bcube.getstate(np.array(new_faceid), lehcode)
        edge_statecode = bcube.getstate_edge(np.array(new_faceid), edge_lehcode)
        edge1_statecode = bcube.getstate_edgesplit(np.array(new_faceid), edge1_lehcode, 0)
        edge2_statecode = bcube.getstate_edgesplit(np.array(new_faceid), edge2_lehcode, 1)
        print(statecode, edge_statecode, edge1_statecode, edge2_statecode)
        print('Corner cube faces used')
        print(np.array(new_faceid)[bcube.corner_faces])
        print('Edge cube faces used')
        print(np.array(new_faceid)[bcube.edge_faces])

        print('New Scores')
        cs = cornerDB[statecode]
        ce = edgeDB[edge_statecode]
        ce1 = edge1DB[edge1_statecode]
        ce2 = edge2DB[edge2_statecode]
        score = np.max([cs, ce, ce1, ce2])
        print(cs,ce,ce1 ,ce2 )
        print(score, curLevel)
        
    print('Final corner cube faces used')
    print(np.array(new_faceid)[bcube.corner_faces])
    print('Final edge cube faces used')
    print(np.array(new_faceid)[bcube.edge_faces])

    
    #totMoves = bcube.DFS_Turn(init_faceids)
    #print('Done {0:d} moves'.format(totMoves))

