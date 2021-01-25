#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:46:24 2020

@author: cjburke
"""

import numpy as np
import lehmer_code as lc
import rubik_cython_moves as rcm
from collections import deque as dq

class rubiks_cube():

    rotnames = ["DR", "DL", "DH",\
                "UR", "UL", "UH",\
                "RU", "RD", "RH",\
                "LU", "LD", "LH",\
                "FC", "FG", "FH",\
                "BC", "BG", "BH"]
    
    transfer_matrix = np.array([\
        [12,14,13, 8, 9, 2, 0, 1,15,16, 3, 4,18,17,19,10,11, 5, 6, 7,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],\
        [ 6, 7, 5,10,11,17,18,19, 3, 4,15,16, 0, 2, 1, 8, 9,13,12,14,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],\
        [18,19,17,15,16,13,12,14,10,11, 8, 9, 6, 5, 7, 3, 4, 2, 0, 1,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],\
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,41,40,42,36,37,29,28,30,43,44,31,32,46,45,47,38,39,34,33,35],\
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,34,33,35,38,39,46,45,47,31,32,43,44,29,28,30,36,37,41,40,42],\
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,45,46,47,43,44,40,41,42,38,39,36,37,33,34,35,31,32,28,29,30],\
        [ 0, 1, 2, 3, 4,19,17,18, 8, 9,26,27,12,13,14,15,16,45,47,46,20,21,10,11,24,25,38,39,28,29,30,31,32, 6, 7, 5,36,37,22,23,40,41,42,43,44,33,35,34],\
        [ 0, 1, 2, 3, 4,35,33,34, 8, 9,22,23,12,13,14,15,16, 6, 7, 5,20,21,38,39,24,25,10,11,28,29,30,31,32,45,47,46,36,37,26,27,40,41,42,43,44,17,19,18],\
        [ 0, 1, 2, 3, 4,46,45,47, 8, 9,38,39,12,13,14,15,16,33,34,35,20,21,26,27,24,25,22,23,28,29,30,31,32,17,18,19,36,37,10,11,40,41,42,43,44, 6, 5, 7],\
        [14,13,12, 3, 4, 5, 6, 7,24,25,10,11,40,42,41,15,16,17,18,19, 8, 9,22,23,36,37,26,27, 2, 1, 0,31,32,33,34,35,20,21,38,39,28,30,29,43,44,45,46,47],\
        [30,29,28, 3, 4, 5, 6, 7,20,21,10,11, 2, 1, 0,15,16,17,18,19,36,37,22,23, 8, 9,26,27,40,42,41,31,32,33,34,35,24,25,38,39,12,14,13,43,44,45,46,47],\
        [41,42,40, 3, 4, 5, 6, 7,36,37,10,11,28,29,30,15,16,17,18,19,24,25,22,23,20,21,26,27,12,13,14,31,32,33,34,35, 8, 9,38,39, 2, 0, 1,43,44,45,46,47],\
        [ 5, 6, 7,23,22,34,35,33, 8, 9,10,11,12,13,14,15,16,17,18,19, 4, 3,32,31,24,25,26,27, 1, 0, 2,21,20,30,29,28,36,37,38,39,40,41,42,43,44,45,46,47],\
        [29,28,30,21,20, 0, 1, 2, 8, 9,10,11,12,13,14,15,16,17,18,19,32,31, 4, 3,24,25,26,27,35,34,33,23,22, 7, 5, 6,36,37,38,39,40,41,42,43,44,45,46,47],\
        [34,35,33,31,32,29,28,30, 8, 9,10,11,12,13,14,15,16,17,18,19,22,23,20,21,24,25,26,27, 6, 5, 7, 3, 4, 2, 0, 1,36,37,38,39,40,41,42,43,44,45,46,47],\
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,19,18,17,27,26,47,46,45,20,21,22,23,16,15,44,43,28,29,30,31,32,33,34,35,36,37,38,39,14,13,12,25,24,42,41,40],\
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,42,41,40,25,24,14,13,12,20,21,22,23,44,43,16,15,28,29,30,31,32,33,34,35,36,37,38,39,47,46,45,27,26,19,18,17],\
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,45,46,47,43,44,40,41,42,20,21,22,23,26,27,24,25,28,29,30,31,32,33,34,35,36,37,38,39,17,18,19,15,16,12,13,14]
        ])

    facenames = np.array(["01my", "01mz", "01mx", "02my", "02mz","03my", "03px", "03mz",\
                 "04mx", "04mz","06px", "06mz",\
                 "07mx", "07py", "07mz","08py", "08mz","09px", "09py", "09mz",\
                 "10mx", "10my","12px", "12my",\
                 "16mx", "16py","18px", "18py",\
                 "19mx", "19my", "19pz","20my", "20pz","21px", "21my", "21pz",\
                 "22mx", "22pz","24px", "24pz",\
                 "25mx", "25py", "25pz","26py", "26pz","27px", "27py", "27pz"])
    #https://www.sfu.ca/~jtmulhol/math302/puzzles-rc-cubology.html
    facecodes = np.array(["c082","c080","c081","e111","e110","c071","c072","c070",\
                         "e121","e120","e101","e100",\
                         "c052","c051","c050","e091","e090","c061","c062","c060",\
                         "e080","e081","e070","e071",\
                         "e050","e051","e060","e061",\
                         "c042","c041","c040","e031","e030","c031","c032","c030",\
                         "e041","e040","e021","e020",\
                         "c011","c012","c010","e011","e010","c022","c021","c020"])

    facename_dict = {"01my":0, "01mz":1, "01mx":2, "02my":3, "02mz":4,"03my":5, "03px":6, "03mz":7,\
                 "04mx":8, "04mz":9,"06px":10, "06mz":11,\
                 "07mx":12, "07py":13, "07mz":14,"08py":15, "08mz":16,"09px":17, "09py":18, "09mz":19,\
                 "10mx":20, "10my":21,"12px":22, "12my":23,\
                 "16mx":24, "16py":25,"18px":26, "18py":27,\
                 "19mx":28, "19my":29, "19pz":30,"20my":31, "20pz":32,"21px":33, "21my":34, "21pz":35,\
                 "22mx":36, "22pz":37,"24px":38,"24pz":39,\
                 "25mx":40, "25py":41, "25pz":42,"26py":43, "26pz":44,"27px":45, "27py":46, "27pz":47}

    # 1 - orange, 2 - blue, 3 - yellow, 4 - green, 5 - white, 6 - red
    # pz - 1, px - 2, py - 3, mx - 4, my - 5, mz - 6
    facecolors = np.array([5,6,4,5,6,5,2,6,\
                           4,6,2,6,\
                           4,3,6,3,6,2,3,6,\
                           4,5,2,5,\
                           4,3,2,3,\
                           4,5,1,5,1,2,5,1,\
                           4,1,2,1,\
                           4,3,1,3,1,2,3,1])
    cubieids  = {"c01":0,"c02":1,"c03":2,"c04":3,"c05":4,"c06":5,"c07":6,"c08":7,\
                 "e01":8,"e02":9,"e03":10,"e04":11,"e05":12,"e06":13,\
                 "e07":14,"e08":15,"e09":16,"e10":17,"e11":18,"e12":19}
                    #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
    solved_faceids = [30,28,29,73,72,25,26,24,77,76,69,68,18,17,16,65,64,21,22,20,60,61,56,57,48,49,52,53,14,13,12,41,40, 9,10, 8,45,44,37,36, 1, 2, 0,33,32, 6, 5, 4]

    corner_faces = np.array([1,7,14,19,30,35,42,47], dtype=np.int)
    all_moves = np.zeros((18,48), dtype=np.int)

    
    def getstate(self, fc, lc):
        corner_faceids = fc[self.corner_faces]
        corner_cubieids = np.right_shift(corner_faceids, 2)
        corner_orientids = np.bitwise_and(corner_faceids, 3)[0:-1]
        corner_cubie_index, othval = lc.encode(corner_cubieids)
        corner_cubie_index = corner_cubie_index * 2187
        corner_orient_index = np.sum(corner_orientids * np.array([729,243,81,27,9,3,1], dtype=np.int))
        return corner_cubie_index + corner_orient_index, othval
        
    def make_pathlist(self, fc, curLevel, vst_arr, lehcode):
        tmp = dq([])
        all_moves, all_states = rcm.move_with_cython(fc, self.transfer_matrix)
        for curf in range(18):
            ist = 48*curf
            ied = ist + 48
            statecode = all_states[curf]
            if vst_arr[statecode] == 0:
                curfc = all_moves[ist:ied]
                tmp.append(curfc)
                vst_arr[statecode] = curLevel
                
        return tmp, vst_arr

    
    def BFS_CornerDB(self, bcube):
        # run the corner cube BFS to exhaustion to record
        #  min number of moves needed to complete
        # make the large db array storing moves
        mv_db = np.zeros((88179840,), dtype=np.int)

        
        begfaces = np.array(bcube.solved_faceids, np.int)
        lehcode = lc.lehmer_code(8)
        statecode, corp = bcube.getstate(begfaces, lehcode)
        mv_db[statecode] = -1
        
        # initialize neighbor list
        nlist = dq([])
        curLevel = 1
        newMoves, mv_db = bcube.make_pathlist(begfaces.tolist(), curLevel, mv_db, lehcode)
        for i, curfc in enumerate(newMoves):
            nlist.append([curfc, curLevel])

        foundValidEnd = False
        while not foundValidEnd:#  and curLevel<7:
            curData = nlist.popleft()
            curfc = curData[0]
            curLevel = curData[1] + 1
            newMoves, mv_db = bcube.make_pathlist(curfc, curLevel, mv_db, lehcode)
            for i, curf in enumerate(newMoves):
                nlist.append([curf, curLevel])
            if np.mod(len(nlist), 1000) == 0:
                print('Visit length: {0:d} Level: {1:d}'.format(len(nlist), curLevel))
                

            # Also check for empty nlist
            if len(nlist) == 0:
                foundValidEnd = True
                    
        return mv_db

    def save_db(self, arr, outfile):
        np.savez_compressed(outfile, db=arr)
            

if __name__ == '__main__':

    # Initialize the cube
    solvedCube = rubiks_cube()
        
    allEndStateArray = solvedCube.BFS_CornerDB(solvedCube)
    print('Done. Start saving')
    solvedCube.save_db(allEndStateArray, 'rubik_corner_db')
