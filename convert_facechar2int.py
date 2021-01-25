#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:29:53 2021

@author: cjburke
"""
import numpy as np
if __name__ == '__main__':
    #https://www.sfu.ca/~jtmulhol/math302/puzzles-rc-cubology.html
    facecodechars = ["c012","e051","c051","e091","c062","e061","c021","e011",\
                 "c022","e060","c061","e101","c072","e070","c031","e021",\
                 "c032","e071","c071","e111","c082","e081","c041","e031",\
                 "c042","e080","c081","e121","c052","e050","c011","e041",\
                 "c020","e020","c030","e030","c040","e040","c010","e010",\
                 "c050","e120","c080","e110","c070","e100","c060","e090"]
    cubieids  = {"c01":0,"c02":1,"c03":2,"c04":3,"c05":4,"c06":5,"c07":6,"c08":7,\
                 "e01":8,"e02":9,"e03":10,"e04":11,"e05":12,"e06":13,\
                 "e07":14,"e08":15,"e09":16,"e10":17,"e11":18,"e12":19}
    
    # Convert the cubie identifier strings into their integer form
    finints = []
    for curcode in facecodechars:
        cubeint = cubieids[curcode[0:3]]
        cubeintbits = bin(cubeint)
        cubeintshift = cubeint << 2 
        cubeintshiftbits = bin(cubeintshift)
        orientint= int(curcode[3])
        orientintbits = bin(orientint)
        finint = cubeintshift | orientint
        finintbit = bin(finint)
        finints.append(finint)
    print(finints)
    
    
    #  Old system to new system ordering of facenames t new order
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
    
    corner_order_list = []
    for cln in corner_list_names:
        tmp = []
        for i in range(3):
            curcn = cln[i]
            faceid = facename_faceid_dict[curcn]
            idx = np.where(facecodeints == faceid)[0]
            tmp.append(idx[0])
        corner_order_list.append(tmp)
        
    print(corner_order_list)
    
    edge_order_list = []
    for cln in edge_list_names:
        tmp = []
        for i in range(2):
            curcn = cln[i]
            faceid = facename_faceid_dict[curcn]
            idx = np.where(facecodeints == faceid)[0]
            tmp.append(idx[0])
        edge_order_list.append(tmp)
        
    print(edge_order_list)

