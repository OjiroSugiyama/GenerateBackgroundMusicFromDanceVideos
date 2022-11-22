import csv
import glob
from tkinter.constants import E
import numpy as np
import os.path
import abc
from excitement import Excitement
from numpy.lib.function_base import average

import json

f = open('C:\\Users\gorim\Desktop\Motion-Aware-Sequencer\motions\json\Produce_000000000000_keypoints.json')
j = json.load(f)

print(str(len(j['people'][1]['pose_keypoints_2d'])))

# [0:鼻, 1:首, 2:右肩, 3:右肘, 4:右手首, 5:左肩, 6:左肘, 7:左手首, 8:腰, 9:右腰, 
# 10:右膝, 11:右足首, 12:左腰, 13:左膝, 14:左足首, 15:右目, 16:左目, 17:右耳, 18:左耳, 19:左足親指, 
# 20:左足小指, 21:左足かかと, 22:右足親指, 23:右足小指, 24:右足かかと]


"""
class Openpose_Reader(Excitement):

    def __init__(self, file):
        self.file = file
    
    def Openpose_read(self):
         print (self.file)
        with open(self.file, 'rb')as web:
            pre_nodes = pickle.load(web)
        # 初期化
        nodes= self.makeMotionNodes(pre_nodes['smpl_poses'])
        print(len(nodes))
        print(len(nodes[0]))
        print(len(nodes[0][0]))
        basename = os.path.basename(self.file)
        music_name = basename.split('_')
        # 分割するフレーム数の計算
        bpm = self.get_BPM(music_name[4])
        frame = self.calc_frame(bpm)
        # # csvから配列を取り出す
        # node = [[0]* 3 for i in range(14)]
        # for line in csv.reader(web, quoting=csv.QUOTE_NONNUMERIC):
        #     node[i] = line
        #     i+=1
        #     if i >= 14:
        #         i = 0
        #         # print(node)
        #         nodes.append(node)
        #         node = [[0]* 3 for i in range(14)]
        # print("nodes:"+str(len(nodes)))
        return nodes, frame
"""