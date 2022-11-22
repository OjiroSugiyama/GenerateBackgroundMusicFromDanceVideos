import csv
import glob
from tkinter.constants import E
import numpy as np
import os.path
import abc
from excitement import Excitement
from numpy.lib.function_base import average


class Csv_Reader(Excitement):

    def __init__(self, file):
        self.file = file
    
    def csv_read(self):
        print (self.file)
        with open(self.file, "r") as csv_file:
            # 初期化
            nodes, node = [], [[0]* 3 for i in range(14)]
            i= 0
            basename = os.path.basename(self.file)
            music_name = basename.split('_')
            # 分割するフレーム数の計算
            bpm = self.get_BPM(music_name[4])
            frame = self.calc_frame(bpm)
        #　csvから配列を取り出す
        for line in csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC):
            node[i] = line
            i+=1
            if i >= 14:
                i = 0
                # print(node)
                nodes.append(node)
                node = [[0]* 3 for i in range(14)]
        return nodes, frame