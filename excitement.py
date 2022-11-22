import csv
import glob
from tkinter.constants import E
import numpy as np
import random
import os.path
import abc
from numpy.lib.function_base import average
import pickle




class Excitement(abc.ABC):
    """
    盛り上がり度の計算
    関節：
    [0:左腰, 1:左膝, 2:左足, 3:右腰, 4:右膝, 5:右足, 6:首, 7:頭, 8:左肩, 9:左肘, 10:左手首, 11:右肩, 12:右肘, 13:右手首]
    面積
    計算する骨
    左太もも:(0,1), 左足:(1,2), 右太もも:(3,4), 右足:(4,5), 胴体:(6,14), 首:(6,7), 左肩:(6,8), 左上腕:(8,9), 左前腕:(9,10), 右肩:(6,11), 右上腕:(11,12), 右前腕:(12,13)
    """

    # def __init__(self):
    #     # self.file = file
    #     self.motion_excitement_array = list()
    #     # print (self.file)
    #     # with open(self.file, "r") as csv_file:
    #     #     # 初期化
    #     #     nodes, node = [], [[0]* 3 for i in range(14)]
    #     #     i, j = 0, 0
    #     #     basename = os.path.basename(self.file)
    #     #     music_name = basename.split('_')
    #     #     # 分割するフレーム数の計算
    #     #     bpm = self.get_BPM(music_name[4])
    #     #     frame = self.calc_frame(bpm)
    #     #     #　csvから配列を取り出す
    #     #     for line in csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC):
    #     #         node[i] = line
    #     #         i+=1
    #     #         if i >= 14:
    #     #             i = 0
    #     #             # print(node)
    #     #             nodes.append(node)
    #     #             node = [[0]* 3 for i in range(14)]
    #     # self.calcMotionExcitement(nodes, frame)


    @abc.abstractmethod
    def pkl_read(self, file):
        pass

    @abc.abstractclassmethod
    def makeMotionNodes(self, poses):
        pass

    @abc.abstractmethod
    def Openpose_read(self, file):
        pass


    def excitement(self, ns): 
        for t in range(len(ns)-1):
            excitementS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # print(len(ns[t][0]))
            excitementS = self.calc_S(t, ns)

            for i in range(14):
                v[i] = self.make_vector(ns[t][i], ns[t+1][i])
            v[14] = self.make_vector(self.make_midpoint(ns[t][0], ns[t][3]), self.make_midpoint(ns[t+1][0], ns[t+1][3]))
            # 案1：平均値
            excitementV = self.average_V(v)

            # 案2：最大値
            # excitementV = self.max_V(v)

        return excitementS, excitementV
    
    def calc_S(self, t, ns):
        excitementS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        excitementS[0] = self.triangular_Pyramid_Hulf(ns[t][0], ns[t][1], ns[t+1][0], ns[t+1][1])      # 左太もも
        excitementS[1] = self.triangular_Pyramid_Hulf(ns[t][1], ns[t][2], ns[t+1][1], ns[t+1][2])      # 左足
        excitementS[2] = self.triangular_Pyramid_Hulf(ns[t][3], ns[t][4], ns[t+1][3], ns[t+1][4])      # 右太もも
        excitementS[3] = self.triangular_Pyramid_Hulf(ns[t][4], ns[t][5], ns[t+1][4], ns[t+1][5])      # 右足
        excitementS[4] = self.triangular_Pyramid_Hulf(ns[t][6], self.make_midpoint(ns[t][0], ns[t][3]), ns[t+1][6], self.make_midpoint(ns[t+1][0], ns[t+1][3])) # 胴体
        excitementS[5] = self.triangular_Pyramid_Hulf(ns[t][6], ns[t][7], ns[t+1][6], ns[t+1][7])      # 首
        excitementS[6] = self.triangular_Pyramid_Hulf(ns[t][6], ns[t][8], ns[t+1][6], ns[t+1][8])      # 左肩
        excitementS[7] = self.triangular_Pyramid_Hulf(ns[t][8], ns[t][9], ns[t+1][8], ns[t+1][9])      # 左上腕
        excitementS[8] = self.triangular_Pyramid_Hulf(ns[t][9], ns[t][10], ns[t+1][9], ns[t+1][10])    # 左前腕
        excitementS[9] = self.triangular_Pyramid_Hulf(ns[t][6], ns[t][11], ns[t+1][6], ns[t+1][11])    # 右肩
        excitementS[10] = self.triangular_Pyramid_Hulf(ns[t][11], ns[t][12], ns[t+1][11], ns[t+1][12]) # 右上腕
        excitementS[11] = self.triangular_Pyramid_Hulf(ns[t][12], ns[t][13], ns[t+1][12], ns[t+1][13]) # 右前腕

        return excitementS

    def average_V(self, v):
        excitementV = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        excitementV[0] = (np.linalg.norm(v[0]) + np.linalg.norm(v[1]))/2
        excitementV[1] = (np.linalg.norm(v[1]) + np.linalg.norm(v[2]))/2
        excitementV[2] = (np.linalg.norm(v[3]) + np.linalg.norm(v[4]))/2
        excitementV[3] = (np.linalg.norm(v[4]) + np.linalg.norm(v[5]))/2
        excitementV[4] = (np.linalg.norm(v[6]) + np.linalg.norm(v[14]))/2
        excitementV[5] = (np.linalg.norm(v[6]) + np.linalg.norm(v[7]))/2
        excitementV[6] = (np.linalg.norm(v[6]) + np.linalg.norm(v[8]))/2
        excitementV[7] = (np.linalg.norm(v[8]) + np.linalg.norm(v[9]))/2
        excitementV[8] = (np.linalg.norm(v[9]) + np.linalg.norm(v[10]))/2
        excitementV[9] = (np.linalg.norm(v[6]) + np.linalg.norm(v[11]))/2
        excitementV[10] = (np.linalg.norm(v[11]) + np.linalg.norm(v[12]))/2
        excitementV[11] = (np.linalg.norm(v[12]) + np.linalg.norm(v[13]))/2
        return excitementV

    def max_V(self, v):
        excitementV = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        excitementV[0] = self.v_max(np.linalg.norm(v[0]), np.linalg.norm(v[1]))
        excitementV[1] = self.v_max(np.linalg.norm(v[1]), np.linalg.norm(v[2]))
        excitementV[2] = self.v_max(np.linalg.norm(v[3]), np.linalg.norm(v[4]))
        excitementV[3] = self.v_max(np.linalg.norm(v[4]), np.linalg.norm(v[5]))
        excitementV[4] = self.v_max(np.linalg.norm(v[6]), np.linalg.norm(v[14]))
        excitementV[5] = self.v_max(np.linalg.norm(v[6]), np.linalg.norm(v[7]))
        excitementV[6] = self.v_max(np.linalg.norm(v[6]), np.linalg.norm(v[8]))
        excitementV[7] = self.v_max(np.linalg.norm(v[8]), np.linalg.norm(v[9]))
        excitementV[8] = self.v_max(np.linalg.norm(v[9]), np.linalg.norm(v[10]))
        excitementV[9] = self.v_max(np.linalg.norm(v[6]), np.linalg.norm(v[11]))
        excitementV[10] = self.v_max(np.linalg.norm(v[11]), np.linalg.norm(v[12]))
        excitementV[11] = self.v_max(np.linalg.norm(v[12]), np.linalg.norm(v[13]))
        return excitementV



    def triangular_Pyramid_Hulf(self, nodeA, nodeB, nodeC, nodeD):# 三角錐の面積の半分を計算
        vectors = [[0] * 3 for i in range(5)]
        # ベクトルAB(0), AC(1), AD(2), BC(3), BD(4)を計算
        vectors[0] = self.make_vector(nodeA, nodeB)
        vectors[1] = self.make_vector(nodeA, nodeC)
        vectors[2] = self.make_vector(nodeA, nodeD)
        vectors[3] = self.make_vector(nodeB, nodeC)
        vectors[4] = self.make_vector(nodeB, nodeD)


        s = (np.linalg.norm(np.cross(vectors[0],vectors[1])) + np.linalg.norm(np.cross(vectors[0],vectors[2])) + np.linalg.norm(np.cross(vectors[1],vectors[2])) + np.linalg.norm(np.cross(vectors[3],vectors[4])))*0.25
        
        return s


    def make_vector(self, nodeA, nodeB):# 二つの三次元座標からベクトルを作成
        vector = [0, 0, 0]
        
        vector[0] = nodeB[0] - nodeA[0]
        vector[1] = nodeB[1] - nodeA[1]
        vector[2] = nodeB[2] - nodeA[2]
  

        return vector

    def make_midpoint(self, nodeA, nodeB):
        node = [0, 0, 0]
        node[0] = (nodeA[0] + nodeB[0])/2
        node[1] = (nodeA[1] + nodeB[1])/2
        node[2] = (nodeA[2] + nodeB[2])/2

        return node

    def v_max(self, a, b):
        if a >= b:
            return a
        else:
            return b

    def split_list(self, l, n):
        for idx in range(0, len(l), n):
            yield l[idx:idx + n]

    def get_BPM(self, name):
        """AISTDance Databaseにおける楽曲のBPMのList"""
        music_list = {
            "mBR0" : 80, "mBR1" : 90, "mBR2" : 100, "mBR3" : 110, "mBR4" : 120, "mBR5" : 130,
            "mPO0" : 80, "mPO1" : 90, "mPO2" : 100, "mPO3" : 110, "mPO4" : 120, "mPO5" : 130, 
            "mLO0" : 80, "mLO1" : 90, "mLO2" : 100, "mLO3" : 110, "mLO4" : 120, "mLO5" : 130, 
            "mMH0" : 80, "mMH1" : 90, "mMH2" : 100, "mMH3" : 110, "mMH4" : 120, "mMH5" : 130, 
            "mLH0" : 80, "mLH1" : 90, "mLH2" : 100, "mLH3" : 110, "mLH4" : 120, "mLH5" : 130,
            "mHO0" : 110, "mHO1" : 115, "mHO2" : 120, "mHO3" : 125, "mHO4" : 130, "mHO5" : 135,  
            "mWA0" : 80, "mWA1" : 90, "mWA2" : 100, "mWA3" : 110, "mWA4" : 120, "mWA5" : 130, 
            "mKR0" : 80, "mKR1" : 90, "mKR2" : 100, "mKR3" : 110, "mKR4" : 120, "mKR5" : 130, 
            "mJS0" : 80, "mJS1" : 90, "mJS2" : 100, "mJS3" : 110, "mJS4" : 120, "mJS5" : 130, 
            "mJB0" : 80, "mJB1" : 90, "mJb2" : 100, "mJB3" : 110, "mJB4" : 120, "mJB5" : 130, 
            }
        return music_list[name]

    def calc_frame(self, bpm):
        """BPMから1小節の間隔を計算"""
        frame = 30 * 60 * 8 / bpm
        print("frame:"+str(frame))
        return int(frame)

    def min_max(self, x, axis = None):
        """min-max で正規化"""
        min = np.min(x, axis=axis, keepdims=True)
        max = np.max(x, axis=axis, keepdims=True)

        result = (x-min)/(max-min)
        return result

    def calcMotionExcitement(self, nodes, frame):
        """動画から抽出した骨格情報から盛り上がり度を計算する"""
        print("calcMotionExcitement:start")
        s, v = [], []
        frame_S, frame_V = [], []
        split_nodes = list(self.split_list(nodes, frame))

        # print("split_nodes:")
        # print(np.shape(split_nodes))
        # print("nodes:")
        # print(np.shape(nodes))


        for node in split_nodes:
            s, v = self.excitement(node)
            frame_S.append(s)
            frame_V.append(v)
        
        min_max_S = self.min_max(frame_S, axis = 0)
        min_max_V = self.min_max(frame_V, axis = 0)

        e =  (min_max_S + min_max_V)*0.5  #ここは一旦
        print("e:"+str(len(e)))
        self.motion_excitement_array = np.sum(e, axis = 1)/len(e[0]) # ここは仮
        # print(np.sum(e, axis = 1))
        # print("motion_excitement_array:"+str(self.motion_excitement_array))
        print("calcMotionExcitement:finish")

    def randomMotionExcitement(self, nodes, frame):
        """ランダムに盛り上がり度を決定する"""
        split_nodes = list(self.split_list(nodes, frame))
        random_excitement_array = []
        for node in split_nodes:
          random_excitement_array.append(random.uniform(0,1))
        self.motion_excitement_array = random_excitement_array
        print("randomMotionExcitement:finish")

    # files = glob.glob("C:\\Users\gorim\Desktop\Motion-Aware-Sequencer\output-20211016T031203Z-001\output\g*.csv")
    # 一動画ごとに正規化をしているので高いところはその動画の盛り上がっている部分を指している？
    

class Pkl_Reader(Excitement):

    def __init__(self, file):
        self.file = file
    
    def pkl_read(self):
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

    def makeMotionNodes(self, poses):
        l = [[[0] * 3 for i in range(24)] for j in range(len(poses))]

        n = 3

        for i in range(len(poses)):
            l[i] = [poses[i][idx:idx + n] for idx in range(0,len(poses[i]), n)] 

        # OpenPoseとAIST++の共通している関節を抽出
        # AIST++:
        # [ "root(根本)", "left_hip(左腰〇)", "left_knee(左膝〇)", "left_foot(左足〇)", "left_toe(左足先)", 
        #  "right_hip(右腰〇)", "right_knee(右膝〇)", "right_foot(右足〇)", "right_toe(右足先)", 
        #  "waist(腰)", "spine(背骨)", "chest(胸)", "neck(首〇)", "head(頭〇)", 
        #  "left_in_shoulder(左内肩)", "left_shoulder(左肩〇)", "left_elbow(左肘〇)", "left_wrist(左手首〇)", "left_hand(左手)"
        #  "right_in_shoulder(右内肩)", "right_shoulder(右肩〇)", "right_elbow(右肘〇)", "right_wrist(右手首〇)", "right_hand(右手)"]
        # 1,2,3,5,6,7,12,13,15,16,17,20,21,22を使用

        nodes = [[[0] * 3 for i in range(14)] for j in range(int(len(poses)/2))]
        useNode = [0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,1,1,0,0,1,1,1,0]


        for i in range(len(poses)):
            n = 0
            t = int(i/2)

            if (i+1)%2 == 0:
                for j in range(24):
                    if useNode[j] == 1:
                        nodes[t][n] = l[i][j]

                        n+=1
        return nodes

    
    def Openpose_read(self, file):
        return 



