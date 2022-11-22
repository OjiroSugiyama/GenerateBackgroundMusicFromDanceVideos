import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import math 
from scipy.stats import stats
from scipy.stats import norm
import random
import os.path
import abc
import pickle
# import plotly.express as px
import csv
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class LabanFutureValue():

    def __init__(self, filename):
        with open(filename, 'rb')as web:
            pre_nodes = pickle.load(web)

        self.interval = 1/30
        self.nodes= self.makeMotionNodes(pre_nodes['keypoints3d'])
        basename = os.path.basename(filename)
        music_name = basename.split('_')
        bpm = self.get_BPM(music_name[4])
        self.subsection, self.beat= self.calc_subsection(bpm)
        self.AccelNodes, self.JerkNodes = [],[]

        self.velocityVector, self.accelerationVector, self.JerkVecor , self.spaceVector= [],[],[],[]
        self.motion_excitement_array = list()
        self.t = 5
        self.delta = 1
        self.T = self.delta/self.interval
        self.timeAlpha, self.weightAlpha, self.spaceAlpha, self.flowAlpha = [],[],[],[]
        self.weightBeta, self.timeBeta, self.spaceBeta, self.flowBeta = 0.4,0.1,0.4,0.1
        self.excitement = []
        self.limits = []
        self.sc_v = pickle.load(open("C:\\Users\gorim\Desktop\Motion-Aware-Sequencer\Scaler\scaler_velocity.pickle", "rb"))
        self.sc_s = pickle.load(open("C:\\Users\gorim\Desktop\Motion-Aware-Sequencer\Scaler\scaler_space.pickle", "rb"))
        self.sc_a = pickle.load(open("C:\\Users\gorim\Desktop\Motion-Aware-Sequencer\Scaler\scaler_accelerate.pickle", "rb"))
        self.sc_j = pickle.load(open("C:\\Users\gorim\Desktop\Motion-Aware-Sequencer\Scaler\scaler_jerk.pickle", "rb"))
        self.limits = []
        self.prefer()

    def prefer(self):
        l = []
        self.timeAlpha = [0.2, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1]
        self.weightAlpha = [0.2, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1]
        self.spaceAlpha = [0.2, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1]
        self.flowAlpha = [0.2, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1]

        with open('C:\\Users\gorim\Desktop\Motion-Aware-Sequencer\limits.csv')as f:
            for line in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                l.append(line)
        self.limits = list(l)

    def OnInterval(self):
        split_nodes = list(self.split_list(self.nodes, self.subsection))
        e = []
        # print("T"+str(self.delta/self.interval))
        # print(len(split_nodes))
        for node in split_nodes:
            # 初期化
            self.velocityVector, self.accelerationVector, self.JerkVecor , self.spaceVector= [],[],[],[]
            if len(node) <= 1:
                print("break")
                break

            # calculating weight effort
            self.velocityVector= self.calc_Velocity(node)
            # print("min_max_velocityVector:"+str(self.velocityVector))
            weight = 0.0
            # print(node)
            # print("self.vector:",format(self.velocityVector))
            for i in range(len(self.velocityVector)):
                weight += (self.weightAlpha[i] * self.velocityVector[i])
            print("weight:"+str(weight))
            
            # calculating time effort
            self.accelerationVector = self.calc_Acceleration(node)
            timeVector = self.calc_Time(self.accelerationVector)
            # print("self.vector:",format(timeVector))
            time = 0.0
            for i in range(len(timeVector)):
                time += (self.timeAlpha[i] * timeVector[i])
            print("time:"+str(time))
            
            # calculating space effort
            self.spaceVector = self.calc_Space(node)
            # print("self.vector:",format(self.spaceVector))
            space = 0.0
            # print(self.spaceVector)

            for i in range(len(self.spaceVector)):
                space += (self.spaceAlpha[i] * self.spaceVector[i])
            print("space:"+str(space))
            
            # calculating flow effort
            self.JerkVector = self.calc_Jerk(node)
            flowVector = self.calc_Flow(self.JerkVector)
            # print("self.vector:",format(flowVector))
            flow = 0.0
            for i in range(len(flowVector)):
                flow += (self.flowAlpha[i] * flowVector[i])
            print("flow:"+str(flow))


            self.excitement.append(self.weightBeta*weight + self.timeBeta*time + self.spaceBeta*space + self.flowBeta*flow)
        # ロジスティック関数で調整
        self.excitement = self.logistic(np.array(self.excitement), 10, 1, 0.5)

        print("excitement:"+str(self.excitement))

        # self.excitement = self.excitement/10

    def makeMotionNodes(self, poses):
            l = [[[0] * 3 for i in range(17)] for j in range(len(poses))]
            n = 3
            # for i in range(len(poses)):
            #     l[i] = [poses[i][idx:idx + n] for idx in range(0,len(poses[i]), n)] 
            # OpenPoseとAIST++の共通している関節を抽出
            # AIST++:
            # [ "root(根本)", "left_hip(左腰〇:0)", "left_knee(左膝〇:1)", "left_foot(左足〇:2)", "left_toe(左足先)", 
            #  "right_hip(右腰〇:3)", "right_knee(右膝〇:4)", "right_foot(右足〇:5)", "right_toe(右足先)", 
            #  "waist(腰)", "spine(背骨)", "chest(胸)", "neck(首〇:6)", "head(頭〇:7)", 
            #  "left_in_shoulder(左内肩)", "left_shoulder(左肩〇:8)", "left_elbow(左肘〇:9)", "left_wrist(左手首〇:10)", "left_hand(左手)"
            #  "right_in_shoulder(右内肩)", "right_shoulder(右肩〇:11)", "right_elbow(右肘〇:12)", "right_wrist(右手首〇:13)", "right_hand(右手)"]
            # 1,2,3,5,6,7,12,13,15,16,17,20,21,22を使用
            # ：['keypoints3d']
            # [鼻:0, 右耳:1, 右目:2, 左耳:3, 左目:4, 左肩:5, 右肩:6, 左肘:7, 右肘:8, 左手:9, 右手:10, 左腰:11, 右腰:12, 左膝:13, 右膝:14, 左足:15, 右足:16]


            nodes = [[[0] * 3 for i in range(13)] for j in range(int(len(poses)/2))]
            # useNode = [0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,1,　1,0,0,1,1,1,0]
            useNode = [1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
            for i in range(len(poses)):
                n = 0
                t = int(i/2)
                if (i+1)%2 == 0:
                    for j in range(17):
                        if useNode[j] == 1:
                            nodes[t][n] = poses[i][j]
                            n+=1
            return nodes

    def make_midpoint(self, nodeA, nodeB):
            node = [0, 0, 0]
            node[0] = (nodeA[0] + nodeB[0])/2
            node[1] = (nodeA[1] + nodeB[1])/2
            node[2] = (nodeA[2] + nodeB[2])/2
            return node

    def make_InnerPart(self, NodeA, NodeB, m, n):
        """線分ABをm:nに内分する点"""
        r=(-n*NodeA+m*NodeB)/(-n+m)
        return r


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
    
    def split_list(self, l, n):
        for idx in range(0, len(l), n):
            yield l[idx:idx + n]
    
    def min_max(self, x, p, axis = None):
        """min-max で正規化"""
        max = self.limits[p*2]
        min = self.limits[p*2+1]
        result = x
        # print("max:",format(max))
        if p == 4:
            for i in range(len(x)):
                result[i] = (x[i]-max[1])/(max[0]-max[1])
        else:
            for i in range(len(x)):
                for j in range(len(x[0])):
                    result[i][j] = (x[i][j]-min[j])/(max[j]-min[j])
        
        return result
    
    def logistic(self, x, a, k, x0):
      y = k / (1 + np.exp(-a * k * (x - x0)))
      return y

    def calc_Velocity(self, node):
        """関節ごとの速さの計算"""
        velocityVector_ave = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        velocityVector = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(int(len(node)-1))]
        v =[]
        for i in range(len(node)-1):
            for j in range(len(node[0])):
                velocityVector[i][j] =(node[i+1][j]-node[i][j])/self.interval
                
        for i in range(len(velocityVector)):
            for j in range(len(velocityVector[0])):
                v = np.linalg.norm(velocityVector[i][j])
                if v > self.limits[0][j]:
                    velocityVector[i][j] = self.limits[0][j]
                # elif v < self.limits[1][j]:
                #   velocityVector[i][j] = self.limits[1][j]
                elif np.isnan(v):
                    velocityVector[i][j] = 1
                else:
                  velocityVector[i][j] = v
        # print("velocityVector:"+str(velocityVector))
        velocityVector_log = np.log(velocityVector)
        # print("velocityVector_log:"+str(velocityVector_log))
        # 標準化
        velocityVector_log_std = self.sc_v.transform(velocityVector_log)
        print("velocityVector_log_std:"+str(velocityVector_log_std))

        # 累積分布関数のやつ：
        velocityVector_norm = norm.cdf(velocityVector_log_std, loc=0, scale=1)
        print("velocityVector_norm:"+str(velocityVector_norm))
        # ロジスティック関数のやつ：
        velocityVector_logistic = self.logistic(velocityVector_log_std, 1, 1, 0)
        # print("velocityVector_y_logistic:"+str(velocityVector_y))

        velocityVector_ave = np.sum(velocityVector_norm,axis=0)/len(velocityVector_norm)

        # print(velocityVector_ave)

        return velocityVector_ave

    def calc_Acceleration(self, node):
        """関節ごとの加速度の計算"""
        accelerationVector = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(int(len(node)-2))]
        for i in range(len(node)-2):
            for j in range(len(node[0])):
                a = np.linalg.norm((node[i+2][j]-2*node[i+1][j]+node[i][j])/pow(self.interval,2))
                if a > self.limits[4][j]:
                  accelerationVector[i][j] = self.limits[4][j]
                # elif a < self.limits[5][j]:
                #   accelerationVector[i][j] = self.limits[5][j]
                elif np.isnan(a):
                  accelerationVector[i][j] = 1
                else:
                  accelerationVector[i][j] = a
        # print("accelerationVector:"+str(accelerationVector))
        accelerationVector_log = np.log(accelerationVector)
        # print("accelerationVector_log:"+str(accelerationVector_log))
        # 標準化
        accelerationVector_log_std = self.sc_a.transform(accelerationVector_log)
        # accelerationVector = self.min_max(accelerationVector, 0)
        # print("accelerationVector_log_std:"+str(accelerationVector_log_std))

        # 累積分布関数のやつ：
        accelerationVector_norm = norm.cdf(accelerationVector_log_std, loc=0, scale=1)
        # print("accelerationVector_y_norm:"+str(accelerationVector_y))
        # ロジスティック関数のやつ：
        accelerationVector_logistic = self.logistic(accelerationVector_log_std, 1, 1, 0)

        return accelerationVector_norm

    def calc_Jerk(self, node):
        """関節ごとの力の変化の計算"""
        JerkVector = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(int(len(node)-4))]
        for i in range(len(node)-4):
            for j in range(len(node[0])):
                je = np.linalg.norm(((node[i+4][j]-2*node[i+3][j]+2*node[i+1][j]-node[i][j])/2*pow(self.interval,3)))
                if je > self.limits[6][j]:
                    JerkVector[i][j] = self.limits[6][j]
                # elif je < self.limits[7][j]:
                #     JerkVector[i][j] = self.limits[7][j]
                elif np.isnan(je):
                    JerkVector[i][j] = 1
                else:
                    JerkVector[i][j] = je
        # print("JerkVector:"+str(JerkVector))
        JerkVector_log = np.log(JerkVector)
        # print("JerkVector_log:"+str(JerkVector_log))
        # 標準化
        JerkVector_log_std = self.sc_j.transform(JerkVector_log)
        # JerkVector = self.min_max(JerkVector, 0)
        # print("JerkVector_log_std:"+str(JerkVector_log_std))

        # 累積分布関数のやつ：
        JerkVector_norm = norm.cdf(JerkVector_log_std, loc=0, scale=1)
        # print("JerkVector_y_norm:"+str(JerkVector_y))
        # ロジスティック関数のやつ：
        JerkVector_logistic = self.logistic(JerkVector_log_std, 1, 1, 0)

        return JerkVector_norm

    def calc_subsection(self, bpm):
            """BPMから1小節の間隔を計算"""
            subsection = 60 * 8/ (bpm * self.interval)
            beat = 60/ (bpm * self.interval)
            # print("subsection:"+str(subsection))
            return int(subsection), int(beat)

    def calc_Space(self, nodes):
        """単位方向ベクトルのばらつき"""
        # print("calc_Space:start")
        space_nodes = list(self.split_list(nodes, self.beat))
        space = []
        spacevector = []
        i = 0
        for node in space_nodes:
          i+=1
          if i > len(space_nodes)-1:
            break
          spaceNumeratorX = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(int(len(node)-1))]
          spaceNumeratorY = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(int(len(node)-1))]
          spaceNumeratorZ = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(int(len(node)-1))]

          spaceNumeratorX_sum =[]
          spaceNumeratorY_sum =[]
          spaceNumeratorZ_sum =[]

          spaceDenominatorX = []
          spaceDenominatorY = []
          spaceDenominatorZ = []

          s_vector = []

          for t in range(len(node)-1):
            for j in range(len(node[0])):
                spaceNumeratorX[t][j] = abs(node[t+1][j][0]-node[t][j][0])
                spaceNumeratorY[t][j] = abs(node[t+1][j][1]-node[t][j][1])
                spaceNumeratorZ[t][j] = abs(node[t+1][j][2]-node[t][j][2])
          spaceNumeratorX_T = np.transpose(spaceNumeratorX)
          spaceNumeratorY_T = np.transpose(spaceNumeratorY)
          spaceNumeratorZ_T = np.transpose(spaceNumeratorZ)
          spaceNumeratorX_sum = np.sum(spaceNumeratorX_T, axis=1)
          spaceNumeratorY_sum = np.sum(spaceNumeratorY_T, axis=1)
          spaceNumeratorZ_sum = np.sum(spaceNumeratorZ_T, axis=1)

          for j in range(len(node[0])):
            s_v = []
            # print("spaceNumeratorX[j]:"+str(spaceNumeratorX[j]))
            denominatorX = abs(spaceNumeratorX[j][0] - spaceNumeratorX[j][-1])
            denominatorY = abs(spaceNumeratorY[j][0] - spaceNumeratorY[j][-1])
            denominatorZ = abs(spaceNumeratorZ[j][0] - spaceNumeratorZ[j][-1])

            if denominatorX!=0:spaceDenominatorX.append(denominatorX)
            if denominatorY!=0:spaceDenominatorY.append(denominatorY)
            if denominatorZ!=0:spaceDenominatorZ.append(denominatorY)

            # print(j)
            s_v.append(spaceNumeratorX_sum[j]/spaceDenominatorX[j])
            s_v.append(spaceNumeratorY_sum[j]/spaceDenominatorY[j])
            s_v.append(spaceNumeratorZ_sum[j]/spaceDenominatorZ[j])

            s = np.linalg.norm(s_v)
            if s > self.limits[2][j]:
                s_vector.append(self.limits[2][j])
            # elif s < self.limits[3][j]:
            #     s_vector.append(self.limits[3][j])
            elif np.isnan(s):
                s_vector.append(1)
            else:
                s_vector.append(s)
          spacevector.append(s_vector)
        if len (spacevector) == 0:
            # print("0")
            space = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            return space
        # print("spacevector:"+str(spacevector))
        spacevector_log = np.log(spacevector)
        # print("spacevector_log:"+str(spacevector_log))
        # 標準化
        spacevector_log_std = self.sc_s.transform(spacevector_log)
        # spacevector = self.min_max(spacevector, 0)
        # print("spacevector_log_std:"+str(spacevector_log_std))

        # 累積分布関数のやつ：
        spacevector_norm = norm.cdf(spacevector_log_std, loc=0, scale=1)
        # print("spacevector_y_norm:"+str(spacevector_y))
        # ロジスティック関数のやつ：
        spacevector_logistic = self.logistic(spacevector_log_std, 1, 1, 0)
        # print("spacevector_y_norm:"+str(spacevector_y))
        # ロジスティック関数のやつ：
        # spacevector_y = self.logistic(spacevector_log_std, 1, 1, 0)
        # print("spacevector_y_logistic:"+str(spacevector_y))
        space = np.sum(spacevector_norm,axis=0)/len(spacevector_norm)
        # print("calc_Space:finish")
        return space

    def calc_Time(self, accelVector):
        """加速度"""
        # print("calc_Time:start")
        time = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # for accel in accelVector:
        #     for i in range(len(accel)):
        #         time[i] += accel[i]
        time=np.sum(accelVector,axis=0)/len(accelVector)
        # time = self.min_max(time)
        # print("time:"+str(time))
        # print("calc_Time:finish")
        return time

    def calc_Weight(self, Nodes):
        """速度"""
        print("calc_Weight:start")
        print("calc_Weight:finish")

    def calc_Flow(self, flowVector):
        """力の変化量"""
        # print("calc_Flow:start")
        flow = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # for jerk in flowVector:
        #     for i in range(len(jerk)):
        #         flow[i] += jerk[i]
        flow=np.sum(flowVector,axis=0)/len(flowVector)
        # flow = self.min_max(flow)
        # print("flow:"+str(flow))
        # print("calc_flow:finish")
        return flow



# """test"""

# filename = "C:\\Users\gorim\Desktop\Motion-Aware-Sequencer\keypoints3d\gHO_sFM_cAll_d20_mHO2_ch10.pkl"
# with open(filename, 'rb')as web:
#             pre_nodes = pickle.load(web)
#         # 初期化
# LMA = LabanFutureValue(filename)

# print("Nodes[time][node][x,y,z]")
# print("Nodes:"+ str(len(LMA.nodes)))
# print("Nodes[0]:"+ str(len(LMA.nodes[0])))
# print("Nodes[0][0]:"+ str(len(LMA.nodes[0][0])))

# LMA.OnInterval()
# print("limits".format(LMA.limits[0]))

# print("space:",format(LMA.spaceVector))

# e = LMA.excitement