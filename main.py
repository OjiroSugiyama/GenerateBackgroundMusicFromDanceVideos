import tkinter as tk
from view import View


def main():
    root = tk.Tk()
    view = View(master=root)
    view.mainloop()

if __name__ == "__main__":
    main()
    # excitement = Csv_Reader("C:\\Users\gorim\Desktop\Motion-Aware-Sequencer\output-20211016T031203Z-001\output\gHO_sFM_cAll_d19_mHO4_ch05.csv")
    # nodes, frame = excitement.csv_read()
    # excitement.calcMotionExcitement(nodes, frame)
    # print("excitement.motion_excitement_array:"+str(excitement.motion_excitement_array))
    # excitement.motion_excitement_array:[ 7.34200221 14.23878162 18.28757119  9.23919891  6.35620429 14.35150375 0.81573965]