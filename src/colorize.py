import os, sys
import cv2
import numpy as np
import scipy.sparse
import imageio
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import logging

class Colorization():
    def __init__(self, ori_file, mk_file):
        self.execute(ori_file, mk_file)

    def execute(self, ori_file, mk_file):
        self.ori_img = cv2.imread(ori_file).astype(np.float32)/255
        self.ori_yuv = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2YUV)
        self.res_path = ori_file[:-4] + "_res.bmp"

        self.mk_img = cv2.imread(mk_file).astype(np.float32)/255
        self.mk_yuv = cv2.cvtColor(self.mk_img, cv2.COLOR_BGR2YUV)

        self.rows = self.ori_img.shape[0]
        self.cols = self.ori_img.shape[1]
        self.img_size = self.rows * self.cols

        self.y_ori = self.ori_yuv[:,:,0]
        self.u_mk = self.mk_yuv[:,:,1].reshape(self.img_size)
        self.v_mk = self.mk_yuv[:,:,2].reshape(self.img_size)

        self.get_mk_pos()
        self.color()


    def get_mk_pos(self):
        assert self.ori_img.shape == self.mk_img.shape, 'not the same image'
        self.mk_pos = np.zeros((self.ori_img.shape[0], self.ori_img.shape[1]))
        for i in range(self.mk_pos.shape[0]):
            for j in range(self.mk_pos.shape[1]):
                if(self.ori_img[i][j][0] != self.mk_img[i][j][0]):
                    self.mk_pos[i][j] = 1
    
    def color(self):
        weight_data = []
        col_inds = []
        row_inds = []
        print("Building weights...")
        for i in range(self.rows):
            for j in range(self.cols):
                if self.mk_pos[i][j] == 0:
                    neighbor_value = []
                    for s in range(i-1, i+2):
                        for t in range(j-1, j+2):
                            if(0 <= s and s < self.rows-1 and 0 <= t and t < self.cols-1):
                                if (i != s) | (j != t):
                                    neighbor_value.append(self.y_ori[i,j])
                                    row_inds.append(i*self.cols+j)
                                    col_inds.append(s*self.cols+t)
                    sig = np.var(np.append(neighbor_value, self.y_ori[i,j]))
                    if sig < 1e-6:
                        sig = 1e-6
                    # print(len(neighbor_value))
                    wrs = np.exp(- np.power(neighbor_value - self.y_ori[i][j], 2)/ sig)
                    wrs = - wrs / np.sum(wrs)
                    for item in wrs:
                        weight_data.append(item)
                    # print(len(row_inds), len(col_inds), len(weight_data))
                
                weight_data.append(1)
                row_inds.append(i * self.cols + j)
                col_inds.append(i * self.cols + j)
            
        print("Solving matrix...")
        A = scipy.sparse.csc_matrix( (weight_data, (row_inds, col_inds)), shape=(self.img_size, self.img_size) )

        b_u = np.zeros(self.img_size)
        b_v = np.zeros(self.img_size)
        mk_pos_vec = np.nonzero(self.mk_pos.reshape(self.img_size))
        b_u[mk_pos_vec] = self.u_mk[mk_pos_vec]
        b_v[mk_pos_vec] = self.v_mk[mk_pos_vec]
        u_color = scipy.sparse.linalg.spsolve(A, b_u).reshape((self.rows, self.cols))
        v_color = scipy.sparse.linalg.spsolve(A, b_v).reshape((self.rows, self.cols))

        colored = np.dstack((self.y_ori.astype(np.float32), u_color.astype(np.float32), v_color.astype(np.float32)))
        colored = cv2.cvtColor(colored, cv2.COLOR_YUV2RGB)
        
        print("Solving finished!!!")
        for i in range(colored.shape[0]):
            for j in range(colored.shape[1]):
                for t in range(3):
                    if(colored[i][j][t] > 1):
                        colored[i][j][t] = 1
                    if(colored[i][j][t] < 0):
                        colored[i][j][t] = 0

        fig = plt.figure()
        fig.add_subplot(1,3,1).set_title("Grey Image")
        imgplot = plt.imshow(self.ori_img)
        self.mk_img = cv2.cvtColor(self.mk_img, cv2.COLOR_RGB2BGR)
        fig.add_subplot(1,3,2).set_title('Color Hints')
        imgplot = plt.imshow(self.mk_img)
        fig.add_subplot(1,3,3).set_title('Result')
        imgplot = plt.imshow(colored)
        plt.show()

        imageio.imwrite(self.res_path, colored)



if __name__ == "__main__":
    color = Colorization("../image/hair_res.bmp", "../image/hair_res_mk.bmp")
    # color.execute()
