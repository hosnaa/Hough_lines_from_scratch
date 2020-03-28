# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:23:42 2020

##Basic understanding: several links but found wikipedia somehow good along with this one 
(https://www.learnopencv.com/hough-transform-with-opencv-c-python/)
##houghLines: https://alyssaq.github.io/2014/understanding-hough-transform/
##sort 2D: https://stackoverflow.com/questions/43386432/how-to-get-indexes-of-k-maximum-values-from-a-numpy-multidimensional-array

@author: hosna
"""

import numpy as np
import imageio
import math
import cv2
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def k_largest_index_argsort(two_dim_arr, k):  #k is number of lines (number of indices of max. values you want to return)
    idx = np.argsort(two_dim_arr.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, two_dim_arr.shape))

def hough_line(img_path, thresh1, thresh2 ,save_path, lines_are_white=True, value_threshold=5):
    #read image and get its canny edges
    img_orig = imageio.imread(img_path)
    if img_orig.ndim == 3:
        img_orig = rgb2gray(img_orig)
    img = cv2.Canny(img_orig, thresh1, thresh2)
 
#1) initialize Hough parameter space rs, thetas
    thetas = np.deg2rad(np.arange(-90.0, 90.0)) #Theta in range from -90 to 90 degrees
    width, height = img.shape #width: row, height: column 
    #round(5.1) = 5,, round(5.9) = 6
    diag_len = int(round(math.sqrt(width * width + height * height)))
    #linear space: Return evenly spaced numbers over a specified interval
    #np.linspace(2.0, 3.0, num=5) ---> array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # save some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

#2) Create accumulator array and initialize to zero 
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)

#3) Check if it is an edge pixel
  # value_when_true if condition else value_when_false
  # are_edges =[false...True..], I can't understand how is it going actually but it's much faster than the 3 for loops
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    #Return the indices of the elements that are non-zero/not-false
    y_idxs, x_idxs = np.nonzero(are_edges)

#4) Loop over each edge pixel (edge_row_idx: has the total array while x has each value)
    for i in range(len(x_idxs)): #or len(y_idx) they're the same
        x = x_idxs[i]
        y = y_idxs[i]
#5) Map edge pixel of edges to hough space
        for t_idx in range(num_thetas):
            # Calculate rho. N.B: r has value -max to max, so we need to map r to its idx 0 : 2*max// for postive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1
#6) show the hough space--> each pixel mapped to curve with r and theta
    plt.imshow(accumulator, cmap='jet', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    plt.savefig(save_path)
    plt.show()
    return accumulator, rhos, thetas 

def superimpose_maxima_on_cartesian (accumulator_hough, rho_hough, theta_hough, img_path, save_path_lines, lines):
    #for i in range(lines):
    #arr.argsort()[-3:][::-1]
    #idx = np.argpartition(a.ravel(),a.size-k)[-k:]
    #return np.column_stack(np.unravel_index(idx, a.shape))
    #idx = accumulator.argsort()
    #print (idx)
    
    img2 = cv2.imread(img_path)
    idx = k_largest_index_argsort(accumulator, lines)  #from this we get 2d array containing index of r(1st col) & theta(2nd col), corresponding to the biggest values in the accumulator
#    i_rho= idx[:,0]
#    j_theta = idx[:,1]
#    for j in idx[0][:lines-1]:
    for idx_rho,idx_theta in zip(idx[:,0], idx[:,1]): #or in idx (directly)
        rho_max = int(rhos[idx_rho])
        theta_max = thetas[idx_theta]
        # rho_max = rhos[idx_rho]
        # theta_max = thetas[idx_theta]
        print("rho={0:.0f}, theta={1:.0f}".format(rho_max, np.rad2deg(theta_max)))
        a = np.cos(theta_max)
        b = np.sin(theta_max)
        x0 = a*rho_max
        y0 = b*rho_max
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        print (x1,y1, x2,y2)
        print('-----------------')
        my_lines = cv2.line(img2,(x1,y1), (x2,y2),(0,0,255),2)
    #cv2.imshow('window',img2)
    #cv2.waitKey(0)
    cv2.imwrite(save_path_lines, img2)
    return my_lines

imgpath = 'D:/Biometrics/trials/cv404-2020-assignment-02-sbe404-2020-team19/Hosna_trials_2/binary_crosses.png.gif'
savepath = 'D:/Biometrics/trials/cv404-2020-assignment-02-sbe404-2020-team19/Hosna_trials_2/fig.png'
savepathLines = 'D:/Biometrics/trials/cv404-2020-assignment-02-sbe404-2020-team19/Hosna_trials_2/hough_lines_4.png'
accumulator,rhos, thetas = hough_line(imgpath, 50,150, savepath)
drawn_lines = superimpose_maxima_on_cartesian(accumulator,rhos,thetas,imgpath,savepathLines, 15)


