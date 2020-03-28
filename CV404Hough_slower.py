# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:34:50 2020

https://sbme-tutorials.github.io/2018/cv/notes/5_week5.html

@author: hosna
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def k_largest_index_argsort(two_dim_arr, k):  #k is number of lines (number of indices of max. values you want to return)
    idx = np.argsort(two_dim_arr.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, two_dim_arr.shape))

def houghLine(img_path, thresh1, thresh2):
    ''' Basic Hough line transform that builds the accumulator array
    Input : image : edge image (canny)
    Output : accumulator : the accumulator of hough space
             thetas : values of theta (-90 : 90)
             rs : values of radius (-max distance : max distance)
    '''
    img_orig = imageio.imread(img_path)
    if img_orig.ndim == 3:
        img_orig = rgb2gray(img_orig)
    image = cv2.Canny(img_orig, thresh1, thresh2)
        #Get image dimensions
    # y for rows and x for columns 
    Ny = image.shape[0]
    Nx = image.shape[1]

    #Max diatance is diagonal one 
    Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
     # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90))
    #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    accumulator = np.zeros((2 * Maxdist, len(thetas)))
    for y in range(Ny):
     for x in range(Nx):
         # Check if it is an edge pixel
         #  NB: y -> rows , x -> columns
          if image[y,x] > 0:
              for k in range(len(thetas)):
                  print('entered')
                  r = int(round(x*(np.cos(thetas))[k] + y * (np.sin(thetas))[k]))
                  accumulator[int(r) + Maxdist,k] += 1
     
    return accumulator, thetas, rs

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
            
image_path ='D:/Biometrics/trials/cv404-2020-assignment-02-sbe404-2020-team19/Hosna_trials_2/trial_image.jpg' 
savepathLines = 'D:/Biometrics/trials/cv404-2020-assignment-02-sbe404-2020-team19/Hosna_trials_2/hough_lines_4.png'
accumulator, thetas, rhos = houghLine(image_path, 50,150)
drawn_lines = superimpose_maxima_on_cartesian(accumulator,rhos,thetas,image_path,savepathLines, 15)
