# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:53:48 2020

https://gist.github.com/ilyakava/c2ef8aed4ad510ee3987

@author: hosna
"""
import numpy as np
import imageio
import math
import cv2
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def hough_line(img_path, thresh1, thresh2 ,save_path, lines_are_white=True, value_threshold=0):
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
  # are_edges =[false...True..], I can't understand how is it going
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
    
    return accumulator, thetas, rhos

def top_n_rho_theta_pairs(ht_acc_matrix, n, rhos, thetas):
  '''
  @param hough transform accumulator matrix H (rho by theta)
  @param n pairs of rho and thetas desired
  @param ordered array of rhos represented by rows in H
  @param ordered array of thetas represented by columns in H
  @return top n rho theta pairs in H by accumulator value
  @return x,y indexes in H of top n rho theta pairs
  '''
  flat = list(set(np.hstack(ht_acc_matrix)))
  flat_sorted = sorted(flat, key = lambda n: -n)
  coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat_sorted[0:n]]

  rho_theta = []
  x_y = []

  for coords_for_val_idx in range(0, len(coords_sorted), 1):
    coords_for_val = coords_sorted[coords_for_val_idx]
    for i in range(0, len(coords_for_val), 1):
      n,m = coords_for_val[i] # n by m matrix
      rho = rhos[n]
      theta = thetas[m]
      rho_theta.append([rho, theta])
      x_y.append([m, n]) # just to unnest and reorder coords_sorted
  return [rho_theta[0:n], x_y]


#def valid_point(pt, ymax, xmax): #not correct: doesn't work even on the true built-in
#  '''
#  @return True/False if pt is with bounds for an xmax by ymax image
#  '''
#  x, y = pt
#  if x <= xmax and x >= 0 and y <= ymax and y >= 0:
#    return True
#  else:
#    return False

def round_tup(tup):
  '''
  @return closest integer for each number in a point for referencing
  a particular pixel in an image
  '''
  x,y = [int(round(num)) for num in tup]
  return (x,y)

def draw_rho_theta_pairs(target_im, pairs):
  '''
  @param opencv image
  @param array of rho and theta pairs
  Has the side-effect of drawing a line corresponding to a rho theta
  pair on the image provided
  '''
  im_y_max, im_x_max, channels = np.shape(target_im)
  for i in range(0, len(pairs), 1):
    point = pairs[i]
    rho = point[0]
    theta = point[1] * np.pi / 180 # degrees to radians
    # y = mx + b form
    m = -np.cos(theta) / np.sin(theta)
    b = rho / np.sin(theta)
    # possible intersections on image edges
    left = (0, b)
    right = (im_x_max, im_x_max * m + b)
    top = (-b / m, 0)
    bottom = ((im_y_max - b) / m, im_y_max)
    print(left, right, top, bottom)
    print('---------------------------')

    pts = [pt for pt in [left, right, top, bottom] if valid_point(pt, im_y_max, im_x_max)]
    if len(pts) == 2:
      cv2.line(target_im, round_tup(pts[0]), round_tup(pts[1]), (0,0,255), 1)
      



imgpath = 'D:/Biometrics/trials/cv404-2020-assignment-02-sbe404-2020-team19/Hosna_trials_2/pentagon.png'
savepath = 'D:/Biometrics/trials/cv404-2020-assignment-02-sbe404-2020-team19/Hosna_trials_2/fig.png'
H, thetas, rhos = hough_line(imgpath, 50,150, savepath)

img = cv2.imread ('pentagon.png')
edges = cv2.Canny(img, 50, 150)
rho_theta_pairs, x_y_pairs = top_n_rho_theta_pairs(H, 22, rhos, thetas)
im_w_lines = img.copy()
draw_rho_theta_pairs(im_w_lines, rho_theta_pairs)

# also going to draw circles in the accumulator matrix

#for i in range(0, len(x_y_pairs), 1):
#  x, y = x_y_pairs[i]
#  cv2.circle(img = H, center = (x, y), radius = 12, color=(0,0,0), thickness = 1)


plt.subplot(141),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(edges,cmap = 'gray')
plt.title('Image Edges'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(H)
plt.title('Hough Transform Accumulator'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(im_w_lines)
plt.title('Detected Lines'), plt.xticks([]), plt.yticks([])



plt.show()