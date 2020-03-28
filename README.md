# Hough_lines_from_scratch
This repo contain an implementation for hough lines from scratch using python
## Description for the Files:
* The "images" directory contains several images and the lines detected on them using this code (CV404Hough.py)
* The "trial from the internet" directory contains 3 files found on several websites (that are mentioned at the top of the .py file) but didn't work well for me
* The [CV404Hough.py](https://github.com/hosnaa/Hough_lines_from_scratch/blob/master/CV404Hough.py) file is the one that I updated with the help of some links (can be seen at the top of the .py file). What you need to specify to run this code safely on your PC:
    * you need to specify the image_path which is the path of the image you want to try the code on (as trial_image in images direc)
    * The save_path to save the hough space figure (as fig2 in images direc)
    * The savepathLines to save your image but with houghlines superimposed on it (as hough_lines3 in images direc)
    * you specify the thresholds of Canny to be used (here I made it 50 & 150)
    * you need to specify the number of lines you want to be detected while calling the "superimpose_maxima_on_cartesian" function (here I specified it to 15)

* The "CV404Hough_slower.py" file is the same as the one above, having exactly the same functions except for the (hough_lines) one, that is used for the hough space, it's just slower but might be more understandable.

## Functions in brief:
1) "rgb2gray": simple RGB2gray function to change the colored image to grayscale (many other ways can be used as 'cv2.imread(...,0)' or Image.open(...,'L'))
2) "k_largest_index_argsort": function to return the indices of N max values in an ndarray (check this [website](https://stackoverflow.com/questions/43386432/how-to-get-indexes-of-k-maximum-values-from-a-numpy-multidimensional-array) for an example) it takes the ndarray that you want to investigate and the N indices for N max values that you want to extract (which are number of lines in our case) and returns ndarray containing these indices (here it returned 2d array, the 1st col is for the rhos indices and 2nd col for the thetas indices)
3) "hough_line": is the core function for hough space, you'll find detailed comments for each step (explaining it here would be troublesome :D), you give it the image and save paths and the 2 thresholds for canny.
4) "superimpose_maxima_on_cartesian": firstly it uses the (2)nd function and uses the returned indices of the maximas (cells that have maximum values in the accumulator) to get their corresponding rhos and thetas and use them for drawing using the default method for drawing at cv2.houghlines.

##### Note: the accumulator is a 2d array where the row is for the rhos indices, cols for thetas indices(0-->180) and the value of the cells is the voting, as the value increases this means more voting which means that the corresponding rho and theta of this cell (its i and j) most likely denote a straight line in the cartesian space. These votes are the number of points on the line of that specific rho and theta. (for basic understanding check [wikipedia: Theory and Implementation](https://en.wikipedia.org/wiki/Hough_transform) and [OpenCv_houghlines](https://www.learnopencv.com/hough-transform-with-opencv-c-python/))