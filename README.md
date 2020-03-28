# Hough_lines_from_scratch
This repo contain an implementation for hough lines from scratch using python
## Description for the Files:
* The "images" directory contains several images and the lines detected on them using this code (CV404Hough.py)
* The "trial from the internet" directory contains 3 files found on several websites (that are mentioned at the top of the .py file) but didn't work well for me
* The [CV404Hough.py](https://github.com/hosnaa/Hough_lines_from_scratch/blob/master/CV404Hough.py) file is the one that I updated with the help of some links (can be seen at the top of the .py file) and generated these images:
    * you need to specify the image_path which is the path of your image
    * The save_path to save the hough space figure
    * The savepathLines to save the image with lines superimposed on it
    * you specify the thresholds of Canny to be used (here I made it 50 & 150)
    * you need to specify the number of lines you want to be detected while calling the "superimpose_maxima_on_cartesian" function (here I specified it to 15)

* The "CV404Hough_slower.py" file is the same as the one above, having exactly the same function except for the (hough_lines) one, that is used for the hough space, it's just slower but might be more understandable.

## Functions in brief:
1) "rgb2gray": simple RGB2gray function to change the colored image to grayscale (many other ways can be used as 'cv2.imread(...,0)' or Image.open(...,'L'))
2) "k_largest_index_argsort": function to return the indices of N max values in an ndarray (check the website for example ) it takes the ndarray that you want to investigate and the N indices that you want to extract (which are number of lines in our case) and returns ndarray containing the indices (here it returned 2d array, the 1st col is for the rhos indices and 2nd col for the thetas indices)
3) "hough_line": is the core function for hough space, you'll find detailed comments for each step (explaining it here would be troublesome :D), you give it the image and save paths and the 2 thresholds for canny.
4) "superimpose_maxima_on_cartesian": firstly it uses the (2)nd function and uses the indices of the maximas (cells that have maximum values in the accumulator) to get their corresponding rhos and thetas and use them for drawing using the default method for drawing at cv2.houghlines

##### Note: the accumulator is a 2d array where the row is for the rhos indices, cols for thetas indices(0-->180) and the value of the cells is the voting, as the value increase this means more voting that this cell with its corresponding rho and theta [i,j] most likely is a straight line. these votes are the number of points on the line of that specific rho and theta. (for basic understanding check [wikipedia: Theory and Implementation](https://en.wikipedia.org/wiki/Hough_transform) and [OpenCv_houghlines](https://www.learnopencv.com/hough-transform-with-opencv-c-python/))