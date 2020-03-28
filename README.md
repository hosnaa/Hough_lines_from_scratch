# Hough_lines_from_scratch
This repo contain an implementation for hough lines from scratch using python
## Description for the Files:
* The "images" directory contains several images and the lines detected on them using this code (CV404Hough.py)
* The "trial from the internet" directory contains 3 files found on several websites (that are mentioned at the top of the .py file) but didn't work well for me
* The [CV404Hough.py]() file is the one that worked for me and generated these images:
    * you need to specify the image_path which is the path of your image
    * The save_path to save the hough space figure
    * The savepathLines to save the image with lines superimposed on it
    * you specify the thresholds of Canny to be used (here I made it 50 & 150)
    * you need to specify the number of lines you want to be detected while calling the "superimpose" function (here I specified it to 32)

* The "CV404Hough_slower.py" file is the same as the one above, having exactly the same function except for the (hough_lines) one, that is used for the hough space, it's just slower but might be more understandable.