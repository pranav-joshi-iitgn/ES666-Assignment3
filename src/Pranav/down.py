import pdb
import glob
from numpy import *
import cv2
import os
import matplotlib.pyplot as plt
if __name__=="__main__":
    for j in range(1,7):
        imf = path = f"ES666-Assignment3/Images/I{j}"
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        images = [cv2.imread(img) for img in all_images] #ndarray
        if j==1:images = [cv2.resize(img,(0,0),fx=0.1,fy=0.1) for img in images]
        else:images = [cv2.resize(img,(0,0),fx=0.7,fy=0.7) for img in images]
        #plt.imshow(images[0])
        #plt.show()
        for i in range(len(images)):
            cv2.imwrite(f"ES666-Assignment3/Images/I{j}_down/image{i}.png",images[i])