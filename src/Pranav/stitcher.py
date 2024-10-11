import pdb
import glob
from numpy import *
import cv2
import os
import matplotlib.pyplot as plt
#from src.Pranav import some_function
#from src.Pranav.some_folder import folder_func

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        images = [cv2.imread(img) for img in all_images] #ndarray
        grays = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in images]
        #some_function.some_func()
        #folder_func.foo()

        # Collect all homographies calculated for pair of images and return
        homography_matrix_list =[]
        for i in range(len(grays)):
            img1 = grays[i]
            img2 = grays[i+1]
            H = self.Homo(img1,img2)
            homography_matrix_list.append(H)
        #homography_matrix_list = [[homography_matrix_list[max(i,j)][min(i,j)] for j in range(len(grays))] for i in range(len(grays))]
        # Return Final panaroma\
        # I1 = images[0]
        # I2 = images[1]
        # img1= grays[0]
        # img2= grays[1]
        # H =self.Homo(img1,img2,0.5,False,I1,I2)
        stitched_image =images[0]
        return stitched_image, homography_matrix_list 

    def Homo(self,img1,img2,th=0.2,visualise=False,I1=None,I2=None):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        bf = cv2.BFMatcher() #Brute Force matcher
        matches = bf.knnMatch(des1,des2,k=2) #Need k>1 to use ratio test for betterprediction
        good= []
        for t in matches:
            m = t[0]
            n = t[-1]
            if m.distance < th*n.distance:
                good.append([m])
        good = good
        if visualise:
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure()
            plt.imshow(img3)
        good = [(m[0].queryIdx,m[0].trainIdx) for m in good]
        good = [cv2.KeyPoint.convert([kp1[i],kp2[j]]) for (i,j) in good]
        good = [[x[1],x[0]]+[y[1],y[0]] for (x,y) in good]
        #good = [list(y.convert())+list(x.convert()) for (x,y) in good]
        good = array(good,dtype=int32)
        #The pixel coordinates are floats unfortunately .. but are with correct scale
        H = self.RanSac(good)
        if visualise:
            X = array([list(x[:2]) + [1] for x in good])
            kp22 = X @ H.T
            kp22 = int32(around(kp22/(kp22[:,2][:,newaxis])))
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            I1 =0*I1
            I2 =0*I2
            for k,x in enumerate(good):
                try:
                    for i in range(-7,8):
                        for j in range(-7,8):
                            #img1[x[1]+i,x[0]+j] =255
                            I1[x[0]+i,x[1]+j]=[255,255,255]
                            I2[x[2]+i,x[3]+j]=[255,255,255]
                except:pass
                try:
                    for i in range(-2,3):
                        for j in range(-2,3):
                            I2[kp22[k][0]+i,kp22[k][1]+j]=[255,0,0]
                except:pass
            ax1.imshow(I1,cmap="gray")
            ax2.imshow(I2,cmap="gray")
            #ax1.scatter(good[:,0],good[:,1])
            #ax2.scatter(good[:,2],good[:,3])
        return H

    def findH(self,matches):
        A = []
        for p in matches:
            x1,y1,x2,y2 =p
            z1,z2 = 1,1
            #Makinng A (2*9)
            # [0,-z2,y2]
            # [z2,0,-x2]
            # [-z2,x2,0]
            Ai = [
                [0*x1,0*y1,0*z1,-z2*x1,-z2*y1,-z2*z1,y2*x1,y2*y1,y2*z1],
                [z2*x1,z2*y1,z2*z1,0*x1,0*y1,0*z1,-x2*x1,-x2*y1,-x2*z1],
            ]
            A.extend(Ai)
        A = array(A)
        print(A.shape)
        U,Sigma,Vh = linalg.svd(A)
        h = Vh[-1]
        print(linalg.norm(h))
        H = array([h[:3],h[3:6],h[6:]])
        print(H)
        return H

    def RanSac(self,S,N=0,t=None,T=None,s=8,initial_s=10000):
        X = array([list(x[:2]) + [1] for x in S])
        Y = array([list(x[2:]) + [1] for x in S])
        ind = random.randint(0,len(S),min(initial_s,len(S)))
        H = self.findH(S[ind]) # 40 is just an arbitary number..
        Ypred = X @ H.T
        Ypred = Ypred /Ypred[:,2][:,newaxis]
        d = (Ypred - Y)**2
        d =sum(d,axis=-1)
        t = mean(d)**0.5 #This is just a value I am setting arbitarily.
        d = d**0.5
        # I know, we should use_/5.99 sigma .. but we don't have any prior information of the system
        inliers = d < t
        inliers = [i for i in range(len(inliers)) if inliers[i]]
        T = len(inliers)
        # this also gives us w .. but without p, we can't use it to find N
        indi = [ind]
        cardSi = [T]
        for i in range(N):
            print(f"{i}th trial")
            ind = random.randint(0,len(S),s)
            Sit = S[ind]
            #indi.append(ind)
            Hi =self.findH(Sit)
            Ypred = X @ H.T
            Ypred = Ypred /Ypred[:,2][:,newaxis]
            d = (Ypred - Y)**2
            d =sum(d,axis=-1)
            d =d**0.5
            inliers = d<t
            inliers = [j for j in range(len(inliers)) if inliers[j]]
            cardSi.append(len(inliers))
            indi.append(inliers)
            if len(inliers) > T:
                Si = S[inliers]
                H = self.findH(Si)
                print("found")
                return H
        print("picking best")
        i = argmax(cardSi)
        ind = indi[i]
        Sbest =S[ind]
        H =self.findH(Sbest)
        return H

    def stitch_pair(self,img1,img2):
        H = self.Homo(img1,img2)
        rows,cols = img1.shape
        X = []
        I= []
        for y in range(cols):
            X.extend([[x,y,1] for x in range(rows)])
            I.extend([img1[x,y] for x in range(rows)])
        X = array(X)
        I = array(I)
        Ypred = X @ H.T
        Ypred = Ypred/(Ypred[:,2][:,newaxis])
        Ypred = around(Ypred[:,:2])
        Ypred = int32(Ypred)
        left = min(Ypred[:,1])
        up = min(Ypred[:,0])
        right = max(Ypred[:,1])
        down = max(Ypred[:,0])
        left = min(left,0)
        print(left,up,right,down)
        up = min(up,0)
        right = max(right,img2.shape[1])
        down = max(down,img2.shape[0])
        assert left <= 0,left
        assert up <= 0, up
        img2 = concatenate((
            zeros((img2.shape[0],int(-left)),dtype=int8),
            img2,
            zeros((img2.shape[0],int(right-img2.shape[1])),dtype=int8)
            ),axis=1)
        img2 = concatenate((
            zeros((-up,img2.shape[1]),dtype=int8),
            img2,
            zeros((int(down-img2.shape[0]),img2.shape[1]),dtype=int8)
            ),axis=0)
        for i,x in enumerate(Ypred):
            try:
                img2[x[0]-up,x[1]-left] =I[i]
            except:pass
        plt.imshow(img2)

if __name__=="__main__":
    Pan = PanaromaStitcher()
    #Pan.make_panaroma_for_images_in("ES666-Assignment3/Images/I1")
    imf = path = "ES666-Assignment3/Images/I1"
    all_images = sorted(glob.glob(imf+os.sep+'*'))
    print('Found {} Images for stitching'.format(len(all_images)))
    ####  Your Implementation here
    #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
    #### Just make sure to return final stitched image and all Homography matrices from here
    images = [cv2.imread(img) for img in all_images] #ndarray
    grays = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in images]
    Pan.stitch_pair(grays[0],grays[1])#,I1=images[0],I2=images[1],visualise=True)
    plt.show()