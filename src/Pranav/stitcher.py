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

    def make_panaroma_for_images_in(self,path,th=0.5,limit=1000,careful=False,t=1):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        n = len(all_images)
        print(f'Found {n} Images for stitching')

        images = [cv2.imread(img) for img in all_images] #ndarray
        grays = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in images]
        mean_bright = [mean(x,axis=None) for x in grays]
        #print(mean_bright)
        mean_mean_bright = mean(mean_bright)
        grays = [uint8(mean_mean_bright *grays[i]/mean_bright[i]) for i in range(len(grays))]
        homography_matrix_list =[]

        print("Making Left half")
        leftIm = grays[0]
        origin1 = array([0,0,0])
        for i in range(n//2 - 1):
            print("Calculating Homography")
            H = Pan.Homo(grays[i],grays[i+1],
            trials=10000,t=t,T=None,s=10,initial_s=1000,th=th,limit=limit)
            homography_matrix_list.append(H)
            print("Stitching")
            leftIm,origin1 = Pan.stitch_pair(leftIm,grays[i+1],H,origin1,epsilon=0,th=th,Xlim=limit)
        #plt.figure()
        #plt.imshow(leftIm)
        #plt.plot([origin1[1]],[origin1[0]],'ro')
        #plt.title("left")
        print("Making Right half")
        rightIm = grays[n-1]
        origin2= array([0,0,0])
        for i in range(n-1,n//2,-1):
            print("Calculating Homography")
            H = Pan.Homo(grays[i],grays[i-1],
            trials=10000,t=t,T=None,s=10,initial_s=1000,th=th,limit=limit)
            homography_matrix_list.append(H)
            print("Stitching")
            rightIm,origin2 = Pan.stitch_pair(rightIm,grays[i-1],H,origin2,epsilon=0,th=th,Xlim=limit)
            col = rightIm.shape[1]
            rightIm = rightIm[:,:limit]
        print("Homography between centermost images")
        H = Pan.Homo(grays[n//2-1],grays[n//2],
        trials=10000,t=t,T=None,s=4,initial_s=1000,th=th,limit=limit)#,visualise=True)
        behind,origin = Pan.stitch_pair(
            grays[n//2 -1],grays[n//2],H,
            epsilon=0,th=th,
            Xlim=limit,Ylim=limit)
        homography_matrix_list.append(H)
        print("Stitching Left and Right")
        im,origin = Pan.stitch_pair(
            leftIm,rightIm,H,
            origin1,origin2=origin2,
            epsilon=0,th=th,
            Xlim=2*limit,Ylim=limit)
        print("Done with",path)
        return im, homography_matrix_list

    def Homo(self,img1,img2,th=0.5,th2=None,visualise=False,I1=None,I2=None,trials=1000,t=None,T=None,s=8,initial_s=100,limit=1000,return_t=False):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        bf = cv2.BFMatcher() #Brute Force matcher
        matches = bf.knnMatch(des1,des2,k=2) #Need k>1 to use ratio test for better prediction
        good= []
        for x in matches:
            m = x[0]
            n = x[-1]
            if m.distance < th*n.distance:
                if th2 is None or m.distance < th2:
                    good.append([m])
        good0 = good
        if visualise:
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure()
            plt.imshow(img3)
        good = [(m[0].queryIdx,m[0].trainIdx) for m in good]
        good = [cv2.KeyPoint.convert([kp1[i],kp2[j]]) for (i,j) in good]
        good = [[x[1],x[0]]+[y[1],y[0]] for (x,y) in good]
        good = array(good,dtype=int32)
        N,M = img1.shape
        # T1 =[
        #     [2/N,0,-1],
        #     [0,2/M,-1],
        #     [0,0,1]
        # ]
        # T2 =[
        #     [2/img2.shape[0],0,-1],
        #     [0,2/img2.shape[1],-1],
        #     [0,0,1]
        # ]
        # if limit is None:
        #     limit = max(img2.shape)
        #     limit = max(limit,N,M)
        # T1 = array(T1)
        # T2 = array(T2)
        #H = self.RanSac(good,trials,t,T,s,initial_s,T1=T1,T2=T2,N=N,M=M,limit=limit)
        H,ind,t = self.RanSac(good,trials,t,T,s,initial_s,T1=None,T2=None,N=N,M=M,limit=limit,return_t=True)
        if visualise:
            good = [good0[i] for i in ind]
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure()
            plt.imshow(img3)
        if return_t:return H,t
        return H

    def findH(self,matches,T1=None,T2=None):
        A = []
        for p in matches:
            x1,y1,x2,y2 = p
            z1,z2 = 1,1
            Ai = [
                [0*x1,0*y1,0*z1,-z2*x1,-z2*y1,-z2*z1,y2*x1,y2*y1,y2*z1],
                [z2*x1,z2*y1,z2*z1,0*x1,0*y1,0*z1,-x2*x1,-x2*y1,-x2*z1],
            ]
            A.extend(Ai)
        A = array(A)
        U,Sigma,Vh = linalg.svd(A)
        h = Vh[-1]
        H = array([h[:3],h[3:6],h[6:]])
        if T1 is not None : H = linalg.inv(T2) @ H @ T1
        return H

    def RanSac(self,S,trials=1000,t=None,T=None,s=8,initial_s=100,T1=None,T2=None,N=None,M=None,limit=1e6,return_t=False):
        X = array([list(x[:2]) + [1] for x in S])
        Y = array([list(x[2:]) + [1] for x in S])
        if T1 is not None : X = X @ T1.T
        if T2 is not None : Y = Y @ T2.T
        if t is None:
            ind = random.randint(0,len(S),min(initial_s,len(S)))
            H = self.findH(S[ind],T1=T1,T2=T2) # 40 is just an arbitary number..
            Ypred = X @ H.T
            Ypred = Ypred /Ypred[:,2][:,newaxis]
            d = (Ypred - Y)**2
            d =sum(d,axis=-1)
            t = mean(d)**0.5 #This is just a value I am setting arbitarily.
            d = d**0.5
            # I know, we should use_/5.99 sigma .. but we don't have any prior information of the system
            inliers = d < t
            inliers = [i for i in range(len(inliers)) if inliers[i]]
            #T = len(inliers)
            print("found params")
            # this also gives us w .. but without p, we can't use it to find N
            #indi = [ind]
            #cardSi = [len(inliers)]
        #else:
        #    indi=[]
        #    cardSi=[]
        indi = []
        cardSi = []
        #Split the image into boxes
        print("Running RanSac on S with |S|=",S.shape[0])
        for i in range(trials):
            #print(f"{i}th trial")
            ind = random.randint(0,len(S),s)
            Sit = S[ind]
            Hi =self.findH(Sit)
            #corners=array([[0,0,1],[0,M,1],[N,0,1],[N,M,1]])
            #corners = corners @ Hi.T
            #corners = corners/(corners[:,2][:,newaxis])
            #worst = amax(abs(corners))
            #print(worst)
            #if worst > limit:continue
            #print(X.shape,Hi.shape)
            Ypred = X @ Hi.T
            Ypred = Ypred /Ypred[:,2][:,newaxis]
            d = (Ypred - Y)**2
            d =sum(d,axis=-1)
            d =d**0.5
            inliers = d<t
            inliers = [j for j in range(len(inliers)) if inliers[j]]
            cardSi.append(len(inliers))
            indi.append(inliers)
            if T is not None and len(inliers) > T:
                Si = S[inliers]
                H = self.findH(Si)
                print(f"found on {i}th trial")
                return H
            if (i+1) % (trials//10) == 0 : print(f"{100*(i+1)//trials}% trials done")
        print("100% trials done")
        print("picking best")
        i = argmax(cardSi)
        ind = indi[i]
        Sbest =S[ind]
        H =self.findH(Sbest)
        if T1 is not None : H =linalg.inv(T2) @ H @ T1
        if return_t:return H,ind,t
        return H,ind

    def stitch_pair(self,
        img1,img2,
        H=None,
        origin=array([0,0,0]),
        epsilon=1e-11,
        t=None,T=None,s=8,initial_s=100,th=0.5,
        origin2=array([0,0,0]),
        correction_diameter=None,
        Xlim=2000,Ylim=1000,
        careful=False):
        """
        H : img1 - origin -> img2 - origin2
        """
        print(img1.shape,img2.shape)
        if H is None:
            H = self.Homo(img1,img2,th=th,t=t,T=T,s=s,initial_s=initial_s)
            origin = array([0,0,0])
            origin2 = array([0,0,0])
        rows,cols = img1.shape
        X = []
        I= []
        for y in range(cols):
            X.extend([[x,y,1] for x in range(rows)])
            I.extend([img1[x,y] for x in range(rows)])
        X = array(X)
        I = array(I)
        Ypred = (X-origin) @ H.T
        z =Ypred[:,2][:,newaxis]
        Ypred = Ypred/(epsilon*sign(z) + z)
        Ypred = Ypred + origin2
        Ypred = around(Ypred[:,:2])
        Ypred = int64(Ypred)
        left = min(Ypred[:,1])
        up = min(Ypred[:,0])
        right = max(Ypred[:,1])
        down = max(Ypred[:,0])
        left = min(left,0)
        up = min(up,0)
        right = max(right,img2.shape[1]-1)
        down = max(down,img2.shape[0]-1)
        #print(left,up,right,down)
        assert left <= 0,left
        assert up <= 0, up
        #print(img2.shape[0],-left)
        #print(-up,img2.shape[1])
        #print(img2.shape[0],right-img2.shape[1]+1)
        #print(down-img2.shape[0]+1,img2.shape[1])
        o = array([0,0,0])
        rightextra = right-img2.shape[1]+1
        if careful:
            o[1] = -left
            img2 = concatenate((
                zeros((img2.shape[0],-left),dtype=int8),
                img2,
                zeros((img2.shape[0],right-img2.shape[1]+1),dtype=int8)
                ),axis=1)
        elif -left + img2.shape[1] > Xlim and left<0 and right < img2.shape[1]:
            print("Killing some left columns")
            padleft = min(-left,max(Xlim-img2.shape[1],0))
            o[1] = padleft
            img2 = concatenate((
                zeros((img2.shape[0],padleft),dtype=int8),
                img2,
                zeros((img2.shape[0],right-img2.shape[1]+1),dtype=int8)
                ),axis=1)
        elif right > Xlim and left >= 0:
            print("Killing some right columns")
            o[1] =0
            img2 = concatenate((
                img2,
                zeros((img2.shape[0],min(right-img2.shape[1]+1,Xlim-img2.shape[1])),dtype=int8)
                ),axis=1)
        elif right - left > Xlim and left<0 and right > img2.shape[1]:# kill columns from both sides
            print("killing from both sides")
            o[1] = min(-left,(Xlim-img2.shape[1])//2)
            img2 = concatenate((
                zeros((img2.shape[0],min(-left,(Xlim-img2.shape[1])//2)),dtype=int8),
                img2,
                zeros((img2.shape[0],min(right-img2.shape[1]+1,(Xlim-img2.shape[1])//2)),dtype=int8)
                ),axis=1)
            print(min(-left,(Xlim-img2.shape[1])//2),min(right-img2.shape[1]+1,(Xlim-img2.shape[1])//2))
        else:
            print("normal")
            o[1] = -left
            img2 = concatenate((
                zeros((img2.shape[0],-left),dtype=int8),
                img2,
                zeros((img2.shape[0],right-img2.shape[1]+1),dtype=int8)
                ),axis=1)
        print("first concatenation done")
        #print("o is",o)
        downextra = down-img2.shape[0]+1
        if careful:
            o[0] = -up
            img2 = concatenate((
                zeros((-up,img2.shape[1]),dtype=int8),
                img2,
                zeros((down-img2.shape[0]+1,img2.shape[1]),dtype=int8)
                ),axis=0)
        elif down - up > Ylim:
            print("killing rows from both sides")
            pad=max(0,(Ylim-img2.shape[0])//2)
            o[0] = min(pad,-up)
            img2 = concatenate((
                zeros((min(pad,-up),img2.shape[1]),dtype=int8),
                img2,
                zeros((min(pad,down-img2.shape[0]+1),img2.shape[1]),dtype=int8)
                ),axis=0)
        else:
            print("normal")
            o[0] = -up
            img2 = concatenate((
                zeros((-up,img2.shape[1]),dtype=int8),
                img2,
                zeros((down-img2.shape[0]+1,img2.shape[1]),dtype=int8)
                ),axis=0)
        print("second concatenation done")
        n,m = img2.shape

        for i,x in enumerate(Ypred):
            if x[0]+o[0] >= n or x[0]+o[0]<0:continue
            if x[1]+o[1] >= m or x[1]+o[1]<0:continue
            try:img2[x[0]+o[0],x[1]+o[1]] =I[i]
            except:pass
        if correction_diameter is not None:img2 = cv2.medianBlur(img2,correction_diameter)
        return img2,o+origin2

if __name__=="__main__":
    Pan = PanaromaStitcher()
    imf = path = "ES666-Assignment3/Images/I3_down"
    im,Hom = Pan.make_panaroma_for_images_in(path,th=0.5,limit=1000,careful=False,t=1)
    plt.imshow(im)
    plt.show()