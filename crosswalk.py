# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:32:39 2020

@author: 33672
"""

import numpy as np
import cv2
import glob


def resize(img):
    #Variables
    width = 600
    height = 400
    dim = (width, height)
    #Resize
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  
    return img


def crop(img):
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        ## (2) Find the target green color region in HSV
        hsv_lower = np.array([50, 104, 141])
        hsv_upper = np.array([86, 245, 245])
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        ## (3) morph-op to remove horizone lines
        kernel = np.ones((5,1), np.uint8)
        mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        
        
        ## (4) crop the region
    
        ys, xs = np.nonzero(mask2)
    
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
    
        
        croped = img[ymin:ymax, xmin:xmax]
    
        
        pts = np.int32([[xmin, ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]])
        cv2.drawContours(img, [pts], -1, (0,255,0), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Feu vert',(xmin, ymin-10), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        
        cv2.imshow("croped", croped)
        cv2.imshow("img", img)

    except ValueError:
        pass


def passage_pieton(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 255) 

    cv2.imshow("Canny", canny)
    
    
def feu_pieton(frame):
    
    # Variables
    
    
    # Process
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
    
    # Green color
    low_green = np.array([50, 104, 141])
    high_green = np.array([86, 245, 245])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    
    # Application du masque   
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    
    th, proc = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY)
    
    


    cv2.imshow("Result", green)
    cv2.imshow("Feu vert", proc)
    

#-----------------------------------------
#                 MAIN
#-----------------------------------------


a = input('Appuyez sur "v" pour analyser des video, sinon appuyez sur "entrée"\n\nVotre choix ("entrée" pour valider):')

if (a=='v'):
    for video_path in glob.glob('video\*.mp4'):
        cap = cv2.VideoCapture(video_path)
        #cap = cv2.VideoCapture(0)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            if ret == True:
                
                #cv2.imshow("Original", resize(frame))
                #passage_pieton(resize(frame))
                #feu_pieton(resize(frame))
                crop(resize(frame))
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
            
        cap.release()
        cv2.destroyAllWindows()
else:
    for img_path in glob.glob('feux vert\*.jpg'):
        img = cv2.imread(img_path)
        #passage_pieton(resize(img))
        #feu_pieton(resize(img))
        crop(resize(img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    